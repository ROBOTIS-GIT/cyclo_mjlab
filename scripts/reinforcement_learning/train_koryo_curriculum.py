# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Train taekwondo motions in blocks, releasing GPU memory before full-motion evaluations.

Use this entry point for automatic promotion. Plain train.py keeps the loaded
stage fixed. Checkpoints retain PPO/optimizer state; simulator episodes restart
between blocks. The input checkpoint is never overwritten.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from source.tasks.mimic.koryo_schedule import (
  MIN_EVAL_EPISODES, MIN_EVAL_SEEDS, migrate_state, record_full_evaluation,
)
from source.tasks.mimic.motion_curriculum import MOTIONS, motion_identity, validate_checkpoint_motion


def training_task(args):
  if args.contact_v2:
    return 'Cyclo-Mimic-K1-Rev1-Koryo-ContactV2-Curriculum'
  return MOTIONS[args.motion].training_task


def validate_contact_checkpoint(infos):
  if infos.get('koryo_performance', {}).get('version') != 2:
    raise ValueError('Contact curriculum requires a ContactV2 checkpoint.')
  steps = infos.get('contact_training_steps')
  if type(steps) is not int or steps < 0:
    raise ValueError('Invalid ContactV2 training step counter.')


def write_json(path, value):
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(value, indent=2) + "\n")
  tmp.replace(path)


def save_checkpoint(path, checkpoint):
  tmp = path.with_suffix(".tmp")
  torch.save(checkpoint, tmp)
  tmp.replace(path)


def copy_checkpoint(source, destination):
  tmp = destination.with_suffix(".tmp")
  shutil.copyfile(source, tmp)
  tmp.replace(destination)


def run_logged(command, path, tee=False):
  with path.open("w") as log:
    if tee:
      process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
      try:
        for line in process.stdout:
          log.write(line)
          log.flush()
          print(line, end="", flush=True)
        returncode = process.wait()
      except BaseException:
        process.terminate()
        try:
          process.wait(timeout=10)
        except subprocess.TimeoutExpired:
          process.kill()
          process.wait()
        raise
      finally:
        process.stdout.close()
    else:
      returncode = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
  if returncode:
    raise RuntimeError(f"Process exited {returncode}; see {path}")


def evaluate(checkpoint, directory, name, iteration, noise, push, trials, device, motion="koryo"):
  summaries = []
  for index in range(MIN_EVAL_SEEDS):
    # Fresh, reproducible seeds at each evaluated checkpoint; a resumed run
    # continues the schedule instead of repeatedly optimizing the same seeds.
    seed = 100_001 + 3 * iteration + index
    output = directory / f"{name}_{seed}.json"
    run_logged([
      sys.executable, str(ROOT / "scripts/reinforcement_learning/evaluate_koryo.py"),
      "--motion", motion,
      "--checkpoint", str(checkpoint), "--condition", "train", "--full-motion",
      "--trials", str(trials), "--noise-scale", str(noise), "--push-scale", str(push),
      "--seed", str(seed), "--device", device, "--output", str(output),
    ], output.with_suffix(".log"))
    summary = json.loads(output.read_text())["summary"]
    if summary["episodes"] != trials or summary["motion"] != motion:
      raise RuntimeError(f"Incomplete evaluation in {output}")
    summaries.append(summary)
  return summaries


def aggregate(summaries):
  n = sum(s["episodes"] for s in summaries)
  return {
    "episodes": n,
    "success_rate": sum(s["reference_end_successes"] for s in summaries) / n,
    "mean_body_error_m": sum(s["mean_body_error_m"] * s["episodes"] for s in summaries) / n,
    "mean_anchor_error_m": sum(s["mean_anchor_error_m"] * s["episodes"] for s in summaries) / n,
  }


def train_worker(args):
  # Each worker owns its CUDA context. Parent/evaluation workers never keep
  # a training environment on the GPU concurrently.
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  os.environ["MUJOCO_GL"] = "egl"
  import source.tasks  # noqa: F401
  from train import TrainConfig, run_train
  task = training_task(args)
  cfg = TrainConfig.from_task(task)
  cfg.env.scene.num_envs = args.num_envs
  cfg.agent.max_iterations = args.additional_iterations
  cfg.agent.save_interval = min(500, args.additional_iterations)
  cfg.agent.upload_model = False
  cfg.agent.seed = args.seed
  # Explicit path bypasses the regex selector and is safe across run folders.
  from dataclasses import replace
  cfg = replace(cfg, checkpoint=str(args.checkpoint) if args.checkpoint else None)
  run_train(task, cfg, args.output_dir)


def main(default_motion="koryo", default_contact_v2=False):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion", choices=tuple(MOTIONS), default=default_motion)
  parser.add_argument('--contact-v2', action='store_true', default=default_contact_v2,
                      help='Preserve ContactV2 rewards, optimizer and contact ramp during curriculum training.')
  parser.add_argument("--checkpoint", type=Path, help="Resume this motion's checkpoint; omit to train from scratch.")
  parser.add_argument("--additional-iterations", type=int, default=30_000)
  parser.add_argument("--eval-every", type=int, default=1000)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--eval-trials-per-seed", type=int, default=32)
  parser.add_argument("--gpu", type=int, default=0, help="Physical CUDA device index.")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--quiet-training", action="store_true", help="Write iteration logs only to files.")
  parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.contact_v2 and (args.motion != 'koryo' or args.checkpoint is None):
    parser.error('--contact-v2 requires --motion koryo and a ContactV2 --checkpoint.')
  if args.checkpoint is not None:
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
      parser.error("Checkpoint does not exist.")
  if (args.additional_iterations < 1 or args.eval_every < 1 or args.num_envs < 10
      or args.eval_trials_per_seed * MIN_EVAL_SEEDS < MIN_EVAL_EPISODES or args.gpu < 0):
    parser.error("Positive budgets, >=10 training envs, >=32 trials/seed and a nonnegative GPU index are required.")
  if args.worker:
    train_worker(args)
    return
  # Load on CPU only; do not allocate a parent CUDA context.
  spec = MOTIONS[args.motion]
  identity = motion_identity(args.motion)
  checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
  infos = dict(checkpoint.get("infos") or {}) if checkpoint else {}
  if checkpoint:
    validate_checkpoint_motion(infos, args.motion, identity)
  if args.contact_v2:
    validate_contact_checkpoint(infos)
  elif infos.get(args.motion + '_performance'):
    parser.error('Use the matching contact training task to retain contact rewards; this ordinary curriculum cannot resume a contact checkpoint.')
  state = migrate_state(infos.get(spec.state_key))
  iteration = int(checkpoint["iter"]) if checkpoint else -1
  if args.dry_run:
    print(json.dumps({
      "checkpoint": str(args.checkpoint) if args.checkpoint else None,
      "motion": identity, "training_task": training_task(args),
      "last_completed_iteration": iteration,
      "curriculum": state, "additional_iterations": args.additional_iterations,
      "eval_every": args.eval_every, "eval_episodes_per_condition": args.eval_trials_per_seed * 3,
      "conditions": ["current stage", "100% disturbances"],
      "promotion": "90% full-motion completion twice consecutively; final stage stops training",
    }, indent=2))
    return
  if state["mastered"]:
    print("[DONE] This checkpoint already passed the final full-motion gate.")
    return
  run_dir = (args.output_dir or ROOT / "logs/rsl_rl/k1_mimic" /
             (datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") +
              ("_koryo_contact_v2_curriculum" if args.contact_v2 else f"_{args.motion}_curriculum_v2"))).resolve()
  run_dir.mkdir(parents=True, exist_ok=False)
  # Physical GPU selection is shared by all short-lived workers.
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  os.environ["MUJOCO_GL"] = "egl"
  latest = None
  if checkpoint:
    infos[spec.state_key] = state
    infos["motion_identity"] = identity
    checkpoint["infos"] = infos
    latest = run_dir / "resume.pt"
    save_checkpoint(latest, checkpoint)
  del checkpoint
  best = infos.get(spec.best_key)
  if best and Path(best["checkpoint"]).is_file():
    copy_checkpoint(Path(best["checkpoint"]), run_dir / "best_full.pt")
    best = {**best, "checkpoint": str(run_dir / "best_full.pt")}
  else:
    best = None
  progress = {
    "status": "training", "input_checkpoint": str(args.checkpoint) if args.checkpoint else None,
    "motion": identity, "latest_checkpoint": str(latest) if latest else None,
    "completed_additional_iterations": 0,
    "curriculum": state, "best_full": best,
  }
  write_json(run_dir / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
  write_json(run_dir / "status.json", progress)
  print(f"[INFO] Run directory: {run_dir}", flush=True)
  print(f"[INFO] Starting noise={state['noise_scale']:.0%}, push={state['push_scale']:.0%}", flush=True)
  remaining = args.additional_iterations
  try:
    while remaining and not state["mastered"]:
      block = min(args.eval_every, remaining)
      next_iteration = iteration + block
      progress["status"] = "training"
      write_json(run_dir / "status.json", progress)
      print(f"[TRAIN] {block} updates through iteration {next_iteration}", flush=True)
      run_logged([
        sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--motion", args.motion,
        *(['--contact-v2'] if args.contact_v2 else []),
        *(["--checkpoint", str(latest)] if latest else []), "--additional-iterations", str(block),
        "--num-envs", str(args.num_envs), "--gpu", str(args.gpu),
        "--seed", str((args.seed + max(iteration, 0)) % (2**31)), "--output-dir", str(run_dir),
      ], run_dir / f"train_{next_iteration}.log", tee=not args.quiet_training)
      trained = run_dir / f"model_{next_iteration}.pt"
      if not trained.is_file():
        raise RuntimeError(f"Training did not produce {trained}")
      iteration = next_iteration
      remaining -= block
      progress.update(status="evaluating", latest_checkpoint=str(trained),
                      completed_additional_iterations=args.additional_iterations-remaining)
      write_json(run_dir / "status.json", progress)
      evaluation_dir = run_dir / "evaluations" / str(iteration)
      evaluation_dir.mkdir(parents=True)
      print(f"[EVAL] Full motion at push={state['push_scale']:.0%} and at 100%", flush=True)
      current = evaluate(trained, evaluation_dir, "current", iteration,
                         state["noise_scale"], state["push_scale"], args.eval_trials_per_seed, "cuda:0", motion=args.motion)
      full = current if state["noise_scale"] == state["push_scale"] == 1 else evaluate(
        trained, evaluation_dir, "full", iteration, 1.0, 1.0, args.eval_trials_per_seed, "cuda:0", motion=args.motion)
      current_metrics, full_metrics = aggregate(current), aggregate(full)
      # First key is complete-motion survival; body error breaks exact ties.
      score = (full_metrics["success_rate"], -full_metrics["mean_body_error_m"])
      new_best = best is None or score > tuple(best["score"])
      if new_best:
        best = {"score": score, "iteration": iteration, "metrics": full_metrics,
                "checkpoint": str(run_dir / "best_full.pt")}
      checkpoint = torch.load(trained, map_location="cpu", weights_only=False)
      validate_checkpoint_motion(checkpoint["infos"], args.motion, identity)
      state = record_full_evaluation(checkpoint["infos"][spec.state_key], current, iteration)
      checkpoint["infos"].update({spec.state_key: state, spec.best_key: best, "motion_identity": identity})
      evaluated = run_dir / f"evaluated_{iteration}.pt"
      save_checkpoint(evaluated, checkpoint)
      del checkpoint
      latest = run_dir / "resume.pt"
      copy_checkpoint(evaluated, latest)
      if new_best:
        copy_checkpoint(evaluated, run_dir / "best_full.pt")
      write_json(evaluation_dir / "summary.json", {
        "iteration": iteration, "current": current_metrics, "full": full_metrics,
        "next_curriculum": state,
      })
      progress.update(latest_checkpoint=str(latest), curriculum=state, best_full=best,
                      last_evaluation=str(evaluation_dir / "summary.json"))
      write_json(run_dir / "status.json", progress)
      print(f"[RESULT] current={current_metrics['success_rate']:.1%}, full={full_metrics['success_rate']:.1%}; "
            f"next push={state['push_scale']:.0%}, pass streak={state['evaluation_streak']}, mastered={state['mastered']}", flush=True)
    progress["status"] = "mastered" if state["mastered"] else "budget_exhausted"
    write_json(run_dir / "status.json", progress)
    print(f"[DONE] {progress['status']}; resume: {latest}; best full: {run_dir / 'best_full.pt'}", flush=True)
  except BaseException as error:
    progress.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
    write_json(run_dir / "status.json", progress)
    raise


if __name__ == "__main__":
  main()
