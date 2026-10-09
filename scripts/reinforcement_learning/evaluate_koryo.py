# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a Koryo checkpoint at fixed starts without clip-end teleport credit."""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import MethodType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import source.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--condition", choices=("train", "clean"), default="clean")
  parser.add_argument("--start-times", type=float, nargs="+", default=list(range(86)))
  parser.add_argument("--duration", type=float, default=30.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error("Checkpoint does not exist.")
  if not 0 < args.duration < float("inf") or any(not 0 <= t < 85.84 for t in args.start_times):
    parser.error("Duration must be positive and finite; starts must be in [0, 85.84).")
  configure_torch_backends()
  task = "Cyclo-Mimic-K1-Rev1-Koryo"
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = len(args.start_times)
  cfg.episode_length_s = args.duration
  cfg.seed = args.seed
  cmdcfg = cfg.commands["reference_trajectory"]
  cmdcfg.debug_vis = False
  if args.condition == "clean":
    cfg.events = {}
    cfg.observations["actor"].enable_corruption = False
    cmdcfg.reset_pose_noise = {}
    cmdcfg.reset_velocity_noise = {}
    cmdcfg.reset_joint_position_noise = (0., 0.)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    agentcfg = load_rl_cfg(task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agentcfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agentcfg), device=args.device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    cmd = env.command_manager.get_term("reference_trajectory")
    starts = torch.tensor(args.start_times, device=args.device).mul(cmd.reference.fps).round().long()
    if torch.any(starts >= cmd.reference.num_frames - 1):
      parser.error("Start time must precede the final reference frame.")
    original_sample = cmd._sample_start_frames

    def sample_fixed(self, env_ids):
      self.frame_ids[env_ids] = starts[env_ids]

    cmd._sample_start_frames = MethodType(sample_fixed, cmd)
    wrapped.reset()
    cmd._sample_start_frames = original_sample
    # Reset forwards physics but does not refresh aligned reference caches.
    cmd.frame_ids -= 1
    cmd._update_command()
    obs = wrapped.get_observations()
    records = {}
    terms = env.termination_manager
    original_compute = terms.compute
    end_names = cfg.terminations["reference_body_height"].params["body_names"]
    body_ids = [cmd.cfg.body_names.index(n) for n in end_names]

    def compute_record():
      done = original_compute()
      clip_end = cmd.frame_ids >= cmd.reference.num_frames - 1
      for i in (done | clip_end).nonzero().flatten().tolist():
        if i in records:
          continue
        errors = (cmd.aligned_body_position_w[i, body_ids, 2]
                  - cmd.robot_body_pos_w[i, body_ids, 2]).abs()
        failed = bool(terms.terminated[i])
        records[i] = {
          "start_s": float(starts[i]) / cmd.reference.fps,
          "duration_s": float(env.episode_length_buf[i]) * env.step_dt,
          "reference_s": float(cmd.frame_ids[i]) / cmd.reference.fps,
          "failed": failed,
          "reached_time_limit": bool(terms.time_outs[i]) and not failed,
          "reached_reference_end": bool(clip_end[i]) and not failed,
          "terminations": [n for n in terms.active_terms if bool(terms.get_term(n)[i])],
          "height_errors_m": dict(zip(end_names, errors.tolist())),
        }
      return done

    terms.compute = compute_record
    with torch.inference_mode():
      for _ in range(env.max_episode_length + 1):
        obs, _, _, _ = wrapped.step(policy(obs))
        if len(records) == env.num_envs:
          break
    rows = [records[i] for i in sorted(records)]
    summary = {
      "checkpoint": str(args.checkpoint.resolve()), "condition": args.condition,
      "seed": args.seed, "episode_cap_s": args.duration,
      "episodes": len(rows),
      "mean_duration_s": sum(r["duration_s"] for r in rows) / len(rows),
      "time_limit_successes": sum(r["reached_time_limit"] for r in rows),
      "reference_end_successes": sum(r["reached_reference_end"] for r in rows),
      "failures": sum(r["failed"] for r in rows),
      "termination_counts": dict(Counter(t for r in rows for t in r["terminations"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
