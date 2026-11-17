"""Task-specific isolated entry point for the released recipe (never bare defaults)."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from .runtime import ROOT, VENDOR
from .config import TASKS
from .prepare import resolve_motion_files

TASK = "Cyclo-BlinkMimic-K1-Rev1-001"


def latest_training_checkpoint(log_root, task):
    candidates = []
    for record_path in (log_root / task).glob("*/run-record.json"):
        try:
            record = json.loads(record_path.read_text())
            if record["task"] != task or record["action"] != "train":
                continue
            started_at = datetime.datetime.fromisoformat(record["started_at"]).timestamp()
            checkpoint = select_checkpoint(None, record_path.parent)
        except (OSError, ValueError, KeyError, TypeError):
            # In-progress records and runs without a checkpoint are not playable yet.
            continue
        candidates.append((started_at, checkpoint))
    if not candidates:
        raise ValueError(
            f"No saved training checkpoints for {task} in {log_root}; "
            "use --checkpoint-dir or --checkpoint for a run outside this directory"
        )
    return max(candidates)[1]


def training_motion_directory(checkpoint):
    record_path = checkpoint.parent / "run-record.json"
    try:
        record = json.loads(record_path.read_text())
        motion_dir = Path(record["manifest"]).parent
        if not motion_dir.is_dir():
            # Host and container share the checkout but mount it at different paths.
            motion_dir = ROOT / motion_dir.relative_to(record["cwd"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Cannot recover training motion directory from {record_path}; use --motion-dir"
        ) from error
    return motion_dir


def select_checkpoint(checkpoint, checkpoint_dir):
    if checkpoint is None:
        if checkpoint_dir is None or not checkpoint_dir.is_dir():
            raise ValueError("play requires --checkpoint or an existing --checkpoint-dir")
        candidates = [
            path for path in checkpoint_dir.iterdir()
            if path.is_file() and re.fullmatch(r"ckpt_[0-9]{6}\.pt", path.name)
        ]
        if not candidates:
            raise ValueError(f"No ckpt_NNNNNN.pt checkpoints in {checkpoint_dir}")
        checkpoint = max(candidates, key=lambda path: int(path.stem[5:]))
    if not checkpoint.is_file() or not re.fullmatch(r"ckpt_[0-9]{6}\.pt", checkpoint.name):
        raise ValueError(f"Expected an existing ckpt_NNNNNN.pt checkpoint: {checkpoint}")
    return checkpoint.resolve()


def export_training_config_when_ready(child, run_dir, robot):
    """Use the trainer's saved arguments without modifying its parser or runtime."""
    from .runtime import prepare

    while True:
        try:
            settings = json.loads((run_dir / "args.json").read_text())
            break
        except (FileNotFoundError, json.JSONDecodeError):
            # args.json is written in place; retry while the trainer starts or writes.
            if child.poll() is not None:
                raise RuntimeError("Trainer exited before saving complete args.json")
            try:
                child.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass

    prepare()
    from .export_sim2real_cfg import export_sim2real_cfg

    if settings["robot"] != robot.name:
        raise ValueError("Training robot does not match the deployment configuration")
    return export_sim2real_cfg(run_dir, SimpleNamespace(**settings), robot)


def recipe():
    text = (VENDOR / "scripts/presets.sh").read_text()
    return shlex.split(re.search(r'\[main\]="([^"]+)"', text)[1])


def execution_recipe(mode):
    """An explicit systems-only alternative; model and learning settings are intact."""
    flags = recipe()
    removed = []
    if mode == "compiled_no_torch_graphs":
        switches = {
            "--cuda_graph",
            "--graph_early",
            "--collect_graph",
            "--reset_masked",
            "--eprobe_graph",
        }
        result = []
        i = 0
        while i < len(flags):
            flag = flags[i]
            if flag in switches:
                removed.append(flag)
            elif flag == "--eprobe_async":
                removed.extend(flags[i : i + 2])
                i += 1
            else:
                result.append(flag)
            i += 1
        flags = result
    return flags, removed


def prepare_task_motions(task, output):
    from .runtime import prepare

    files = resolve_motion_files(TASKS[task].motion_paths)
    prepare()
    from .prepare import prepare_library

    prepare_library(files, output, TASKS[task].robot_factory().joint_names)
    return output


def worker_environment(cache=None):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), WANDB_MODE="disabled",
               CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4")
    if cache is not None:
        env.update(
            MPLCONFIGDIR=str(cache / "matplotlib"),
            TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"),
            TRITON_CACHE_DIR=str(cache / "triton"),
            XDG_CACHE_HOME=str(cache),
        )
    return env


def run_gui(args):
    command = [sys.executable, "-u", "-m", "source.tasks.blinkmimic.worker", "play",
               "--task", args.task,
               "--checkpoint", str(args.checkpoint), "--motion-dir",
               str(args.motion_dir.resolve()), "--port", str(args.port)]
    if args.num_steps is not None:
        command.extend(["--num-steps", str(args.num_steps)])
    cache = args.cache_dir.resolve() if args.cache_dir else None
    return subprocess.run(command, cwd=ROOT, env=worker_environment(cache)).returncode


def main(action="train", argv=None):
    parser = argparse.ArgumentParser(
        description="K1 Rev.1 motion-only official BlinkMimic"
    )
    parser.add_argument("task", choices=tuple(TASKS))
    parser.add_argument("--motion-dir", type=Path,
                        help="Override the task motions with a prepared library; play defaults to recorded motions")
    parser.add_argument("--run-dir", type=Path,
                        help="Output directory; training defaults to logs/<task>/<timestamp>")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="Training directory; defaults to the latest saved run for this task")
    parser.add_argument("--viewer", choices=("viser", "headless"), default="viser")
    parser.add_argument("--port", type=int, default=8080, help="Browser viewer port")
    parser.add_argument("--num-steps", type=int, help="Stop GUI playback after this many policy steps")
    parser.add_argument(
        "--execution",
        choices=("official", "compiled_no_torch_graphs"),
        default="official",
        help="Explicit systems-only memory alternative; preserves models/batches/horizons. Native MuJoCo physics graphs remain.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Task-local compiler cache shared across isolated runs",
    )
    args, extra = parser.parse_known_args(argv)
    if action not in ("train", "play"):
        parser.error("Supported actions are train and play")
    # Recipe overrides remain explicit and recorded, with no implicit resource reduction.
    forbidden = {"--robot", "--motion_dir", "--log_dir", "--run_dir", "--ckpt"}
    if any(x.split("=", 1)[0] in forbidden for x in extra):
        parser.error("Use task-specific motion/run/checkpoint options")
    if action == "train" and args.run_dir is None:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.run_dir = ROOT / "logs" / args.task / stamp
    if action == "play":
        try:
            if args.checkpoint is None and args.checkpoint_dir is None:
                args.checkpoint = latest_training_checkpoint(ROOT / "logs", args.task)
            else:
                args.checkpoint = select_checkpoint(args.checkpoint, args.checkpoint_dir)
            if args.motion_dir is None:
                args.motion_dir = training_motion_directory(args.checkpoint)
        except ValueError as error:
            parser.error(str(error))
        if args.num_steps is not None and args.num_steps < 1:
            parser.error("--num-steps must be positive")
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        print(f"[INFO] Selected checkpoint: {args.checkpoint}", flush=True)
        print(f"[INFO] Motion directory: {args.motion_dir}", flush=True)
    if args.motion_dir is not None and not (args.motion_dir / "manifest.json").is_file():
        parser.error("Motion directory requires its provenance manifest.json; use --motion-dir")
    if action == "play" and args.viewer == "viser":
        if extra:
            parser.error(f"Unsupported GUI options: {' '.join(extra)}")
        if args.run_dir is not None:
            parser.error("--run-dir is only used for training or headless evaluation")
        raise SystemExit(run_gui(args))
    if args.run_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.run_dir = args.checkpoint.parent / "play" / stamp
    from .runtime import prepare

    prepare()
    robot = TASKS[args.task].robot_factory()
    out = args.run_dir.resolve()
    effective_recipe, removed_systems_flags = execution_recipe(args.execution)
    if action == "train":
        out.mkdir(parents=True, exist_ok=False)
        if args.motion_dir is None:
            args.motion_dir = prepare_task_motions(args.task, out / "motions")
        flags = [
            *effective_recipe,
            *extra,
            "--robot",
            robot.name,
            "--motion_dir",
            str(args.motion_dir.resolve()),
            "--log_dir",
            str(out),
        ]
        mode = "train"
    else:
        name = args.checkpoint.name
        out.mkdir(parents=True, exist_ok=False)
        # The upstream one-shot directory evaluator retains per-clip JSON and CSV.
        (out / name).symlink_to(args.checkpoint.resolve())
        flags = [
            *extra,
            "--robot",
            robot.name,
            "--motion_dir",
            str(args.motion_dir.resolve()),
            "--run_dir",
            str(out),
        ]
        mode = "evaluate"
        print(f"[INFO] Playback log: {out / 'console.log'}", flush=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "source.tasks.blinkmimic.worker",
        mode,
        "--task", args.task,
        *flags,
    ]
    cache = args.cache_dir.resolve() if args.cache_dir else out / "cache"
    env = worker_environment(cache)
    record = dict(
        task=args.task,
        action=action,
        command=command,
        cwd=str(ROOT),
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        repository_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        upstream=json.loads((VENDOR / "UPSTREAM.json").read_text())["commit"],
        manifest=str(args.motion_dir.resolve() / "manifest.json"),
        checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
        manifest_sha256=hashlib.sha256(
            (args.motion_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        runtime={
            name: importlib.metadata.version(name)
            for name in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang")
        },
        python=sys.version,
        compiler_cache=str(cache),
        source_status=subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        recipe=recipe(),
        execution=args.execution,
        removed_systems_flags=removed_systems_flags,
        explicit_overrides=extra,
        status="starting",
        stop="Verify child PID and command, then SIGINT that PID only. No automatic restart.",
    )
    record_path = out / "run-record.json"
    record_path.write_text(json.dumps(record, indent=2))
    with (out / "console.log").open("x") as log:
        child = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        record.update(pid=child.pid, status="running")
        record_path.write_text(json.dumps(record, indent=2))
        if action == "train":
            try:
                config_path = export_training_config_when_ready(child, out, robot)
                print(f"[INFO] Exported sim2real config: {config_path}", file=log, flush=True)
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                record.update(status="failed", returncode=child.returncode,
                              finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
                record_path.write_text(json.dumps(record, indent=2))
                raise
        code = child.wait()
    record.update(
        status="completed" if code == 0 else "interrupted" if code == -2 else "failed",
        returncode=code,
        finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    record_path.write_text(json.dumps(record, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main(argv=sys.argv[1:])
