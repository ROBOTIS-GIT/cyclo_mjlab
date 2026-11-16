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

from .runtime import ROOT, VENDOR

TASK = "Cyclo-BlinkMimic-K1-Rev1"


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


def main(action="train", argv=None):
    parser = argparse.ArgumentParser(
        description="K1 Rev.1 motion-only official BlinkMimic"
    )
    parser.add_argument("task", choices=[TASK])
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
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
        parser.error("Supported actions are train and play (headless evaluation)")
    # Recipe overrides remain explicit and recorded, with no implicit resource reduction.
    forbidden = {"--robot", "--motion_dir", "--log_dir", "--run_dir", "--ckpt"}
    if any(x.split("=", 1)[0] in forbidden for x in extra):
        parser.error("Use task-specific motion/run/checkpoint options")
    if not (args.motion_dir / "manifest.json").is_file():
        parser.error("Motion directory requires its provenance manifest.json")
    out = args.run_dir.resolve()
    effective_recipe, removed_systems_flags = execution_recipe(args.execution)
    if action == "train":
        out.mkdir(parents=True, exist_ok=False)
        flags = [
            *effective_recipe,
            *extra,
            "--robot",
            "k1",
            "--motion_dir",
            str(args.motion_dir.resolve()),
            "--log_dir",
            str(out),
        ]
        mode = "train"
    else:
        if args.checkpoint is None or not args.checkpoint.is_file():
            parser.error(
                "play requires an existing --checkpoint (headless all-clip evaluation)"
            )
        name = args.checkpoint.name
        if not re.fullmatch(r"ckpt_[0-9]{6}\.pt", name):
            parser.error("Evaluation requires an official ckpt_NNNNNN.pt checkpoint")
        out.mkdir(parents=True, exist_ok=False)
        # The upstream one-shot directory evaluator retains per-clip JSON and CSV.
        (out / name).symlink_to(args.checkpoint.resolve())
        flags = [
            *extra,
            "--robot",
            "k1",
            "--motion_dir",
            str(args.motion_dir.resolve()),
            "--run_dir",
            str(out),
        ]
        mode = "evaluate"
    command = [
        sys.executable,
        "-u",
        "-m",
        "source.tasks.blinkmimic.worker",
        mode,
        *flags,
    ]
    cache = args.cache_dir.resolve() if args.cache_dir else out / "cache"
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT),
        WANDB_MODE="disabled",
        CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="4",
        MPLCONFIGDIR=str(cache / "matplotlib"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        XDG_CACHE_HOME=str(cache),
    )
    record = dict(
        task=TASK,
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
