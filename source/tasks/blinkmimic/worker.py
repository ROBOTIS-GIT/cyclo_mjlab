"""Private subprocess boundary for upstream MJLab performance patches."""

import argparse
import json
import runpy
import sys
import traceback
from pathlib import Path

from .runtime import install_robot, prepare
from .config import TASKS


def record_training_failure(error):
    # Read only Python bookkeeping from the failing frame; do not touch a
    # failed
    # CUDA context or restart. Preserve the exact loop index absent from sparse
    # logs.
    tb = error.__traceback__
    state = {}
    while tb:
        frame = tb.tb_frame
        if (
            frame.f_code.co_filename.endswith("/blinkmimic/train.py")
            and "args" in frame.f_globals
        ):
            g = frame.f_globals
            it = g.get("it")
            state = {
                "loop_index_zero_based": it,
                "last_completed_iterations": it,
                "wm_update_calls": getattr(
                    g.get("wm_update_g"), "calls", None
                ),
                "policy_update_calls": getattr(
                    g.get("policy_update_g"), "calls", None
                ),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            out = Path(g["args"].log_dir)
        tb = tb.tb_next
    if state:
        (out / "failure.json").write_text(json.dumps(state, indent=2))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("mode", choices=("train", "evaluate", "play"))
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--resume")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.resume:
        sys.argv.extend(["--resume", args.resume])
    prepare()
    robot = TASKS[args.task].robot_factory()
    install_robot(robot)
    try:
        if args.mode == "play":
            from .play import main as play

            play(robot=robot)
        elif args.mode == "train":
            from .checkpoint_adapter import include_critic_in_snapshots

            if "--ckpt_light" in remaining:
                raise ValueError(
                    "--ckpt_light is incompatible with critic snapshots"
                )
            from .resume import allow_model_only_resume

            with (
                include_critic_in_snapshots(),
                allow_model_only_resume(args.resume),
            ):
                runpy.run_module("blinkmimic.train", run_name="__main__")
        else:
            runpy.run_module(f"blinkmimic.{args.mode}", run_name="__main__")
    except BaseException as error:
        record_training_failure(error)
        raise


if __name__ == "__main__":
    main()
