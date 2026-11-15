"""Separate runtime entry point; forwards training/play to the real upstream runner."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("command", choices=("train", "play", "convert", "smoke"))
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        return
    args, remaining = parser.parse_known_args()
    if args.command == "convert":
        from .motion import convert_motion, model_joint_names
        from .tasks import get_k1_rev1_cfg

        p = argparse.ArgumentParser()
        p.add_argument("input")
        p.add_argument("output")
        files = p.parse_args(remaining)
        model = get_k1_rev1_cfg().spec_fn().compile()
        print(convert_motion(files.input, files.output, model_joint_names(model)))
        return
    if args.command == "smoke":
        from .smoke import main as smoke_main

        smoke_main(remaining)
        return
    from .tasks import TASKS, register_tasks

    if not remaining or remaining[0] not in TASKS:
        parser.error("Pass a Cyclo task ID: " + ", ".join(TASKS))
    # Never silently download a G1 reference motion for a K1 task.
    if (
        not any(a in remaining for a in ("--help", "-h"))
        and "--motion-file" not in remaining
        and not any(a.startswith("--motion-file=") for a in remaining)
    ):
        parser.error(
            "K1 train/play requires --motion-file pointing to a converted K1 clip"
        )
    if not any(a in remaining for a in ("--help", "-h")):
        import glob
        import os

        import numpy as np

        from .motion import model_joint_names, validate_motion
        from .robot import get_k1_rev1_cfg

        motion_arg = next(
            (a.split("=", 1)[1] for a in remaining if a.startswith("--motion-file=")),
            None,
        )
        if motion_arg is None:
            motion_arg = remaining[remaining.index("--motion-file") + 1]
        names = model_joint_names(get_k1_rev1_cfg().spec_fn().compile())
        files = [
            file
            for pattern in motion_arg.split(",")
            for file in sorted(glob.glob(os.path.expanduser(pattern.strip())))
        ]
        if not files:
            parser.error(f"No local K1 motion files matched {motion_arg!r}")
        for file in files:
            with np.load(file, allow_pickle=False) as motion:
                validate_motion(motion, names)
                if tuple(motion["joint_names"]) != names:
                    parser.error(
                        f"{file}: convert first to put joints in the robot's order"
                    )
    register_tasks()
    sys.argv = [sys.argv[0], *remaining]
    if args.command == "train":
        from supermimic.scripts.train import main as upstream_main
    else:
        from supermimic.scripts.play import main as upstream_main
    upstream_main()


if __name__ == "__main__":
    main()
