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
        print(
            convert_motion(
                files.input, files.output, model_joint_names(model), model=model
            )
        )
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
            "K1 train/play requires --motion-file pointing to a 50 Hz K1 reference"
        )
    if not any(a in remaining for a in ("--help", "-h")):
        from . import REPO_ROOT
        from .motion import prepare_motion_files
        from .robot import get_k1_rev1_cfg

        motion_index = next(
            i
            for i, value in enumerate(remaining)
            if value == "--motion-file" or value.startswith("--motion-file=")
        )
        inline = remaining[motion_index].startswith("--motion-file=")
        if not inline and motion_index + 1 >= len(remaining):
            parser.error("--motion-file requires a reference path")
        source = (
            remaining[motion_index].split("=", 1)[1]
            if inline
            else remaining[motion_index + 1]
        )
        try:
            prepared = prepare_motion_files(
                source,
                get_k1_rev1_cfg().spec_fn().compile(),
                REPO_ROOT / ".supermimic-data" / "adapter",
            )
        except ValueError as error:
            parser.error(str(error))
        if inline:
            remaining[motion_index] = "--motion-file=" + prepared
        else:
            remaining[motion_index + 1] = prepared
        print(
            f"[INFO] Validated 50 Hz reference {source}; internal SuperMimic loader data: {prepared}"
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
