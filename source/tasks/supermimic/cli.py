"""
Native task entry point; forwards training/play to the official shared learner.
"""

import argparse
import sys


def convert_motion_command(argv):
    from .reference.motion import convert_motion, model_joint_names
    from source.assets.robots.ai_sapiens.k1_rev1 import get_k1_rev1_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    files = parser.parse_args(argv)
    model = get_k1_rev1_cfg().spec_fn().compile()
    print(
        convert_motion(
            files.input, files.output, model_joint_names(model), model=model
        )
    )


def resolve_motion_argument(arguments, env_cfg, parser):
    # Reject duplicates before rewriting paths: Tyro would otherwise accept the
    # final value, including an unvalidated option with the alternate spelling.
    options = [
        i
        for i, value in enumerate(arguments)
        if value.split("=", 1)[0].replace("_", "-") == "--motion-file"
    ]
    if len(options) > 1:
        parser.error(
            "Use at most one --motion-file option; use a comma list or glob"
            " for multiple clips"
        )
    if not options:
        from source.utils.motion_paths import resolve_motion_files

        motion_paths = env_cfg.commands["motion"].motion_file
        if isinstance(motion_paths, str):
            motion_paths = (motion_paths,)
        source = ",".join(
            str(path) for path in resolve_motion_files(motion_paths)
        )
        index = len(arguments)
        arguments.extend(["--motion-file", source])
    else:
        index = options[0]
    inline = "=" in arguments[index]
    if not inline and index + 1 >= len(arguments):
        parser.error("--motion-file requires a reference path")
    source = (
        arguments[index].split("=", 1)[1] if inline else arguments[index + 1]
    )
    return index, source, inline


def prepare_task_motion_arguments(arguments, parser):
    from mjlab.tasks.registry import load_env_cfg
    from . import REPO_ROOT
    from .reference.motion import prepare_motion_files

    env_cfg = load_env_cfg(arguments[0])
    try:
        index, source, inline = resolve_motion_argument(
            arguments, env_cfg, parser
        )
        prepared = prepare_motion_files(
            source,
            env_cfg.scene.entities["robot"].spec_fn().compile(),
            REPO_ROOT / ".supermimic-data" / "adapter",
        )
    except ValueError as error:
        parser.error(str(error))
    if inline:
        arguments[index] = "--motion-file=" + prepared
    else:
        arguments[index] = "--motion-file"
        arguments[index + 1] = prepared
    print(
        f"[INFO] Validated 50 Hz reference {source}; internal SuperMimic"
        f" loader data: {prepared}"
    )


def run_task(command, arguments, parser):
    from .config import TASKS

    if not arguments or arguments[0] not in TASKS:
        parser.error("Pass a Cyclo task ID: " + ", ".join(TASKS))
    if not any(arg in arguments for arg in ("--help", "-h")):
        prepare_task_motion_arguments(arguments, parser)
    if command == "train":
        from .train import main as task_main
    else:
        from .play import main as task_main
    task_main(arguments)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "command", choices=("train", "play", "convert", "smoke")
    )
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return
    args, remaining = parser.parse_known_args(argv)
    if args.command == "convert":
        convert_motion_command(remaining)
    elif args.command == "smoke":
        from .smoke import main as smoke_main

        smoke_main(remaining)
    else:
        run_task(args.command, remaining, parser)


if __name__ == "__main__":
    main()
