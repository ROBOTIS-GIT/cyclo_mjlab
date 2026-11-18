"""Shared command-line options and runtime checks for train/play entry points."""

import argparse
import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from scripts.reinforcement_learning.dispatch import task_family

FAMILIES = ("blinkmimic", "beyondmimic", "supermimic", "xmimic")


def task_names(family):
    if family == "beyondmimic":
        import source.tasks  # noqa: F401
        from mjlab.tasks.registry import list_tasks

        return tuple(task for task in list_tasks() if task.startswith("Cyclo-BeyondMimic-"))
    return tuple(import_module(f"source.tasks.{family}.config").TASKS)


def require_runtime():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Cyclo requires Python 3.12 for training and playback")
    try:
        actual = version("mjlab")
    except PackageNotFoundError:
        actual = "not installed"
    if actual != "1.6.1":
        raise SystemExit(
            f"Cyclo requires mjlab 1.6.1; found {actual}. "
            "Use the locked Python 3.12 runtime described in docs/supermimic.md"
        )


def prepare_args(action, argv=None):
    """Handle shared options and validate the runtime before backend imports."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--algorithm", choices=FAMILIES)
    parser.add_argument("--list-tasks", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    task = next((a for a in remaining if a.startswith("Cyclo-")), None)
    if (
        args.list_tasks
        or task is None
        and any(a in remaining for a in ("-h", "--help"))
    ):
        print(
            f"Usage: python scripts/reinforcement_learning/{action}.py [--algorithm FAMILY] TASK [options]"
        )
        for family in FAMILIES:
            if args.algorithm is None or args.algorithm == family:
                print(f"{family}: " + ", ".join(task_names(family)))
        if args.algorithm is None:
            print("velocity: Cyclo-Velocity-Flat-K1-Rev1-v0")
        print(
            "BeyondMimic and SuperMimic share the locked MJLab 1.6.1 / Python 3.12 runtime."
        )
        return None
    family = task_family(task or "")
    if args.algorithm and family != args.algorithm:
        parser.error(f"--algorithm {args.algorithm} does not match task {task!r}")
    require_runtime()
    return remaining
