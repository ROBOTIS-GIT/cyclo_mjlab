"""Select an algorithm within the common native MJLab runtime.

No installation, process launch, or environment mutation is performed. Native
mjlab task IDs infer the family; --algorithm makes that selection explicit.
"""

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

TASKS = {
    "blinkmimic": ("Cyclo-BlinkMimic-K1-Rev1",),
    "beyondmimic": (
        "Cyclo-BeyondMimic-K1-Rev1-Dance1",
        "Cyclo-BeyondMimic-K1-Rev1-Dance2",
    ),
    "supermimic": (
        "Cyclo-SuperMimic-GenCoord-K1-Rev1",
        "Cyclo-SuperMimic-MaxCoord-K1-Rev1",
    ),
}


def task_family(task):
    if task == "Cyclo-BlinkMimic-K1-Rev1":
        return "blinkmimic"
    if task.startswith("Cyclo-SuperMimic-"):
        return "supermimic"
    if task.startswith(("Cyclo-Mimic-", "Cyclo-BeyondMimic-")):
        return "beyondmimic"
    return None


def require_runtime(algorithm):
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Cyclo requires Python 3.12 for both learning algorithms")
    try:
        actual = version("mjlab")
    except PackageNotFoundError:
        actual = "not installed"
    if actual != "1.6.1":
        raise SystemExit(
            f"Cyclo requires mjlab 1.6.1; found {actual}. "
            "Use the locked Python 3.12 runtime described in docs/supermimic.md"
        )


def dispatch(action):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--algorithm", choices=tuple(TASKS))
    parser.add_argument("--list-tasks", action="store_true")
    args, remaining = parser.parse_known_args()
    task = next((a for a in remaining if a.startswith("Cyclo-")), None)
    if (
        args.list_tasks
        or task is None
        and any(a in remaining for a in ("-h", "--help"))
    ):
        print(
            f"Usage: python scripts/reinforcement_learning/{action}.py [--algorithm FAMILY] TASK [options]"
        )
        for family, tasks in TASKS.items():
            if args.algorithm is None or args.algorithm == family:
                print(f"{family}: " + ", ".join(tasks))
        if args.algorithm is None:
            print("velocity: Cyclo-Velocity-Flat-K1-Rev1-v0")
        print(
            "Historical Cyclo-Mimic-* aliases and Cyclo-Velocity-* commands remain supported."
        )
        print(
            "BeyondMimic and SuperMimic share the locked MJLab 1.6.1 / Python 3.12 runtime."
        )
        return True
    family = task_family(task or "")
    if args.algorithm and family != args.algorithm:
        parser.error(f"--algorithm {args.algorithm} does not match task {task!r}")
    family = args.algorithm or family
    if family == "blinkmimic":
        require_runtime(family)
        from source.tasks.blinkmimic.cli import main

        main(action, remaining)
        return True
    if family == "supermimic":
        require_runtime(family)
        from source.tasks.supermimic.cli import main

        sys.argv = [sys.argv[0], action, *remaining]
        main()
        return True
    require_runtime("beyondmimic")
    sys.argv = [sys.argv[0], *remaining]
    return False
