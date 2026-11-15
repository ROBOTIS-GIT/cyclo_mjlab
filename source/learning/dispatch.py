"""Select an algorithm before importing its incompatible simulator runtime.

No installation, process launch, or environment mutation is performed. Native
mjlab task IDs infer the family; --algorithm makes that selection explicit.
"""

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

TASKS = {
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
    if task.startswith("Cyclo-SuperMimic-"):
        return "supermimic"
    if task.startswith(("Cyclo-Mimic-", "Cyclo-BeyondMimic-")):
        return "beyondmimic"
    return None


def require_runtime(algorithm):
    if algorithm == "supermimic" and sys.version_info[:2] != (3, 13):
        raise SystemExit(
            "supermimic requires Python 3.13; use uv run --project integrations/supermimic python"
        )
    expected = "1.6.0" if algorithm == "supermimic" else "1.2.0"
    try:
        actual = version("mjlab")
    except PackageNotFoundError:
        actual = "not installed"
    if actual != expected:
        setup = (
            "uv sync --project integrations/supermimic --python 3.13 --locked; "
            "then use uv run --project integrations/supermimic python scripts/reinforcement_learning/train.py (or play.py)"
            if algorithm == "supermimic"
            else "use the existing Cyclo Docker / Python 3.11 environment"
        )
        raise SystemExit(
            f"{algorithm} requires mjlab {expected}; found {actual}. {setup}"
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
            "BeyondMimic: existing mjlab 1.2 Docker. SuperMimic: separate mjlab_research 1.6 / Python 3.13 runtime."
        )
        return True
    family = task_family(task or "")
    if args.algorithm and family != args.algorithm:
        parser.error(f"--algorithm {args.algorithm} does not match task {task!r}")
    family = args.algorithm or family
    if family == "supermimic":
        require_runtime(family)
        from source.tasks.supermimic.cli import main

        sys.argv = [sys.argv[0], action, *remaining]
        main()
        return True
    require_runtime("beyondmimic")
    sys.argv = [sys.argv[0], *remaining]
    return False
