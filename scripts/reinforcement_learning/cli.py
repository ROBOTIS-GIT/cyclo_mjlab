# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Insu Park, Kiwoong Park

"""Command-line options and task selection shared by train and play."""

import argparse

TASK_PREFIXES = {
    "blinkmimic": "Cyclo-BlinkMimic-",
    "beyondmimic": "Cyclo-BeyondMimic-",
    "supermimic": "Cyclo-SuperMimic-",
}


def task_names(family):
    if family == "blinkmimic":
        from source.tasks.blinkmimic.config import TASKS

        return TASKS

    import source.tasks  # noqa: F401  # Register Cyclo tasks.
    from mjlab.tasks.registry import list_tasks

    prefix = "Cyclo-Velocity-" if family == "velocity" else TASK_PREFIXES[family]
    return [name for name in list_tasks() if name.startswith(prefix)]


def print_task_help(action, algorithm):
    print(f"Usage: python scripts/reinforcement_learning/{action}.py [--algorithm FAMILY] TASK [options]")
    families = (algorithm,) if algorithm else (*TASK_PREFIXES, "velocity")
    for family in families:
        print(f"{family}: {', '.join(task_names(family))}")
    print("BeyondMimic and SuperMimic share the locked MJLab 1.6.1 / Python 3.12 runtime.")


def route_command(action, argv=None):
    """Return PPO arguments, or None after handling help or a specialized task."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--algorithm", choices=TASK_PREFIXES)
    parser.add_argument("--list-tasks", action="store_true")
    options, remaining = parser.parse_known_args(argv)
    task = next((arg for arg in remaining if arg.startswith("Cyclo-")), None)
    if options.list_tasks or (task is None and ("--help" in remaining or "-h" in remaining)):
        print_task_help(action, options.algorithm)
        return None

    family = next((name for name, prefix in TASK_PREFIXES.items() if task and task.startswith(prefix)), None)
    if options.algorithm and options.algorithm != family:
        parser.error(f"--algorithm {options.algorithm} does not match task {task!r}")

    if family == "blinkmimic":
        from source.tasks.blinkmimic.cli import main as blinkmimic_main

        blinkmimic_main(action, remaining)
    elif family == "supermimic":
        from source.tasks.supermimic.cli import main as supermimic_main

        supermimic_main([action, *remaining])
    else:
        return remaining
    return None
