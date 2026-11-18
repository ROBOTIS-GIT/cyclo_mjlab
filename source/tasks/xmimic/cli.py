"""Route X-Mimic training and checkpoint evaluation."""

import argparse
import json
import subprocess
from pathlib import Path

from source.tasks.blinkmimic.runtime import prepare

from .config import TASKS, robot_contract
from .train import UPSTREAM


def main(action, argv):
    if action == "train":
        from .train import main as train

        return train(argv)

    parser = argparse.ArgumentParser(description="Evaluate X-Mimic checkpoints in MJX.")
    parser.add_argument("task", choices=tuple(TASKS))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--jax-python", type=Path, required=True)
    args, extra = parser.parse_known_args(argv)
    if any(flag.split("=", 1)[0] == "--expected_robot" for flag in extra):
        parser.error("The expected robot is set by the task")
    prepare()
    robot = TASKS[args.task].robot_factory()
    subprocess.run(
        [
            str(args.jax_python.expanduser().absolute()),
            "-m",
            "xmimic.evaluate",
            str(args.run_dir.resolve()),
            "--group",
            "1",
            *extra,
            "--expected_robot",
            json.dumps(robot_contract(robot)),
        ],
        cwd=UPSTREAM,
        check=True,
    )
