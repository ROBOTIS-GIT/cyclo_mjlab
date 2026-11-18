"""Write a reference CSV for the BlinkMimic Sim2Real observation runtime."""

import argparse
from pathlib import Path

import numpy as np

from source.tasks.blinkmimic.runtime import prepare

from .config import TASKS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(TASKS))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    prepare()
    from source.tasks.blinkmimic.motion import load_exact

    robot = TASKS[args.task].robot_factory()
    pos, quat, _, _, joints, _ = load_exact(args.input, robot.joint_names)
    header = (
        "root_pos_x",
        "root_pos_y",
        "root_pos_z",
        "root_quat_x",
        "root_quat_y",
        "root_quat_z",
        "root_quat_w",
        *robot.joint_names,
    )
    with args.output.open("x") as stream:
        np.savetxt(
            stream,
            np.concatenate((pos, quat[:, [1, 2, 3, 0]], joints), axis=1),
            delimiter=",",
            header=",".join(header),
            comments="",
            fmt="%.9g",
        )


if __name__ == "__main__":
    main()
