"""Prepare a Cyclo scene and exact reference library, then run upstream X-Mimic."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from source.tasks.blinkmimic.runtime import ROOT, prepare

from .config import TASKS

UPSTREAM = ROOT / "third_party/xmimic"
STATE_FIELDS = (
    "root_pos_w",
    "root_quat_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "joint_pos",
    "joint_vel",
)


def prepare_references(source_manifest, output, joint_names):
    from source.tasks.blinkmimic.motion import digest, load_exact, verify_manifest

    manifest = verify_manifest(source_manifest)
    output.mkdir()
    clips = []
    for index, row in enumerate(manifest["clips"]):
        arrays = load_exact(row["derived"], joint_names)
        if len(arrays[0]) != row["frames"]:
            raise ValueError(f"Frame count mismatch for {row['song']}")
        path = output / f"{index:04d}.npz"
        np.savez(path, **dict(zip(STATE_FIELDS, arrays, strict=True)))
        clips.append(
            dict(
                song=row["song"],
                derived=path.name,
                frames=row["frames"],
                fps=row["fps"],
                derived_sha256=digest(path),
            )
        )
    (output / "manifest.json").write_text(json.dumps({"clips": clips}, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(TASKS))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--jax-python", type=Path, required=True)
    args, trainer = parser.parse_known_args(argv)
    reserved = {"--model_dir", "--motion_dir", "--log_dir"}
    if any(flag.split("=", 1)[0] in reserved for flag in trainer):
        parser.error("Scene, motion and output paths are managed by the Cyclo launcher")
    python = args.jax_python.expanduser().absolute()
    if not python.is_file():
        parser.error("--jax-python must point to an existing interpreter")

    prepare()
    from .scene import export_scene

    task = TASKS[args.task]
    robot = task.robot_factory()
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True)
    export_scene(robot, task.contact_bodies, directory / "scene")
    prepare_references(args.manifest, directory / "motions", robot.joint_names)
    subprocess.run(
        [
            str(python),
            str(UPSTREAM / "train.py"),
            *trainer,
            "--model_dir",
            str(directory / "scene"),
            "--motion_dir",
            str(directory / "motions"),
            "--log_dir",
            str(directory),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
