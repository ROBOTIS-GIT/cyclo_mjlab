"""Prepare a portable manifest for shared Mimic or normalized 50 Hz K1 reference NPZs."""

import argparse
import json
import shutil
from pathlib import Path

from source.utils.motion_paths import resolve_motion_files

from .runtime import prepare


def prepare_library(inputs, output, joint_names, *, mjcf_path=None):
    from .reference.motion import digest, load_exact

    paths = [Path(p).resolve() for p in inputs]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide a nonempty list of distinct reference files")
    if any(not p.is_file() or p.suffix.lower() != ".npz" for p in paths):
        raise ValueError("Inputs must be regular NPZ reference files")
    names = [p.stem for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("Reference basenames must be unique clip identifiers")
    rows = []
    for i, (path, name) in enumerate(zip(paths, names, strict=True)):
        arrays = load_exact(path, joint_names, mjcf_path=mjcf_path)
        rows.append(
            dict(
                song=name,
                derived=f"{i:04d}_{name}.npz",
                derived_sha256=digest(path),
                frames=len(arrays[0]),
                fps=50,
                source=str(path),
                source_sha256=digest(path),
            )
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for path, row in zip(paths, rows, strict=True):
        target = output / row["derived"]
        shutil.copyfile(path, target)
        if digest(target) != row["derived_sha256"]:
            raise ValueError(f"Reference changed during preparation: {path}")
    manifest = {"schema": "cyclo.blinkmimic.exact-state.v1", "clips": rows}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from .config import TASKS

    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare()
    robot = TASKS[args.task].robot_factory()
    print(prepare_library(args.inputs, args.output, robot.joint_names, mjcf_path=robot.mjcf_path))


if __name__ == "__main__":
    main()
