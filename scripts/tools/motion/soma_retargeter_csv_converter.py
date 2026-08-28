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

"""Convert Soma retargeter CSV files to K1 Rev.1 MuJoCo motion files.

The input is a headered Soma CSV containing root translation, root XYZ Euler
rotation, and 23 ``<joint_name>_dof`` columns. The output is:

* a numeric mimic-style CSV with root xyz, root quaternion xyzw, and 23 joints;
* an NPZ containing MuJoCo-ordered joints and full-body kinematics for MJLab.

Unlike the Isaac Lab version, this converter does not launch Isaac Sim. It uses
the K1 MJCF and MuJoCo forward kinematics through :mod:`csv_to_npz`.

Example:
    python scripts/tools/motion/soma_retargeter_csv_converter.py \
        -f path/to/motion_soma.csv \
        --output_name source/assets/motions/K1_rev1/motion/motion.npz
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

if __package__:
    from .csv_to_npz import (
        DEFAULT_K1_XML,
        K1_REV1_MOTION_CSV_JOINT_NAMES,
        get_model_layout,
        load_model,
        resample_motion,
        save_motion_files,
    )
else:
    from csv_to_npz import (
        DEFAULT_K1_XML,
        K1_REV1_MOTION_CSV_JOINT_NAMES,
        get_model_layout,
        load_model,
        resample_motion,
        save_motion_files,
    )


SOMA_ROOT_POSITION_COLUMNS = (
    "root_translateX",
    "root_translateY",
    "root_translateZ",
)
SOMA_ROOT_EULER_COLUMNS = (
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
)
SOMA_JOINT_COLUMNS = tuple(
    f"{name}_dof" for name in K1_REV1_MOTION_CSV_JOINT_NAMES
)
SOMA_REQUIRED_COLUMNS = (
    "Frame",
    *SOMA_ROOT_POSITION_COLUMNS,
    *SOMA_ROOT_EULER_COLUMNS,
    *SOMA_JOINT_COLUMNS,
)


def euler_xyz_to_quat_wxyz(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles in radians to wxyz quaternions.

    The formula matches Isaac Lab's ``quat_from_euler_xyz`` used by the
    original Soma converter.
    """

    half_angles = 0.5 * np.asarray(euler_xyz, dtype=np.float64)
    roll, pitch, yaw = half_angles.T
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.column_stack(
        (
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        )
    )


def load_soma_csv(
    motion_file: str | Path,
    frame_range: tuple[int, int] | None,
    position_scale: float,
    angle_unit: str,
    root_height_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Soma channels and convert them to SI units and wxyz quaternions."""

    path = Path(motion_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Soma motion CSV was not found: {path}")
    if not np.isfinite(position_scale) or not np.isfinite(root_height_offset):
        raise ValueError("Position scale and root height offset must be finite.")

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")
        missing = [
            name for name in SOMA_REQUIRED_COLUMNS if name not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"Soma CSV is missing required columns: {missing}")
        rows = list(reader)

    if frame_range is not None:
        start, end = frame_range
        if start < 1 or end < start:
            raise ValueError(
                f"Invalid frame range {frame_range}; expected 1-based START END."
            )
        rows = rows[start - 1 : end]
    if len(rows) < 2:
        raise ValueError("Selected Soma motion must contain at least two frames.")

    value_columns = (
        *SOMA_ROOT_POSITION_COLUMNS,
        *SOMA_ROOT_EULER_COLUMNS,
        *SOMA_JOINT_COLUMNS,
    )
    values = np.empty((len(rows), len(value_columns)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(value_columns):
            try:
                values[row_index, column_index] = float(row[column])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value at data row {row_index + 1}, "
                    f"column '{column}': {row[column]!r}"
                ) from error
    if not np.all(np.isfinite(values)):
        raise ValueError("Soma CSV contains NaN or infinite values.")

    root_pos = values[:, :3] * position_scale
    root_pos[:, 2] += root_height_offset
    root_euler = values[:, 3:6]
    joint_pos = values[:, 6:]
    if angle_unit == "degrees":
        root_euler = np.deg2rad(root_euler)
        joint_pos = np.deg2rad(joint_pos)
    elif angle_unit != "radians":
        raise ValueError(f"Unsupported angle unit: {angle_unit}")

    root_quat_wxyz = euler_xyz_to_quat_wxyz(root_euler)
    return root_pos, root_quat_wxyz, joint_pos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Soma retargeter CSV to K1 MuJoCo CSV plus NPZ."
    )
    parser.add_argument(
        "--input_file",
        "--input-file",
        "-f",
        type=Path,
        required=True,
        help="Input headered Soma retargeter CSV.",
    )
    parser.add_argument("--input_fps", "--input-fps", type=int, default=30)
    parser.add_argument("--output_fps", "--output-fps", type=int, default=50)
    parser.add_argument(
        "--frame_range",
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="1-based inclusive input data-row range after the header.",
    )
    parser.add_argument(
        "--output_name",
        "--output-name",
        type=Path,
        help="Output NPZ. Defaults to <input_stem>_converted.npz.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_K1_XML,
        help=f"K1 MJCF path (default: {DEFAULT_K1_XML}).",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=0.01,
        help="Root-position scale. The default converts centimeters to meters.",
    )
    parser.add_argument(
        "--angle-unit",
        choices=("degrees", "radians"),
        default="degrees",
        help="Input unit for root Euler angles and joint positions.",
    )
    parser.add_argument(
        "--root-height-offset",
        type=float,
        default=0.0,
        help="Additional root z offset in meters after position scaling.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Optional odd moving-average window. Values <= 1 disable smoothing.",
    )
    parser.add_argument("--smooth-passes", type=int, default=1)
    parser.add_argument(
        "--smooth-fields",
        nargs="+",
        default=("root_pos", "root_rot", "joints"),
        choices=("root_pos", "root_rot", "joints", "all"),
    )
    return parser.parse_args()


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.output_name is None:
        npz_path = args.input_file.with_name(
            f"{args.input_file.stem}_converted.npz"
        )
    else:
        npz_path = args.output_name
        if npz_path.suffix.lower() == ".csv":
            npz_path = npz_path.with_suffix(".npz")
        elif npz_path.suffix.lower() != ".npz":
            npz_path = Path(f"{npz_path}.npz")
    npz_path = npz_path.expanduser().resolve()
    return npz_path, npz_path.with_suffix(".csv")


def main() -> None:
    args = parse_args()
    frame_range = tuple(args.frame_range) if args.frame_range else None
    root_pos, root_quat_wxyz, joint_pos = load_soma_csv(
        motion_file=args.input_file,
        frame_range=frame_range,
        position_scale=args.position_scale,
        angle_unit=args.angle_unit,
        root_height_offset=args.root_height_offset,
    )
    motion = resample_motion(
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        joint_pos_csv_order=joint_pos,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        smooth_window=args.smooth_window,
        smooth_passes=args.smooth_passes,
        smooth_fields=tuple(args.smooth_fields),
    )
    npz_path, csv_path = resolve_output_paths(args)
    model = load_model(args.model)
    layout = get_model_layout(model)
    save_motion_files(
        model=model,
        layout=layout,
        motion=motion,
        npz_path=npz_path,
        csv_path=csv_path,
    )


if __name__ == "__main__":
    main()
