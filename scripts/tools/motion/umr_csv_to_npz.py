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

"""Convert headered UMR K1 CSV to mimic-compatible NPZ and numeric CSV.

Input columns: time_s, root_{x,y,z}_m, root_q{w,x,y,z}, and named K1
joints in radians. Columns are matched by name, regardless of their order.
Output CSV uses root xyz + quaternion xyzw + 23 joints, without a header.
Kinematics, smoothing, resampling and output validation reuse csv_to_npz.py.

Example:
    python scripts/tools/motion/umr_csv_to_npz.py -f motion_umr.csv --output_fps 50
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
        resolve_output_paths,
        get_model_layout,
        load_model,
        resample_motion,
        save_motion_files,
    )
else:
    from csv_to_npz import (
        DEFAULT_K1_XML,
        K1_REV1_MOTION_CSV_JOINT_NAMES,
        resolve_output_paths,
        get_model_layout,
        load_model,
        resample_motion,
        save_motion_files,
    )


UMR_REQUIRED_COLUMNS = (
    "time_s",
    "root_x_m", "root_y_m", "root_z_m",
    "root_qw", "root_qx", "root_qy", "root_qz",
    *K1_REV1_MOTION_CSV_JOINT_NAMES,
)


def load_umr_csv(
    motion_file: str | Path,
    frame_range: tuple[int, int] | None = None,
    input_fps: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Read named channels and infer FPS, allowing 0.1% timestamp rounding.

    Explicit input_fps overrides timing; timestamps must still increase.
    """

    path = Path(motion_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"UMR motion CSV was not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        names = reader.fieldnames
        if not names:
            raise ValueError(f"Input CSV has no header: {path}")
        if len(names) != len(set(names)):
            raise ValueError("UMR CSV contains duplicate column names.")
        missing = [name for name in UMR_REQUIRED_COLUMNS if name not in names]
        if missing:
            raise ValueError(f"UMR CSV is missing required columns: {missing}")
        rows = list(reader)

    first_row = 1
    if frame_range is not None:
        start, end = frame_range
        if start < 1 or end < start or end > len(rows):
            raise ValueError(
                f"Invalid frame range {frame_range}; expected 1 <= START <= END "
                f"<= {len(rows)}."
            )
        rows = rows[start - 1 : end]
        first_row = start
    if len(rows) < 2:
        raise ValueError("Selected UMR motion must contain at least two frames.")

    values = np.empty((len(rows), len(UMR_REQUIRED_COLUMNS)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        if None in row:
            raise ValueError(f"Extra CSV values at data row {first_row + row_index}.")
        for column_index, column in enumerate(UMR_REQUIRED_COLUMNS):
            try:
                values[row_index, column_index] = float(row[column])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value at data row {first_row + row_index}, "
                    f"column '{column}': {row[column]!r}"
                ) from error
    if not np.all(np.isfinite(values)):
        raise ValueError("UMR CSV contains NaN or infinite values.")
    intervals = np.diff(values[:, 0])
    if np.any(intervals <= 0.0):
        raise ValueError("UMR time_s must be strictly increasing.")
    if input_fps is None:
        dt = float(np.mean(intervals))
        if not np.allclose(intervals, dt, rtol=1e-3, atol=1e-6):
            raise ValueError(
                "UMR time_s is not uniformly sampled. Resample first or set "
                "--input_fps explicitly to override source timing."
            )
        input_fps = 1.0 / dt
    if not np.isfinite(input_fps) or input_fps <= 0.0:
        raise ValueError("Input FPS must be positive and finite.")
    quaternions = values[:, 4:8]
    if np.any(np.linalg.norm(quaternions, axis=1) < 1e-8):
        raise ValueError("UMR CSV contains a zero-length root quaternion.")
    return values[:, 1:4], quaternions, values[:, 8:], float(input_fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert headered UMR K1 CSV to MuJoCo CSV plus NPZ."
    )
    parser.add_argument(
        "--input_file", "-f", type=Path, required=True, help="Input headered UMR K1 CSV."
    )
    parser.add_argument(
        "--input_fps", type=float,
        help="Override input FPS; otherwise infer uniform sampling from time_s.",
    )
    parser.add_argument("--output_fps", type=int, default=50)
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="1-based inclusive data-row range, excluding the header.",
    )
    parser.add_argument(
        "--output_name",
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


def main() -> None:
    args = parse_args()
    npz_path, csv_path = resolve_output_paths(args)
    if args.input_file.expanduser().resolve() in (npz_path, csv_path):
        raise ValueError("Output paths must not overwrite the input UMR CSV.")
    root_pos, root_quat_wxyz, joint_pos, input_fps = load_umr_csv(
        motion_file=args.input_file,
        frame_range=tuple(args.frame_range) if args.frame_range else None,
        input_fps=args.input_fps,
    )
    print(f"[INFO] UMR input: frames={len(root_pos)}, fps={input_fps:.8f}")
    motion = resample_motion(
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        joint_pos_csv_order=joint_pos,
        input_fps=input_fps,
        output_fps=args.output_fps,
        smooth_window=args.smooth_window,
        smooth_passes=args.smooth_passes,
        smooth_fields=tuple(args.smooth_fields),
    )
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
