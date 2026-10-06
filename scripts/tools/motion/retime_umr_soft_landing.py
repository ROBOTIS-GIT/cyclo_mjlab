# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Slow the UMR pelvis landing while retiming every channel together.

Produces a new headered UMR CSV; convert it with umr_csv_to_npz.py.
The spatial path is preserved by linear interpolation and quaternion SLERP.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation, Slerp

if __package__:
    from .umr_csv_to_npz import load_umr_csv, UMR_REQUIRED_COLUMNS
else:
    from umr_csv_to_npz import load_umr_csv, UMR_REQUIRED_COLUMNS


def retime(root, quat, joints, fps, output_fps=50, start=4.85, end=5.3,
           ramp=0.35, stretch=2.5):
    duration = (len(root) - 1) / fps
    if not (0 < ramp <= start < end < duration - ramp and stretch >= 1
            and output_fps > 0):
        raise ValueError('Invalid landing interval, ramp, stretch or FPS.')
    dense = np.linspace(0, duration, int(np.ceil(duration * 10000)) + 1)
    def smooth(x):
        x = np.clip(x, 0, 1)
        return x**3 * (10 - 15*x + 6*x*x)
    envelope = smooth((dense-start+ramp)/ramp) * smooth((end+ramp-dense)/ramp)
    warped = cumulative_trapezoid(1 + (stretch-1)*envelope, dense, initial=0)
    # Round upward to a full output frame, never truncate the final posture.
    output_times = np.arange(int(np.ceil(warped[-1]*output_fps))+1)/output_fps
    source_times = np.interp(output_times, warped, dense)
    original = np.arange(len(root))/fps
    values = np.column_stack([root, joints])
    values = np.column_stack([np.interp(source_times, original, col) for col in values.T])
    q = Slerp(original, Rotation.from_quat(quat[:, [1,2,3,0]]))(source_times).as_quat()[:, [3,0,1,2]]
    return np.column_stack([output_times, values[:, :3], q, values[:, 3:]]), {
        'source_duration_s': duration, 'output_duration_s': float(output_times[-1]),
        'source_landing_interval_s': [start,end], 'ramp_s': ramp, 'time_stretch': stretch,
        'mapped_preparation_start_seconds': np.interp([5.,7.], dense, warped).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_file', type=Path, required=True)
    parser.add_argument('--output_file', type=Path, required=True)
    args = parser.parse_args()
    if args.input_file.resolve() == args.output_file.resolve():
        raise ValueError('The original UMR motion must not be overwritten.')
    rows, metadata = retime(*load_umr_csv(args.input_file))
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_file, rows, delimiter=',', header=','.join(UMR_REQUIRED_COLUMNS), comments='', fmt='%.10g')
    metadata['source'] = args.input_file.name
    args.output_file.with_suffix('.retiming.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
