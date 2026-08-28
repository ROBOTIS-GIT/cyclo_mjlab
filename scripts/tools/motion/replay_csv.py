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

"""Kinematics-only MuJoCo viewer replay for a K1 Rev.1 motion CSV."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_K1_XML = (
PROJECT_ROOT
    / "source"
    / "assets"
    / "robots"
    / "robotis_k1"
    / "xmls"
    / "k1.xml"
)

K1_REV1_MOTION_CSV_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)


@dataclass(frozen=True)
class ModelLayout:
    root_joint_id: int
    root_body_id: int
    joint_qpos_addresses: np.ndarray
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]


def load_model(model_path: str | Path) -> mujoco.MjModel:
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"K1 MJCF was not found: {path}")
    return mujoco.MjModel.from_xml_path(str(path))


def get_model_layout(model: mujoco.MjModel) -> ModelLayout:
    root_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
    )
    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if root_joint_id < 0 or root_body_id < 0:
        raise ValueError(
            "The K1 model must contain 'floating_base_joint' and body 'pelvis'."
        )
    if model.jnt_type[root_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("'floating_base_joint' must be a MuJoCo free joint.")

    joint_ids = []
    for name in K1_REV1_MOTION_CSV_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"K1 MJCF is missing CSV joint: {name}")
        joint_ids.append(joint_id)

    body_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    )
    return ModelLayout(
        root_joint_id=root_joint_id,
        root_body_id=root_body_id,
        joint_qpos_addresses=model.jnt_qposadr[np.asarray(joint_ids)].copy(),
        joint_names=K1_REV1_MOTION_CSV_JOINT_NAMES,
        body_names=body_names,
    )


def load_csv(
    motion_file: str | Path,
    frame_range: tuple[int, int] | None,
    root_quat_order: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(motion_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Motion CSV was not found: {path}")

    load_options: dict[str, int] = {}
    if frame_range is not None:
        start, end = frame_range
        if start < 1 or end < start:
            raise ValueError(
                f"Invalid frame range {frame_range}; expected 1-based START END."
            )
        load_options["skiprows"] = start - 1
        load_options["max_rows"] = end - start + 1

    motion = np.loadtxt(path, delimiter=",", dtype=np.float64, **load_options)
    if motion.ndim == 1:
        motion = motion[None, :]
    if motion.shape[1] != 30:
        raise ValueError(
            "Expected 30 CSV columns: root xyz + quaternion + 23 joints, "
            f"got {motion.shape[1]}."
        )
    if not np.all(np.isfinite(motion)):
        raise ValueError("Motion CSV contains NaN or infinite values.")

    root_pos = motion[:, :3]
    root_quat = motion[:, 3:7]
    if root_quat_order == "xyzw":
        root_quat = root_quat[:, [3, 0, 1, 2]]
    quaternion_norm = np.linalg.norm(root_quat, axis=1, keepdims=True)
    if np.any(quaternion_norm < 1.0e-8):
        raise ValueError("Motion contains a zero-length root quaternion.")
    root_quat = root_quat / quaternion_norm
    return root_pos, root_quat, motion[:, 7:]


def motion_to_qpos(
    model: mujoco.MjModel,
    layout: ModelLayout,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
) -> np.ndarray:
    qpos = np.tile(model.qpos0, (root_pos.shape[0], 1))
    root_address = int(model.jnt_qposadr[layout.root_joint_id])
    qpos[:, root_address : root_address + 3] = root_pos
    qpos[:, root_address + 3 : root_address + 7] = root_quat
    qpos[:, layout.joint_qpos_addresses] = joint_pos
    return qpos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a K1 Rev.1 CSV in the native MuJoCo viewer."
    )
    parser.add_argument("--input_file", "-f", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_K1_XML, help="K1 MJCF path."
    )
    parser.add_argument(
        "--root-quat-order", choices=("xyzw", "wxyz"), default="xyzw"
    )
    parser.add_argument(
        "--frame_range", nargs=2, type=int, metavar=("START", "END")
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--max-frames", type=int, help="Stop after this many displayed frames."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate without opening a GUI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive.")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive.")

    model = load_model(args.model)
    layout = get_model_layout(model)
    root_pos, root_quat, joint_pos = load_csv(
        args.input_file,
        tuple(args.frame_range) if args.frame_range else None,
        args.root_quat_order,
    )
    qpos = motion_to_qpos(model, layout, root_pos, root_quat, joint_pos)
    frames = qpos.shape[0]
    print(
        f"[INFO] CSV ready: frames={frames}, fps={args.fps:g}, "
        f"joints={len(layout.joint_names)}, bodies={len(layout.body_names)}"
    )
    if args.validate_only:
        print("[INFO] CSV validation complete.")
        return

    import mujoco.viewer

    data = mujoco.MjData(model)
    frame = 0
    rendered_frames = 0
    frame_period = 1.0 / (args.fps * args.speed)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -10.0
        while viewer.is_running():
            frame_start = time.perf_counter()
            data.qpos[:] = qpos[frame]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.cam.lookat[:] = data.xpos[layout.root_body_id]
            viewer.sync()

            rendered_frames += 1
            if args.max_frames is not None and rendered_frames >= args.max_frames:
                break
            frame += 1
            if frame >= frames:
                if not args.loop:
                    break
                frame = 0

            delay = frame_period - (time.perf_counter() - frame_start)
            if delay > 0.0:
                time.sleep(delay)
    print("[INFO] CSV replay complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] CSV replay failed: {error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
