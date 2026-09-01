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

"""Validate and replay a MuJoCo-ordered K1 Rev.1 motion NPZ."""

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

from source.assets.robots import K1_REV1_XML_PATH


DEFAULT_K1_XML = K1_REV1_XML_PATH

REQUIRED_NPZ_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class ModelLayout:
    root_joint_id: int
    root_body_id: int
    joint_qpos_addresses: np.ndarray
    joint_names: tuple[str, ...]
    body_ids: np.ndarray
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

    joint_ids = np.asarray(
        [
            joint_id
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
        ],
        dtype=np.int32,
    )
    joint_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
        for joint_id in joint_ids
    )
    body_ids = np.arange(1, model.nbody, dtype=np.int32)
    body_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
        for body_id in body_ids
    )
    return ModelLayout(
        root_joint_id=root_joint_id,
        root_body_id=root_body_id,
        joint_qpos_addresses=model.jnt_qposadr[joint_ids].copy(),
        joint_names=joint_names,
        body_ids=body_ids,
        body_names=body_names,
    )


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("Motion contains a zero-length quaternion.")
    return quaternions / norms


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    motion_path = Path(path).expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion NPZ was not found: {motion_path}")
    with np.load(motion_path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_NPZ_KEYS if key not in data]
        if missing:
            raise ValueError(f"Motion NPZ is missing required keys: {missing}")
        output = {key: data[key].copy() for key in data.files}
    if not all(np.all(np.isfinite(output[key])) for key in REQUIRED_NPZ_KEYS):
        raise ValueError("Motion NPZ contains NaN or infinite values.")
    return output


def qpos_from_npz(
    model: mujoco.MjModel,
    layout: ModelLayout,
    motion_data: dict[str, np.ndarray],
) -> np.ndarray:
    joint_pos = motion_data["joint_pos"]
    body_pos = motion_data["body_pos_w"]
    body_quat = motion_data["body_quat_w"]
    frames = joint_pos.shape[0]

    if body_pos.shape != (frames, len(layout.body_names), 3):
        raise ValueError(
            "body_pos_w shape does not match the MuJoCo model: "
            f"{body_pos.shape} vs {(frames, len(layout.body_names), 3)}"
        )
    if body_quat.shape != (frames, len(layout.body_names), 4):
        raise ValueError(
            "body_quat_w shape does not match the MuJoCo model: "
            f"{body_quat.shape} vs {(frames, len(layout.body_names), 4)}"
        )

    if "joint_names" in motion_data:
        npz_joint_names = tuple(str(name) for name in motion_data["joint_names"])
    else:
        npz_joint_names = layout.joint_names
        print("[WARNING] NPZ has no joint_names; assuming MuJoCo model order.")
    if set(npz_joint_names) != set(layout.joint_names):
        raise ValueError("NPZ joint_names do not match the K1 MuJoCo model.")

    if "body_names" in motion_data:
        npz_body_names = tuple(str(name) for name in motion_data["body_names"])
    else:
        npz_body_names = layout.body_names
        print("[WARNING] NPZ has no body_names; assuming MuJoCo model order.")
    if npz_body_names != layout.body_names:
        raise ValueError(
            "NPZ body_names are not in K1 MuJoCo order. Re-run csv_to_npz.py."
        )
    if joint_pos.shape != (frames, len(npz_joint_names)):
        raise ValueError(
            f"joint_pos has invalid shape {joint_pos.shape}; "
            f"expected {(frames, len(npz_joint_names))}."
        )

    qpos = np.tile(model.qpos0, (frames, 1))
    root_body_index = npz_body_names.index("pelvis")
    root_address = int(model.jnt_qposadr[layout.root_joint_id])
    qpos[:, root_address : root_address + 3] = body_pos[:, root_body_index]
    qpos[:, root_address + 3 : root_address + 7] = normalize_quaternions(
        body_quat[:, root_body_index]
    )

    joint_index_by_name = {
        name: index for index, name in enumerate(npz_joint_names)
    }
    for name, address in zip(
        layout.joint_names, layout.joint_qpos_addresses, strict=True
    ):
        qpos[:, address] = joint_pos[:, joint_index_by_name[name]]
    return qpos


def validate_forward_kinematics(
    model: mujoco.MjModel,
    layout: ModelLayout,
    motion_data: dict[str, np.ndarray],
    qpos: np.ndarray,
) -> tuple[float, float]:
    data = mujoco.MjData(model)
    maximum_position_error = 0.0
    maximum_orientation_error = 0.0
    expected_pos = motion_data["body_pos_w"]
    expected_quat = normalize_quaternions(motion_data["body_quat_w"])

    for frame in range(qpos.shape[0]):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        actual_pos = data.xpos[layout.body_ids]
        actual_quat = normalize_quaternions(data.xquat[layout.body_ids])
        position_error = np.linalg.norm(actual_pos - expected_pos[frame], axis=1)
        quaternion_dot = np.sum(actual_quat * expected_quat[frame], axis=1)
        aligned_expected_quat = expected_quat[frame].copy()
        aligned_expected_quat[quaternion_dot < 0.0] *= -1.0
        quaternion_chord = np.linalg.norm(
            actual_quat - aligned_expected_quat, axis=1
        )
        orientation_error = 4.0 * np.arcsin(
            np.clip(0.5 * quaternion_chord, 0.0, 1.0)
        )
        maximum_position_error = max(
            maximum_position_error, float(np.max(position_error))
        )
        maximum_orientation_error = max(
            maximum_orientation_error, float(np.max(orientation_error))
        )
    return maximum_position_error, maximum_orientation_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and replay a K1 motion NPZ in MuJoCo."
    )
    parser.add_argument("--input_file", "-f", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_K1_XML, help="K1 MJCF path."
    )
    parser.add_argument(
        "--fps", type=float, help="Replay FPS. Defaults to the NPZ fps field."
    )
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="1-based inclusive replay frame range.",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--max-frames", type=int, help="Stop after this many displayed frames."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run schema and forward-kinematics checks without opening a GUI.",
    )
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=1.0e-5,
        help="Maximum accepted FK body-position error in meters.",
    )
    parser.add_argument(
        "--orientation-tolerance",
        type=float,
        default=1.0e-5,
        help="Maximum accepted FK body-orientation error in radians.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive.")

    model = load_model(args.model)
    layout = get_model_layout(model)
    motion_data = load_npz(args.input_file)
    qpos = qpos_from_npz(model, layout, motion_data)
    frames = qpos.shape[0]

    start = 0
    end = frames
    if args.frame_range is not None:
        start_arg, end_arg = args.frame_range
        if start_arg < 1 or end_arg < start_arg or end_arg > frames:
            raise ValueError(
                f"Invalid frame range {args.frame_range}; motion has {frames} frames."
            )
        start = start_arg - 1
        end = end_arg

    position_error, orientation_error = validate_forward_kinematics(
        model,
        layout,
        {
            **motion_data,
            "body_pos_w": motion_data["body_pos_w"][start:end],
            "body_quat_w": motion_data["body_quat_w"][start:end],
        },
        qpos[start:end],
    )
    print(
        "[INFO] NPZ validated: "
        f"frames={frames}, joints={len(layout.joint_names)}, "
        f"bodies={len(layout.body_names)}"
    )
    print(
        "[INFO] Forward-kinematics error: "
        f"position={position_error:.3e} m, "
        f"orientation={orientation_error:.3e} rad"
    )
    if position_error > args.position_tolerance:
        raise ValueError(
            f"Position error exceeds tolerance {args.position_tolerance:.3e} m."
        )
    if orientation_error > args.orientation_tolerance:
        raise ValueError(
            "Orientation error exceeds tolerance "
            f"{args.orientation_tolerance:.3e} rad."
        )
    if args.validate_only:
        print("[INFO] NPZ validation complete.")
        return

    import mujoco.viewer

    fps = float(np.asarray(motion_data["fps"]).reshape(-1)[0])
    if args.fps is not None:
        fps = args.fps
    if fps <= 0.0:
        raise ValueError("Replay FPS must be positive.")

    data = mujoco.MjData(model)
    frame = start
    rendered_frames = 0
    frame_period = 1.0 / (fps * args.speed)
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
            if frame >= end:
                if not args.loop:
                    break
                frame = start

            delay = frame_period - (time.perf_counter() - frame_start)
            if delay > 0.0:
                time.sleep(delay)
    print("[INFO] NPZ replay complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[ERROR] NPZ replay failed: {error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
