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
# Additional notices:
# The motion conversion workflow is adapted from
# HybridRobotics/whole_body_tracking, licensed under the MIT License.

"""Convert a K1 Rev.1 motion CSV into a MuJoCo-ordered motion NPZ.

The input layout matches cyclo_lab:
  root position xyz + root quaternion xyzw + 23 named K1 joint positions.

Example:
    python scripts/tools/motion/csv_to_npz.py \
        -f source/assets/motions/K1_rev1/dance1/dance1.csv \
        --output_name source/assets/motions/K1_rev1/dance1/dance1.npz \
        --input_fps 50 --output_fps 50
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from source.assets.robots import K1_REV1_XML_PATH


DEFAULT_K1_XML = K1_REV1_XML_PATH

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
    """MuJoCo address and name layout used by a converted motion."""

    root_joint_id: int
    root_body_id: int
    joint_ids: np.ndarray
    joint_qpos_addresses: np.ndarray
    joint_dof_addresses: np.ndarray
    joint_names: tuple[str, ...]
    body_ids: np.ndarray
    body_names: tuple[str, ...]


@dataclass(frozen=True)
class Motion:
    """Resampled K1 motion in CSV channel order."""

    fps: float
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos_csv_order: np.ndarray

    @property
    def frames(self) -> int:
        return self.root_pos.shape[0]


def load_model(model_path: str | Path) -> mujoco.MjModel:
    """Load a MuJoCo model and provide a useful missing-file error."""

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"K1 MJCF was not found: {path}")
    return mujoco.MjModel.from_xml_path(str(path))


def get_model_layout(model: mujoco.MjModel) -> ModelLayout:
    """Resolve and validate K1 root, joint, and body ordering from MuJoCo names."""

    root_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
    )
    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if root_joint_id < 0 or root_body_id < 0:
        raise ValueError(
            "The K1 model must contain 'floating_base_joint' and root body 'pelvis'."
        )
    if model.jnt_type[root_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("'floating_base_joint' must be a MuJoCo free joint.")

    joint_ids = []
    for name in K1_REV1_MOTION_CSV_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"K1 MJCF is missing CSV joint: {name}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"CSV joint must be a scalar hinge joint: {name}")
        joint_ids.append(joint_id)

    model_joint_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]
    model_joint_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in model_joint_ids
    )
    if set(model_joint_names) != set(K1_REV1_MOTION_CSV_JOINT_NAMES):
        missing = sorted(set(K1_REV1_MOTION_CSV_JOINT_NAMES) - set(model_joint_names))
        extra = sorted(set(model_joint_names) - set(K1_REV1_MOTION_CSV_JOINT_NAMES))
        raise ValueError(
            "The K1 MJCF scalar joints do not match the 23 CSV joints. "
            f"Missing={missing}, extra={extra}"
        )

    body_ids = np.arange(1, model.nbody, dtype=np.int32)
    body_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
        for body_id in body_ids
    )
    if any(name is None for name in body_names):
        raise ValueError("Every K1 body must have a MuJoCo name.")

    model_joint_ids_array = np.asarray(model_joint_ids, dtype=np.int32)
    return ModelLayout(
        root_joint_id=root_joint_id,
        root_body_id=root_body_id,
        joint_ids=model_joint_ids_array,
        joint_qpos_addresses=model.jnt_qposadr[model_joint_ids_array].copy(),
        joint_dof_addresses=model.jnt_dofadr[model_joint_ids_array].copy(),
        joint_names=model_joint_names,
        body_ids=body_ids,
        body_names=body_names,
    )


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("Motion contains a zero-length root quaternion.")
    return quaternions / norms


def _make_quaternion_sign_continuous(quaternions: np.ndarray) -> np.ndarray:
    output = quaternions.copy()
    for index in range(1, output.shape[0]):
        if np.dot(output[index - 1], output[index]) < 0.0:
            output[index] *= -1.0
    return output


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    padding = window // 2
    padded = np.pad(values, ((padding, padding), (0, 0)), mode="edge")
    cumulative = np.cumsum(padded, axis=0, dtype=np.float64)
    cumulative = np.vstack((np.zeros((1, values.shape[1])), cumulative))
    return (cumulative[window:] - cumulative[:-window]) / window


def _slerp(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Vectorized shortest-path quaternion interpolation in wxyz order."""

    dot = np.sum(q0 * q1, axis=1)
    q1_adjusted = q1.copy()
    negative = dot < 0.0
    q1_adjusted[negative] *= -1.0
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)

    output = np.empty_like(q0)
    nearly_equal = dot > 0.9995
    if np.any(nearly_equal):
        alpha = blend[nearly_equal, None]
        output[nearly_equal] = (
            (1.0 - alpha) * q0[nearly_equal] + alpha * q1_adjusted[nearly_equal]
        )

    spherical = ~nearly_equal
    if np.any(spherical):
        theta = np.arccos(dot[spherical])
        sin_theta = np.sin(theta)
        alpha = blend[spherical]
        weight_0 = np.sin((1.0 - alpha) * theta) / sin_theta
        weight_1 = np.sin(alpha * theta) / sin_theta
        output[spherical] = (
            weight_0[:, None] * q0[spherical]
            + weight_1[:, None] * q1_adjusted[spherical]
        )
    return _normalize_quaternions(output)


def load_and_resample_csv(
    motion_file: str | Path,
    input_fps: float,
    output_fps: float,
    frame_range: tuple[int, int] | None,
    root_quat_order: str,
    smooth_window: int,
    smooth_passes: int,
    smooth_fields: tuple[str, ...],
) -> Motion:
    """Load, optionally smooth, and resample a 30-column K1 motion CSV."""

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

    values = np.loadtxt(path, delimiter=",", dtype=np.float64, **load_options)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != 30:
        raise ValueError(
            "K1 Rev.1 CSV must have 30 columns "
            f"(root xyz + quaternion + 23 joints), got {values.shape[1]}."
        )
    root_pos = values[:, :3].copy()
    root_quat = values[:, 3:7].copy()
    if root_quat_order == "xyzw":
        root_quat = root_quat[:, [3, 0, 1, 2]]
    elif root_quat_order != "wxyz":
        raise ValueError(f"Unsupported root quaternion order: {root_quat_order}")
    joint_pos = values[:, 7:].copy()

    return resample_motion(
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        joint_pos_csv_order=joint_pos,
        input_fps=input_fps,
        output_fps=output_fps,
        smooth_window=smooth_window,
        smooth_passes=smooth_passes,
        smooth_fields=smooth_fields,
    )


def resample_motion(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_pos_csv_order: np.ndarray,
    input_fps: float,
    output_fps: float,
    smooth_window: int,
    smooth_passes: int,
    smooth_fields: tuple[str, ...],
) -> Motion:
    """Validate, smooth, and resample K1 motion channels.

    This array-based entry point is shared by numeric K1 CSV conversion and
    headered retargeter formats such as Soma.
    """

    if input_fps <= 0.0 or output_fps <= 0.0:
        raise ValueError("Input and output FPS must be positive.")

    root_pos = np.asarray(root_pos, dtype=np.float64).copy()
    root_quat = np.asarray(root_quat_wxyz, dtype=np.float64).copy()
    joint_pos = np.asarray(joint_pos_csv_order, dtype=np.float64).copy()
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape (frames, 3), got {root_pos.shape}.")
    if root_quat.shape != (root_pos.shape[0], 4):
        raise ValueError(
            "root_quat_wxyz must have shape "
            f"{(root_pos.shape[0], 4)}, got {root_quat.shape}."
        )
    expected_joint_shape = (
        root_pos.shape[0],
        len(K1_REV1_MOTION_CSV_JOINT_NAMES),
    )
    if joint_pos.shape != expected_joint_shape:
        raise ValueError(
            f"joint_pos_csv_order must have shape {expected_joint_shape}, "
            f"got {joint_pos.shape}."
        )
    if root_pos.shape[0] < 2:
        raise ValueError("Motion must contain at least two frames.")
    if not all(
        np.all(np.isfinite(values))
        for values in (root_pos, root_quat, joint_pos)
    ):
        raise ValueError("Motion contains NaN or infinite values.")

    root_quat = _make_quaternion_sign_continuous(
        _normalize_quaternions(root_quat)
    )

    selected_fields = set(smooth_fields)
    if "all" in selected_fields:
        selected_fields = {"root_pos", "root_rot", "joints"}
    valid_fields = {"root_pos", "root_rot", "joints"}
    invalid_fields = selected_fields - valid_fields
    if invalid_fields:
        raise ValueError(f"Unsupported smoothing fields: {sorted(invalid_fields)}")

    if smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
            print(f"[INFO] --smooth-window must be odd; using {smooth_window}.")
        if smooth_passes < 1:
            raise ValueError("--smooth-passes must be at least 1.")
        for _ in range(smooth_passes):
            if "root_pos" in selected_fields:
                root_pos = _moving_average(root_pos, smooth_window)
            if "root_rot" in selected_fields:
                root_quat = _make_quaternion_sign_continuous(
                    _normalize_quaternions(_moving_average(root_quat, smooth_window))
                )
            if "joints" in selected_fields:
                joint_pos = _moving_average(joint_pos, smooth_window)

    input_frames = root_pos.shape[0]
    input_dt = 1.0 / input_fps
    duration = (input_frames - 1) * input_dt
    output_frames = int(np.floor(duration * output_fps + 1.0e-9)) + 1
    output_times = np.arange(output_frames, dtype=np.float64) / output_fps
    input_phase = output_times / input_dt
    index_0 = np.floor(input_phase).astype(np.int64)
    index_0 = np.clip(index_0, 0, input_frames - 1)
    index_1 = np.minimum(index_0 + 1, input_frames - 1)
    blend = input_phase - index_0

    root_pos_output = (
        (1.0 - blend[:, None]) * root_pos[index_0]
        + blend[:, None] * root_pos[index_1]
    )
    root_quat_output = _slerp(
        root_quat[index_0], root_quat[index_1], blend
    )
    joint_pos_output = (
        (1.0 - blend[:, None]) * joint_pos[index_0]
        + blend[:, None] * joint_pos[index_1]
    )

    print(
        "[INFO] Motion interpolated: "
        f"input_frames={input_frames}, input_fps={input_fps:g}, "
        f"output_frames={output_frames}, output_fps={output_fps:g}, "
        f"duration={duration:.3f}s"
    )
    return Motion(
        fps=output_fps,
        root_pos=root_pos_output,
        root_quat_wxyz=root_quat_output,
        joint_pos_csv_order=joint_pos_output,
    )


def motion_to_qpos(
    model: mujoco.MjModel, layout: ModelLayout, motion: Motion
) -> np.ndarray:
    """Map named CSV joints to MuJoCo qpos addresses."""

    qpos = np.tile(model.qpos0, (motion.frames, 1))
    root_qpos_address = int(model.jnt_qposadr[layout.root_joint_id])
    qpos[:, root_qpos_address : root_qpos_address + 3] = motion.root_pos
    qpos[:, root_qpos_address + 3 : root_qpos_address + 7] = (
        motion.root_quat_wxyz
    )

    csv_index_by_name = {
        name: index
        for index, name in enumerate(K1_REV1_MOTION_CSV_JOINT_NAMES)
    }
    for joint_name, qpos_address in zip(
        layout.joint_names, layout.joint_qpos_addresses, strict=True
    ):
        qpos[:, qpos_address] = motion.joint_pos_csv_order[
            :, csv_index_by_name[joint_name]
        ]
    return qpos


def differentiate_qpos(
    model: mujoco.MjModel, qpos: np.ndarray, fps: float
) -> np.ndarray:
    """Differentiate MuJoCo generalized positions, including the free quaternion."""

    qvel = np.zeros((qpos.shape[0], model.nv), dtype=np.float64)
    dt = 1.0 / fps
    for frame in range(qpos.shape[0]):
        if frame == 0:
            previous, following, interval = 0, 1, dt
        elif frame == qpos.shape[0] - 1:
            previous, following, interval = frame - 1, frame, dt
        else:
            previous, following, interval = frame - 1, frame + 1, 2.0 * dt
        mujoco.mj_differentiatePos(
            model, qvel[frame], interval, qpos[previous], qpos[following]
        )
    return qvel


def compute_mujoco_trajectory(
    model: mujoco.MjModel,
    layout: ModelLayout,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run MuJoCo forward kinematics and collect the MJLab motion schema."""

    frames = qpos.shape[0]
    body_count = len(layout.body_ids)
    body_pos_w = np.empty((frames, body_count, 3), dtype=np.float64)
    body_quat_w = np.empty((frames, body_count, 4), dtype=np.float64)
    body_lin_vel_w = np.empty((frames, body_count, 3), dtype=np.float64)
    body_ang_vel_w = np.empty((frames, body_count, 3), dtype=np.float64)
    data = mujoco.MjData(model)
    velocity = np.empty(6, dtype=np.float64)

    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        body_pos_w[frame] = data.xpos[layout.body_ids]
        body_quat_w[frame] = data.xquat[layout.body_ids]
        for output_index, body_id in enumerate(layout.body_ids):
            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                velocity,
                0,
            )
            body_ang_vel_w[frame, output_index] = velocity[:3]
            body_lin_vel_w[frame, output_index] = velocity[3:]

    return {
        "joint_pos": qpos[:, layout.joint_qpos_addresses].astype(np.float32),
        "joint_vel": qvel[:, layout.joint_dof_addresses].astype(np.float32),
        "body_pos_w": body_pos_w.astype(np.float32),
        "body_quat_w": body_quat_w.astype(np.float32),
        "body_lin_vel_w": body_lin_vel_w.astype(np.float32),
        "body_ang_vel_w": body_ang_vel_w.astype(np.float32),
    }


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate the basic converted-motion NPZ schema."""

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
    """Reconstruct model qpos from a named cyclo_mjlab motion NPZ."""

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
    root_qpos_address = int(model.jnt_qposadr[layout.root_joint_id])
    qpos[:, root_qpos_address : root_qpos_address + 3] = body_pos[
        :, root_body_index
    ]
    qpos[:, root_qpos_address + 3 : root_qpos_address + 7] = (
        _normalize_quaternions(body_quat[:, root_body_index])
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
    """Compare NPZ body poses against poses reconstructed by MuJoCo."""

    data = mujoco.MjData(model)
    maximum_position_error = 0.0
    maximum_orientation_error = 0.0
    expected_pos = motion_data["body_pos_w"]
    expected_quat = _normalize_quaternions(motion_data["body_quat_w"])

    for frame in range(qpos.shape[0]):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        actual_pos = data.xpos[layout.body_ids]
        actual_quat = _normalize_quaternions(data.xquat[layout.body_ids])
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


def save_motion_files(
    model: mujoco.MjModel,
    layout: ModelLayout,
    motion: Motion,
    npz_path: str | Path,
    csv_path: str | Path,
) -> tuple[float, float]:
    """Convert motion through MuJoCo and save numeric CSV plus motion NPZ.

    Returns the maximum position and orientation errors from an NPZ
    forward-kinematics round trip.
    """

    npz_path = Path(npz_path).expanduser().resolve()
    csv_path = Path(csv_path).expanduser().resolve()
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    qpos = motion_to_qpos(model, layout, motion)
    qvel = differentiate_qpos(model, qpos, motion.fps)
    trajectory = compute_mujoco_trajectory(model, layout, qpos, qvel)

    converted_csv = np.concatenate(
        (
            motion.root_pos,
            motion.root_quat_wxyz[:, [1, 2, 3, 0]],
            motion.joint_pos_csv_order,
        ),
        axis=1,
    )
    np.savetxt(csv_path, converted_csv, delimiter=",", fmt="%.9f")

    np.savez(
        npz_path,
        fps=np.asarray([motion.fps], dtype=np.int64),
        joint_names=np.asarray(layout.joint_names),
        body_names=np.asarray(layout.body_names),
        source_joint_names=np.asarray(K1_REV1_MOTION_CSV_JOINT_NAMES),
        **trajectory,
    )

    with np.load(npz_path, allow_pickle=False) as saved:
        saved_data = {key: saved[key].copy() for key in saved.files}
    reconstructed_qpos = qpos_from_npz(model, layout, saved_data)
    position_error, orientation_error = validate_forward_kinematics(
        model, layout, saved_data, reconstructed_qpos
    )

    print(f"[INFO] Converted CSV saved to: {csv_path}")
    print(f"[INFO] MuJoCo motion NPZ saved to: {npz_path}")
    print(
        "[INFO] NPZ layout: "
        f"frames={motion.frames}, joints={len(layout.joint_names)}, "
        f"bodies={len(layout.body_names)}"
    )
    print("[INFO] MuJoCo joint order:")
    for index, name in enumerate(layout.joint_names):
        print(f"  {index:02d}: {name}")
    print("[INFO] MuJoCo body order:")
    for index, name in enumerate(layout.body_names):
        print(f"  {index:02d}: {name}")
    print(
        "[INFO] Forward-kinematics round-trip: "
        f"max_position_error={position_error:.3e} m, "
        f"max_orientation_error={orientation_error:.3e} rad"
    )
    return position_error, orientation_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert K1 Rev.1 motion CSV to MuJoCo-ordered NPZ."
    )
    parser.add_argument(
        "--input_file", "-f", type=Path, required=True, help="Input K1 motion CSV."
    )
    parser.add_argument("--input_fps", type=int, default=50)
    parser.add_argument("--output_fps", type=int, default=50)
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="1-based inclusive input frame range.",
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
        "--root-quat-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Input CSV root quaternion order. Output NPZ always uses wxyz.",
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
    npz_path, csv_path = resolve_output_paths(args)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)
    layout = get_model_layout(model)
    motion = load_and_resample_csv(
        motion_file=args.input_file,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        frame_range=tuple(args.frame_range) if args.frame_range else None,
        root_quat_order=args.root_quat_order,
        smooth_window=args.smooth_window,
        smooth_passes=args.smooth_passes,
        smooth_fields=tuple(args.smooth_fields),
    )
    save_motion_files(
        model=model,
        layout=layout,
        motion=motion,
        npz_path=npz_path,
        csv_path=csv_path,
    )


if __name__ == "__main__":
    main()
