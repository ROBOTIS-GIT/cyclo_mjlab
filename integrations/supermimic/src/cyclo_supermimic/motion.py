"""Validate and convert Cyclo full-body NPZ clips to research MotionLoader data."""

from pathlib import Path

import numpy as np

FIELDS = {"root_pos_w": 3, "root_quat_w": 4, "root_lin_vel_w": 3, "root_ang_vel_w": 3}


def validate_motion(data, joint_names, fps=50.0):
    names = tuple(str(n) for n in data["joint_names"])
    if len(names) != len(set(names)) or set(names) != set(joint_names):
        raise ValueError("Motion must name every K1 joint exactly once")
    actual_fps = float(np.asarray(data["fps"]).item())
    if not np.isfinite(actual_fps) or not np.isclose(actual_fps, fps):
        raise ValueError(
            f"Motion fps must be {fps}, got {actual_fps}; resample with csv_to_npz.py"
        )
    q = np.asarray(data["joint_pos"])
    if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] != len(names):
        raise ValueError("joint_pos must have shape (frames >= 2, named joints)")
    frames = len(q)
    order = [names.index(n) for n in joint_names]
    result = {"fps": np.array(fps), "joint_names": np.asarray(joint_names)}
    for key, width in {
        **FIELDS,
        "joint_pos": len(names),
        "joint_vel": len(names),
    }.items():
        value = np.asarray(data[key], dtype=np.float32)
        if value.shape != (frames, width) or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite with shape {(frames, width)}")
        result[key] = value[:, order] if key.startswith("joint_") else value
    if not np.allclose(np.linalg.norm(result["root_quat_w"], axis=-1), 1, atol=1e-3):
        raise ValueError("root_quat_w must be normalized wxyz quaternions")
    return result


def convert_motion(input_path, output_path, joint_names):
    with np.load(input_path, allow_pickle=False) as archive:
        data = dict(archive)
    if "root_pos_w" not in data:
        bodies = tuple(str(n) for n in data["body_names"])
        if bodies.count("pelvis") != 1:
            raise ValueError("Cyclo motion must name exactly one pelvis body")
        root = bodies.index("pelvis")
        for target, source in zip(
            FIELDS,
            ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"),
            strict=True,
        ):
            data[target] = data[source][:, root]
    result = validate_motion(data, joint_names)
    output_path = Path(output_path)
    if Path(input_path).resolve() == output_path.resolve():
        raise ValueError("Choose a separate output; source motions are preserved")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        np.savez_compressed(stream, **result)
    return output_path


def model_joint_names(model):
    import mujoco

    return tuple(
        model.joint(i).name.removeprefix("robot/")
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    )
