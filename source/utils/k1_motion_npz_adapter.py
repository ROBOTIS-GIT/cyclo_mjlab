"""Validated 50 Hz Cyclo/MuJoCo reference adapter shared by policy families.

Complete named body poses and velocities are checked against the current K1
MuJoCo model before root states reach the learner.
Root-only files must explicitly declare their coordinate contract.
"""

import glob
import hashlib
import io
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

FIELDS = {
    "root_pos_w": 3,
    "root_quat_w": 4,
    "root_lin_vel_w": 3,
    "root_ang_vel_w": 3,
}
BODY_FIELDS = {
    "body_pos_w": 3,
    "body_quat_w": 4,
    "body_lin_vel_w": 3,
    "body_ang_vel_w": 3,
}
ROOT_CONTRACT = {
    "backend": "mujoco",
    "quaternion_order": "wxyz",
    "velocity_frame": "world",
    "velocity_point": "link_origin",
}
ADAPTER_VERSION = "cyclo-k1-reference-v2"


def _require(data, keys):
    missing = sorted(set(keys) - data.keys())
    if missing:
        raise ValueError(
            f"Reference is missing required fields: {', '.join(missing)}"
        )


def _names(value, field):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "US":
        raise ValueError(f"{field} must be a one-dimensional string array")
    names = tuple(
        n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in array
    )
    if len(names) != len(set(names)):
        raise ValueError(f"{field} contains duplicate names")
    return names


def _text(value, key):
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "US":
        raise ValueError(f"{key} must be a non-empty scalar string")
    text = array.item()
    text = text.decode("utf-8") if isinstance(text, bytes) else str(text)
    if not text:
        raise ValueError(f"{key} must be a non-empty scalar string")
    return text


def _metadata(data, contract, required=False):
    if required:
        _require(data, contract)
    for key, expected in contract.items():
        if key in data and _text(data[key], key).lower() != expected:
            raise ValueError(
                f"Unsupported {key}: {data[key]!r}; expected {expected!r}. No"
                " coordinate/backend guessing is performed"
            )


def _array(data, key, shape):
    original = np.asarray(data[key])
    if original.dtype.kind not in "iuf":
        raise ValueError(
            f"{key} must contain real numeric values, got {original.dtype}"
        )
    value = original.astype(np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{key} must be finite with shape {shape}")
    return value


def validate_motion(data, joint_names, fps=50.0):
    _require(data, ("joint_names", "fps", "joint_pos", "joint_vel", *FIELDS))
    names = _names(data["joint_names"], "joint_names")
    if set(names) != set(joint_names):
        raise ValueError(
            "Motion must name every K1 joint exactly once;"
            f" missing={sorted(set(joint_names) - set(names))},"
            f" extra={sorted(set(names) - set(joint_names))}"
        )
    rate = np.asarray(data["fps"])
    if (
        rate.size != 1
        or rate.dtype.kind not in "iuf"
        or not np.isfinite(rate.item())
        or not np.isclose(float(rate.item()), fps, rtol=0, atol=1e-6)
    ):
        raise ValueError(
            f"Motion fps must be {fps}; adapter does not resample (got"
            f" {rate!r})"
        )
    q = np.asarray(data["joint_pos"])
    if q.ndim != 2 or len(q) < 2 or q.shape[1] != len(names):
        raise ValueError(
            "joint_pos must have shape (frames >= 2, named joints)"
        )
    order = [names.index(n) for n in joint_names]
    result = {
        "fps": np.array(float(rate.item())),
        "joint_names": np.asarray(joint_names),
    }
    for key, width in {
        **FIELDS,
        "joint_pos": len(names),
        "joint_vel": len(names),
    }.items():
        value = _array(data, key, (len(q), width))
        with np.errstate(over="ignore", invalid="ignore"):
            converted = (
                value[:, order] if key.startswith("joint_") else value
            ).astype(np.float32)
        if not np.isfinite(converted).all():
            raise ValueError(
                f"{key} cannot be represented as finite float32 loader data"
            )
        result[key] = converted
    if not np.allclose(
        np.linalg.norm(result["root_quat_w"], axis=-1), 1, rtol=0, atol=1e-3
    ):
        raise ValueError("root_quat_w must be normalized wxyz quaternions")
    return result


def model_joint_names(model):
    import mujoco

    return tuple(
        model.joint(i).name.removeprefix("robot/")
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    )


def _from_body_reference(data, model):
    import mujoco

    _require(
        data,
        (
            "body_names",
            "joint_names",
            "fps",
            "joint_pos",
            "joint_vel",
            *BODY_FIELDS,
        ),
    )
    _metadata(
        data,
        {
            "backend": "mujoco",
            "quaternion_order": "wxyz",
            "velocity_frame": "world",
        },
    )
    bodies = _names(data["body_names"], "body_names")
    model_bodies = tuple(model.body(i).name for i in range(1, model.nbody))
    if set(bodies) != set(model_bodies):
        raise ValueError(
            "Body reference must name all K1 MJCF bodies;"
            f" missing={sorted(set(model_bodies) - set(bodies))},"
            f" extra={sorted(set(bodies) - set(model_bodies))}"
        )
    if "pelvis" not in bodies:
        raise ValueError("Body reference requires named pelvis")
    q = np.asarray(data["joint_pos"])
    if q.ndim != 2 or len(q) < 2:
        raise ValueError(
            "joint_pos must have shape (frames >= 2, named joints)"
        )
    frames = len(q)
    states = {
        key: _array(data, key, (frames, len(bodies), width))
        for key, width in BODY_FIELDS.items()
    }
    if not np.allclose(
        np.linalg.norm(states["body_quat_w"], axis=-1), 1, rtol=0, atol=1e-3
    ):
        raise ValueError(
            "body_quat_w must contain normalized wxyz quaternions"
        )
    root = bodies.index("pelvis")
    canonical = {
        **data,
        **{
            target: states[source][:, root]
            for target, source in zip(FIELDS, BODY_FIELDS, strict=True)
        },
    }
    result = validate_motion(canonical, model_joint_names(model))
    hinges = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
    free = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    if (
        len(free) != 1
        or model.body(int(model.jnt_bodyid[free[0]])).name != "pelvis"
    ):
        raise ValueError("Adapter requires the K1 pelvis free-joint model")
    qa, va = int(model.jnt_qposadr[free[0]]), int(model.jnt_dofadr[free[0]])
    qpos = np.tile(model.qpos0, (frames, 1))
    qpos[:, qa: qa + 3] = result["root_pos_w"]
    qpos[:, qa + 3: qa + 7] = result["root_quat_w"]
    qpos[:, model.jnt_qposadr[hinges]] = result["joint_pos"]
    body_ids = [model.body(name).id for name in bodies]
    mj_data = mujoco.MjData(model)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    profiles = {
        "link_origin": np.zeros((len(bodies), 3)),
        "current_k1_com": model.body_ipos[body_ids].copy(),
    }
    if "velocity_point" in data:
        point = _text(data["velocity_point"], "velocity_point").lower()
        if point == "link_origin":
            profiles = {"link_origin": profiles["link_origin"]}
        elif point in ("body_com", "mujoco_body"):
            profiles.pop("link_origin")
        else:
            raise ValueError(f"Unsupported body velocity_point {point!r}")
    root_linear = {name: np.empty((frames, 3)) for name in profiles}
    root_angular = states["body_ang_vel_w"][:, root].copy()
    errors = {name: 0.0 for name in profiles}
    for i in range(frames):
        mj_data.qpos[:], mj_data.qvel[:] = qpos[i], 0
        mj_data.qvel[model.jnt_dofadr[hinges]] = result["joint_vel"][i]
        mujoco.mj_forward(model, mj_data)
        if not np.allclose(
            mj_data.xpos[body_ids], states["body_pos_w"][i], rtol=0, atol=2e-4
        ):
            raise ValueError(
                "body_pos_w does not match K1 MuJoCo body origins at frame"
                f" {i}; model/backend/frame mismatch"
            )
        expected, actual = mj_data.xquat[body_ids], states["body_quat_w"][i]
        error = np.minimum(
            np.linalg.norm(expected - actual, axis=-1),
            np.linalg.norm(expected + actual, axis=-1),
        )
        if error.max() > 2e-4:
            raise ValueError(
                f"body_quat_w does not match K1 wxyz FK at frame {i};"
                " quaternion order or model mismatch"
            )
        rotations = mj_data.xmat[body_ids].reshape(-1, 3, 3)
        mujoco.mj_jacBody(model, mj_data, jp, jr, body_ids[root])
        root_jacobian = np.concatenate(
            [jp[:, va: va + 6], jr[:, va: va + 6]]
        )
        for profile, offsets in profiles.items():
            root_offset = rotations[root] @ offsets[root]
            root_linear[profile][i] = states["body_lin_vel_w"][
                i, root
            ] - np.cross(root_angular[i], root_offset)
            desired = np.concatenate(
                [root_linear[profile][i], root_angular[i]]
            )
            mj_data.qvel[va: va + 6] = np.linalg.solve(root_jacobian, desired)
            for row, body in enumerate(body_ids):
                mujoco.mj_jacBody(model, mj_data, jp, jr, body)
                angular = jr @ mj_data.qvel
                linear = jp @ mj_data.qvel + np.cross(
                    angular, rotations[row] @ offsets[row]
                )
                errors[profile] = max(
                    errors[profile],
                    float(
                        np.max(
                            np.abs(angular - states["body_ang_vel_w"][i, row])
                        )
                    ),
                    float(
                        np.max(
                            np.abs(linear - states["body_lin_vel_w"][i, row])
                        )
                    ),
                )
    matches = [name for name, error in errors.items() if error <= 2e-4]
    if not matches:
        raise ValueError(
            "No supported world velocity-point/producer profile matches all"
            f" frames and bodies; maximum errors={errors}. Regenerate the NPZ"
            " from its CSV with the current K1 model using csv_to_npz; no"
            " tolerance relaxation or velocity guessing is performed"
        )
    reference = root_linear[matches[0]]
    if any(
        not np.allclose(reference, root_linear[name], rtol=0, atol=1e-5)
        for name in matches[1:]
    ):
        raise ValueError(
            f"Ambiguous velocity-point profiles {matches} produce different"
            " root-origin velocities; producer metadata is required"
        )
    result["root_lin_vel_w"] = reference.astype(np.float32)
    result["root_ang_vel_w"] = root_angular.astype(np.float32)
    result["source_velocity_profiles"] = np.asarray(matches)
    result["profile_max_errors"] = np.asarray(
        [errors[name] for name in matches]
    )
    return result


def adapt_motion(data, joint_names, model=None):
    if any(key in data for key in BODY_FIELDS):
        if any(key in data for key in FIELDS):
            raise ValueError(
                "Mixed body-state/root-state schema is ambiguous; provide one"
                " complete schema"
            )
        if model is None:
            raise ValueError(
                "A K1 MuJoCo model is required to verify a body reference"
            )
        result = _from_body_reference(data, model)
    else:
        _metadata(data, ROOT_CONTRACT, required=True)
        result = validate_motion(data, joint_names)
    for key in ("producer", "backend"):
        if key in data:
            result["source_" + key] = np.asarray(_text(data[key], key))
        else:
            result["source_" + key] = np.asarray("unspecified")
    result.update(
        {key: np.asarray(value) for key, value in ROOT_CONTRACT.items()}
    )
    result["producer"] = np.asarray(ADAPTER_VERSION)
    return result


def convert_motion(input_path, output_path, joint_names, model=None):
    with np.load(input_path, allow_pickle=False) as archive:
        result = adapt_motion(dict(archive), joint_names, model)
    output_path = Path(output_path)
    if Path(input_path).resolve() == output_path.resolve():
        raise ValueError(
            "Choose a separate output; source motions are preserved"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        np.savez_compressed(stream, **result)
    return output_path


def _resolve_motion_sources(source):
    files = []
    for pattern in source.split(","):
        expanded = os.path.expanduser(pattern.strip())
        matches = (
            [expanded]
            if Path(expanded).is_file()
            else sorted(glob.glob(expanded))
        )
        if not matches:
            raise ValueError(f"No reference files matched {pattern!r}")
        files.extend(Path(file).resolve() for file in matches)
    return files


def _read_and_adapt_motion(file, model):
    """
    Hash and validate the same source bytes before writing any cache files.
    """
    try:
        raw = file.read_bytes()
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            result = adapt_motion(
                dict(archive), model_joint_names(model), model
            )
        return result, hashlib.sha256(raw).hexdigest()
    except (ValueError, KeyError, OSError) as error:
        raise ValueError(f"Reference {file}: {error}") from error


def _motion_cache_directory(files, hashes, model, cache_dir):
    signature = hashlib.sha256()
    for attribute in (
        "body_pos",
        "body_quat",
        "body_ipos",
        "jnt_axis",
        "jnt_pos",
        "jnt_qposadr",
        "qpos0",
    ):
        signature.update(np.asarray(getattr(model, attribute)).tobytes())
    model_hash = signature.hexdigest()
    identity = json.dumps(
        [
            (str(path), digest)
            for path, digest in zip(files, hashes, strict=True)
        ]
    )
    group = hashlib.sha256(
        (ADAPTER_VERSION + model_hash + identity).encode()
    ).hexdigest()
    return Path(cache_dir) / group, model_hash


@contextmanager
def _atomic_output(target, mode):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode=mode,
            dir=target.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            yield temporary
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_motion_library(folder, files, results, hashes, model_hash):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (file, result) in enumerate(zip(files, results, strict=True)):
        safe_stem = re.sub(r"[^a-zA-Z0-9_.-]", "_", file.stem)
        target = folder / f"{i:04d}-{safe_stem}.npz"
        with _atomic_output(target, "w+b") as temporary:
            np.savez_compressed(temporary, **result)
        paths.append(str(target.resolve()))
    manifest = {
        "adapter": ADAPTER_VERSION,
        "model_signature": model_hash,
        "sources": [
            {"path": str(p), "sha256": h}
            for p, h in zip(files, hashes, strict=True)
        ],
        "outputs": paths,
    }
    with _atomic_output(folder / "manifest.json", "w") as temporary:
        json.dump(manifest, temporary, indent=2)
        temporary.write("\n")
    return paths[0] if len(paths) == 1 else str(folder.resolve() / "*.npz")


def prepare_motion_files(source, model, cache_dir):
    """
    Validate references, then publish the internal normalized loader library.
    """
    files = _resolve_motion_sources(source)
    # A cache path never bypasses source/model validation.
    prepared = [_read_and_adapt_motion(file, model) for file in files]
    results, hashes = zip(*prepared, strict=True)
    folder, model_hash = _motion_cache_directory(
        files, hashes, model, cache_dir
    )
    return _write_motion_library(folder, files, results, hashes, model_hash)
