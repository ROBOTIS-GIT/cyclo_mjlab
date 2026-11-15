import hashlib

import mujoco
import numpy as np
import pytest
import torch
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.forward_kinematics import ForwardKinematics

from source.tasks.supermimic import REPO_ROOT
from source.tasks.supermimic.motion import (
    adapt_motion,
    model_joint_names,
    prepare_motion_files,
)
from source.tasks.supermimic.robot import get_k1_rev1_cfg


@pytest.fixture(scope="module")
def model():
    return get_k1_rev1_cfg().spec_fn().compile()


def load_clip(dance="dance1"):
    path = REPO_ROOT / "source/assets/motions/K1_rev1" / dance / f"{dance}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return path, dict(archive)


@pytest.mark.parametrize("dance", ["dance1", "dance2"])
def test_real_reference_preserved_and_native_loader_accepts(dance, model, tmp_path):
    path, data = load_clip(dance)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    prepared = prepare_motion_files(str(path), model, tmp_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with np.load(prepared, allow_pickle=False) as output:
        assert output["fps"].item() == data["fps"].item() == 50
        assert output["joint_pos"].shape == data["joint_pos"].shape
        root = list(data["body_names"]).index("pelvis")
        np.testing.assert_array_equal(output["root_pos_w"], data["body_pos_w"][:, root])
        np.testing.assert_array_equal(
            output["root_quat_w"], data["body_quat_w"][:, root]
        )
        assert output["source_velocity_profiles"].tolist() == ["legacy_k1_874c7bd_com"]
        assert output["profile_max_errors"].max() < 2e-4
    hinges = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
    fk = ForwardKinematics(
        model,
        torch.arange(1, model.nbody),
        torch.tensor(model.jnt_qposadr[hinges]),
        torch.arange(model.nbody - 1),
    )
    loader = MotionLoader(prepared, fk)
    positions, quats, linear, angular = loader.body_state(
        torch.arange(loader.time_step_total)
    )
    assert all(torch.isfinite(t).all() for t in (positions, quats, linear, angular))
    body_order = [
        list(data["body_names"]).index(model.body(i).name)
        for i in range(1, model.nbody)
    ]
    np.testing.assert_allclose(
        positions.numpy(), data["body_pos_w"][:, body_order], atol=2e-4
    )
    np.testing.assert_allclose(
        angular.numpy(), data["body_ang_vel_w"][:, body_order], atol=2e-4
    )
    from mjlab.utils.lab_api.math import matrix_from_quat

    from source.tasks.supermimic.reference_profiles import LEGACY_K1_COM

    local_com = torch.tensor(
        [LEGACY_K1_COM[model.body(i).name] for i in range(1, model.nbody)],
        dtype=linear.dtype,
    )
    world_com = (matrix_from_quat(quats) @ local_com[None, :, :, None]).squeeze(-1)
    reconstructed_com_velocity = linear + torch.linalg.cross(angular, world_com)
    np.testing.assert_allclose(
        reconstructed_com_velocity.numpy(),
        data["body_lin_vel_w"][:, body_order],
        atol=2e-4,
    )


def test_permuted_joint_and_body_order(model):
    _, data = load_clip()
    original = adapt_motion(data, model_joint_names(model), model)
    joint_order, body_order = np.arange(23)[::-1], np.arange(25)[::-1]
    data["joint_names"] = data["joint_names"][joint_order]
    data["body_names"] = data["body_names"][body_order]
    for key in ("joint_pos", "joint_vel"):
        data[key] = data[key][:, joint_order]
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        data[key] = data[key][:, body_order]
    permuted = adapt_motion(data, model_joint_names(model), model)
    for key in (
        "joint_pos",
        "joint_vel",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
    ):
        np.testing.assert_array_equal(permuted[key], original[key])


@pytest.mark.parametrize(
    "failure",
    ["xyzw", "backend", "body_name", "unknown_point", "velocity", "frame", "fps"],
)
def test_incompatible_inputs_rejected(failure, model):
    _, data = load_clip()
    if failure == "xyzw":
        data["body_quat_w"] = data["body_quat_w"][:, :, [1, 2, 3, 0]]
    elif failure == "backend":
        data["backend"] = np.asarray("isaacsim")
    elif failure == "body_name":
        data["body_names"][1] = "unknown"
    elif failure == "unknown_point":
        data["velocity_point"] = np.asarray("unknown")
    elif failure == "velocity":
        data["body_lin_vel_w"][:, 3, 0] += 0.02
    elif failure == "frame":
        data["velocity_frame"] = np.asarray("local")
    else:
        data["fps"] = np.asarray(30)
    with pytest.raises(ValueError):
        adapt_motion(data, model_joint_names(model), model)


def test_root_only_requires_coordinate_metadata(model):
    _, data = load_clip()
    output = adapt_motion(data, model_joint_names(model), model)
    del output["quaternion_order"]
    with pytest.raises(ValueError, match="quaternion_order"):
        adapt_motion(output, model_joint_names(model), model)


def test_every_glob_must_match(model, tmp_path):
    path, _ = load_clip()
    with pytest.raises(ValueError, match="No reference files matched"):
        prepare_motion_files(
            f"{path},{tmp_path}/missing*.npz", model, tmp_path / "cache"
        )


def test_current_mujoco_producer_com_profile(model):
    _, data = load_clip()
    for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        data[key] = data[key][100:120].copy()
    canonical = adapt_motion(data, model_joint_names(model), model)
    body_ids = [model.body(str(name)).id for name in data["body_names"]]
    hinges = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
    simulation = mujoco.MjData(model)
    velocity = np.empty(6)
    for i in range(20):
        simulation.qpos[:3] = canonical["root_pos_w"][i]
        simulation.qpos[3:7] = canonical["root_quat_w"][i]
        simulation.qpos[model.jnt_qposadr[hinges]] = canonical["joint_pos"][i]
        mujoco.mj_forward(model, simulation)
        simulation.qvel[:3] = canonical["root_lin_vel_w"][i]
        simulation.qvel[3:6] = (
            simulation.xmat[model.body("pelvis").id].reshape(3, 3).T
            @ canonical["root_ang_vel_w"][i]
        )
        simulation.qvel[model.jnt_dofadr[hinges]] = canonical["joint_vel"][i]
        mujoco.mj_forward(model, simulation)
        for row, body in enumerate(body_ids):
            mujoco.mj_objectVelocity(
                model, simulation, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
            )
            data["body_ang_vel_w"][i, row] = velocity[:3]
            data["body_lin_vel_w"][i, row] = velocity[3:]
    current = adapt_motion(data, model_joint_names(model), model)
    assert current["source_velocity_profiles"].tolist() == ["current_k1_com"]
    np.testing.assert_allclose(
        current["root_lin_vel_w"], canonical["root_lin_vel_w"], atol=2e-6
    )


def test_multiple_references_keep_clip_boundaries(model, tmp_path):
    import glob

    first, data1 = load_clip("dance1")
    second, data2 = load_clip("dance2")
    prepared = prepare_motion_files(f"{first},{second}", model, tmp_path)
    assert len(glob.glob(prepared)) == 2
    hinges = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
    fk = ForwardKinematics(
        model,
        torch.arange(1, model.nbody),
        torch.tensor(model.jnt_qposadr[hinges]),
        torch.arange(model.nbody - 1),
    )
    loader = MotionLoader(prepared, fk)
    assert loader.motion_ends.tolist() == [
        len(data1["joint_pos"]),
        len(data1["joint_pos"]) + len(data2["joint_pos"]),
    ]
