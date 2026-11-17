import mujoco
import numpy as np
import torch
from supermimic.utils.analytic_ik import AnalyticIK

from source.tasks.supermimic.motion import model_joint_names
from source.tasks.supermimic.tasks import (
    TASKS,
    TRACKED_BODIES,
    k1_env_cfg,
    register_tasks,
)


def test_registry_and_robot_contract():
    from mjlab.tasks.registry import list_tasks, load_env_cfg

    if not set(TASKS).issubset(list_tasks()):
        register_tasks()
    for task in TASKS:
        cfg = load_env_cfg(task)
        assert cfg.commands["motion"].body_names == TRACKED_BODIES
        assert cfg.commands["motion"].anchor_body_name == "pelvis"
        assert cfg.sim.mujoco.timestep * cfg.decimation == 0.02
        assert set(cfg.events) == {"foot_friction", "base_com", "encoder_bias"}
        assert (
            len(model_joint_names(cfg.scene.entities["robot"].spec_fn().compile()))
            == 23
        )
        assert load_env_cfg(task, play=True).scene.num_envs == 1


def test_k1_maxcoord_ik_roundtrip_and_gradient():
    model = k1_env_cfg().scene.entities["robot"].spec_fn().compile()
    data = mujoco.MjData(model)
    hinges = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
    qadr = model.jnt_qposadr[hinges]
    # Moderate, non-zero joint states in range exercise each chain's signs.
    q = np.array([np.clip(0.15, *model.jnt_range[j]) for j in hinges])
    v = np.linspace(-0.15, 0.15, 23)
    data.qpos[qadr] = q
    data.qvel[model.jnt_dofadr[hinges]] = v
    root_quat = np.array([0.8, 0.2, -0.3, 0.4])
    root_quat /= np.linalg.norm(root_quat)
    data.qpos[3:7] = root_quat
    data.qvel[:6] = [0.3, -0.2, 0.1, 0.4, 0.2, -0.3]
    mujoco.mj_forward(model, data)
    tracked = np.array([model.body(n).id for n in TRACKED_BODIES])
    # Entity body array excludes world; addresses are model qpos columns.
    body_ids = torch.arange(1, model.nbody)
    ik = AnalyticIK(
        model,
        body_ids,
        torch.tensor(qadr),
        torch.tensor(tracked - 1),
        "cpu",
        torch.float64,
    )
    rotations = torch.tensor(
        data.xmat[tracked].reshape(1, -1, 3, 3), requires_grad=True
    )
    angular = torch.tensor(data.cvel[tracked, :3].reshape(1, -1, 3))
    angles, rates = ik(rotations, angular)
    np.testing.assert_allclose(angles.detach().numpy()[0], q, atol=1e-6)
    np.testing.assert_allclose(rates.detach().numpy()[0], v, atol=1e-6)
    # The representation supplies root-local rotations and angular velocities;
    # the same solver must recover native joint order in either common frame.
    root_rotation = torch.tensor(data.xmat[tracked[0]].reshape(3, 3))
    local_rotations = root_rotation.T @ rotations
    local_angular = (root_rotation.T @ angular.unsqueeze(-1)).squeeze(-1)
    local_angles, local_rates = ik(local_rotations, local_angular)
    torch.testing.assert_close(local_angles, angles)
    torch.testing.assert_close(local_rates, rates)
    from source.tasks.supermimic.forward_kinematics import ForwardKinematics

    fk = ForwardKinematics(
        model,
        body_ids,
        torch.tensor(qadr),
        torch.tensor(tracked - 1),
        dtype=torch.float64,
    )
    q_tensor = torch.tensor(q[None], requires_grad=True)
    root_vel = torch.tensor(data.qvel[:3][None])
    root_omega = torch.tensor(
        (data.xmat[tracked[0]].reshape(3, 3) @ data.qvel[3:6])[None]
    )
    pos, quat, linear, omega = fk(
        torch.tensor(data.qpos[:3][None]),
        torch.tensor(root_quat[None]),
        q_tensor,
        root_vel,
        root_omega,
        torch.tensor(v[None]),
    )
    np.testing.assert_allclose(pos.detach()[0].numpy(), data.xpos[tracked], atol=1e-6)
    np.testing.assert_allclose(
        omega.detach()[0].numpy(), data.cvel[tracked, :3], atol=1e-6
    )
    for row, body in enumerate(tracked):
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, int(body), velocity, 0
        )
        # mj_objectVelocity is at the body COM; FK returns the link origin.
        origin_linear = velocity[3:] - np.cross(
            velocity[:3], data.xipos[body] - data.xpos[body]
        )
        np.testing.assert_allclose(
            linear.detach()[0, row].numpy(), origin_linear, atol=1e-6
        )
    pos.square().sum().backward()
    assert torch.isfinite(q_tensor.grad).all() and q_tensor.grad.abs().sum() > 0
    angles.square().sum().backward()
    assert torch.isfinite(rotations.grad).all()
    assert rotations.grad.abs().sum() > 0
