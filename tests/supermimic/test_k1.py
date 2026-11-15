import mujoco
import numpy as np
import torch
from cyclo_supermimic.motion import model_joint_names
from cyclo_supermimic.tasks import TASKS, TRACKED_BODIES, k1_env_cfg, register_tasks
from supermimic.utils.analytic_ik import AnalyticIK


def test_registry_and_robot_contract():
    from mjlab.tasks.registry import load_env_cfg

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
    data.qpos[3] = 1
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
    angles.square().sum().backward()
    assert torch.isfinite(rotations.grad).all()
    assert rotations.grad.abs().sum() > 0
