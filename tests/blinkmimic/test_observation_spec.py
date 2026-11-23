"""Compare exported field packing with the unmodified native MDP and motion library."""

from types import SimpleNamespace

import pytest
import torch

from source.tasks.blinkmimic.runtime import prepare

prepare()
from blinkmimic.mdp import MDP
from blinkmimic.motion import MotionLib
from blinkmimic.rotations import quat_to_mat, yaw_mat

from source.tasks.blinkmimic.config.sim2real import native_dimensions, observation_config


def pack_terms(group, values):
    parts = []
    for name, cfg in group["terms"].items():
        value = values[name]
        if cfg["clip"] is not None:
            value = value.clamp(*cfg["clip"])
        parts.append(value * torch.tensor(cfg["scale"]))
    return torch.cat(parts, dim=-1)


@pytest.mark.parametrize("joints,future", [(7, 1), (7, 10), (23, 1), (23, 10)])
def test_exported_terms_match_native_values_order_and_clip_boundaries(joints, future):
    torch.manual_seed(735)
    frames = 13
    motion = object.__new__(MotionLib)
    motion.q = torch.randn(frames, joints)
    motion.qd = torch.randn(frames, joints) * 7
    motion.root_p = torch.randn(frames, 3)
    motion.root_R = quat_to_mat(torch.randn(frames, 4))
    motion.root_v_local = torch.randn(frames, 3)
    motion.root_w_local = torch.randn(frames, 3)
    motion.clip_id = torch.tensor([0] * 5 + [1] * 8)
    motion.clip_end = torch.tensor([4, 12])
    current = torch.tensor([0, 3, 4, 5, 11, 12])
    batch = len(current)
    q0 = torch.randn(joints)
    q, qd = torch.randn(batch, joints), torch.randn(batch, joints)
    rotation = quat_to_mat(torch.randn(batch, 4))
    omega = torch.randn(batch, 3)
    previous_action = torch.randn(batch, joints)
    xi = torch.randn(batch, joints + 1) * 0.1
    state = (torch.randn(batch, 3), rotation, torch.randn(batch, 3), omega, q, qd)
    native_obs = MDP.obs(SimpleNamespace(default_q=q0, motion=motion),
                         state, previous_action, xi, current)
    cfg = observation_config(3, future, joints)
    relative = rotation.transpose(-1, -2) @ motion.root_R[current]
    values = dict(
        joint_pos_rel=q - q0 - xi[:, 1:], joint_vel=qd,
        base_ang_vel=(rotation.transpose(-1, -2) @ omega.unsqueeze(-1)).squeeze(-1),
        projected_gravity=-rotation[:, 2, :], last_action=previous_action,
        reference_orientation=relative[:, :, :2].reshape(batch, 6),
    )
    torch.testing.assert_close(pack_terms(cfg["history"], values), native_obs, rtol=0, atol=0)

    goal_cfg = cfg["motion_goal"]
    selected = current[:, None] + goal_cfg["start_offset"] + torch.arange(future) * goal_cfg["stride"]
    selected = selected.minimum(motion.clip_end[motion.clip_id[current]][:, None])
    torch.testing.assert_close(selected, motion.future_frames(current, future))
    heading_inverse = yaw_mat(motion.root_R[current]).transpose(-1, -2)[:, None]
    displacement = motion.root_p[selected] - motion.root_p[current][:, None]
    orientation = heading_inverse @ motion.root_R[selected]
    values = dict(
        future_root_displacement=(heading_inverse @ displacement.unsqueeze(-1)).squeeze(-1),
        future_root_orientation=orientation[..., :2].reshape(batch, future, 6),
        future_root_linear_velocity=motion.root_v_local[selected],
        future_root_angular_velocity=motion.root_w_local[selected],
        future_joint_pos=motion.q[selected], future_joint_vel=motion.qd[selected],
    )
    packed = pack_terms(goal_cfg["groups"][0], values).reshape(batch, -1)
    torch.testing.assert_close(packed, motion.goal(current, future), rtol=0, atol=0)
    obs_width, goal_width = native_dimensions(joints)
    assert native_obs.shape[-1] == obs_width
    assert packed.shape[-1] == motion.goal_dim(future) == future * goal_width


def test_export_rejects_changed_upstream_dimensions(monkeypatch):
    import blinkmimic.mdp

    monkeypatch.setattr(blinkmimic.mdp, "obs_dim", lambda joints: 4)
    with pytest.raises(ValueError, match="differ from upstream"):
        observation_config(10, 10, 23)
