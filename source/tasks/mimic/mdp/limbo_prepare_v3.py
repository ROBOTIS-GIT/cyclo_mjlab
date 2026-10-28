"""Preparation shaping aligned with every unchanged readiness condition."""
import torch
from . import limbo_prepare_v1 as v1, limbo_prepare_v2 as v2

command = v2.command

def gate_errors(env):
    c = command(env)
    _, joint, feet, speed = c.readiness()
    orientation = v1.quat_error_magnitude(c.robot_anchor_quat_w, c.anchor_quat_w)
    joint_speed = c.robot_joint_vel.square().mean(-1).sqrt()
    omega = torch.linalg.vector_norm(c.robot_body_ang_vel_w[:, 0], dim=-1)
    height = (c.robot_body_pos_w[:, 0, 2] - c.body_pos_w[:, 0, 2]).abs()
    contacts = torch.stack([
        torch.linalg.vector_norm(env.scene[f'limbo_{side}_contact'].data.force, dim=-1)
        .reshape(env.num_envs, -1).amax(-1) > 5 for side in ('left', 'right')
    ], -1).all(-1)
    return torch.stack((joint/.1, feet/.07, c.relative_foot_error()/.04,
                        orientation/.15, joint_speed/.5, speed/.15,
                        omega/.4, height/.05, (~contacts).float()), -1)

def preparation_progress(env):
    c = command(env)
    # A potential difference, never a constant reward for waiting.
    cost = gate_errors(env).clamp(max=10).sum(-1)
    delta = torch.where(c.progress_valid & c.preparing,
                        c.progress_previous - cost, torch.zeros_like(cost))
    c.progress_previous[:] = cost.detach()
    c.progress_valid[:] = c.preparing
    return delta / env.step_dt

def preparation_orientation_cost(env):
    c = command(env)
    return gate_errors(env)[:, 3].clamp(max=5).square() * c.preparing

def preparation_stability_cost(env):
    c = command(env)
    errors = gate_errors(env)
    return errors[:, 4:].clamp(max=5).square().mean(-1) * c.preparing
