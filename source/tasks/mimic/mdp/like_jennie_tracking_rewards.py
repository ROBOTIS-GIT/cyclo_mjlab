"""Bounded tracking scores for the Like Jennie refinement experiment."""
import torch


def joint_tracking_score(env, std=0.35):
    command=env.command_manager.get_term('reference_trajectory')
    error=command.robot_joint_pos-command.joint_pos
    return torch.exp(-error.square().mean(dim=-1)/(std*std))


def world_foot_tracking_score(env, std=0.25):
    command=env.command_manager.get_term('reference_trajectory')
    feet=[command.cfg.body_names.index(name) for name in ('left_ankle_roll_link','right_ankle_roll_link')]
    # Both positions are in world coordinates and include the same env origin.
    error=command.robot_body_pos_w[:,feet]-command.body_pos_w[:,feet]
    return torch.exp(-error.square().sum(dim=-1).mean(dim=-1)/(std*std))
