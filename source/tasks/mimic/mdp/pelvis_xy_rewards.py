# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Pelvis world-path tracking, independent of the per-step anchor alignment."""

import math

import torch


def xy_position_score(actual, reference, std: float = 0.5) -> torch.Tensor:
  if not math.isfinite(std) or std <= 0:
    raise ValueError("Pelvis XY std must be finite and positive")
  error = (actual[..., :2] - reference[..., :2]).square().sum(-1)
  return torch.nan_to_num(
    torch.exp(-error / std**2), nan=0.0, posinf=0.0, neginf=0.0
  )


def reference_pelvis_xy(
  env, command_name: str = "reference_trajectory", std: float = 0.5
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  index = command.cfg.body_names.index("pelvis")
  # body_pos_w already includes env_origins. Reset pose noise is an initial
  # perturbation to recover from, not an episode-level placement transform.
  # aligned_body_position_w would hide accumulated world XY drift.
  return xy_position_score(
    command.robot_body_pos_w[:, index], command.body_pos_w[:, index], std
  )
