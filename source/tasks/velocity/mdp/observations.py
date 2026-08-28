"""Cyclo K1-specific locomotion observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def gait_phase(
  env: ManagerBasedRlEnv,
  cycle_time_s: float,
  command_name: str,
  motion_threshold: float = 0.1,
) -> torch.Tensor:
  """Encode the gait phase as sine/cosine features.

  The clock is zeroed while the planar/yaw command magnitude is below the
  motion threshold so the policy does not receive a stepping cue at rest.
  """
  if cycle_time_s <= 0.0:
    raise ValueError("cycle_time_s must be positive.")

  command = env.command_manager.get_command(command_name)
  cycle_angle = (
    torch.remainder(env.episode_length_buf * env.step_dt, cycle_time_s)
    * (2.0 * torch.pi / cycle_time_s)
  )
  clock = torch.stack((torch.sin(cycle_angle), torch.cos(cycle_angle)), dim=-1)

  planar_speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
  motion_level = planar_speed + torch.abs(command[:, 2])
  is_moving = motion_level > motion_threshold
  return clock * is_moving.unsqueeze(-1)
