# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Insu Park

"""Cyclo K1-specific non-contact locomotion penalties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_height_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize root-link height error on flat terrain."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_pos_w[:, 2] - target_height)


def lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize vertical root COM velocity in the body frame."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_com_lin_vel_b[:, 2])


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize root roll and pitch angular velocity in the body frame."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_com_ang_vel_b[:, :2]), dim=1)


def joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize absolute joint-position deviation from the default pose."""
  asset: Entity = env.scene[asset_cfg.name]
  deviation = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  return torch.sum(torch.abs(deviation), dim=1)


def feet_lateral_separation_l2(
  env: ManagerBasedRlEnv,
  min_width: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot pairs whose lateral spacing falls below a minimum width."""
  asset: Entity = env.scene[asset_cfg.name]
  if len(asset_cfg.site_ids) != 2:
    raise ValueError(
      "feet_lateral_separation_l2 requires two sites ordered as (left, right)."
    )

  feet_position_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  separation_w = feet_position_w[:, 0, :] - feet_position_w[:, 1, :]
  separation_b = quat_apply_inverse(asset.data.root_link_quat_w, separation_w)
  lateral_spacing = separation_b[:, 1]

  width_deficit = torch.clamp(min_width - lateral_spacing, min=0.0)
  penalty = torch.square(width_deficit / min_width)

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      commanded_motion = torch.linalg.vector_norm(command[:, :2], dim=1)
      commanded_motion += torch.abs(command[:, 2])
      penalty *= (commanded_motion > command_threshold).float()

  env.extras["log"]["Metrics/foot_separation_mean"] = torch.mean(lateral_spacing)
  return penalty


def gait_contact_swing_tracking(
  env: ManagerBasedRlEnv,
  cycle_time_s: float,
  phase_offsets: tuple[float, ...],
  stance_fraction: float,
  motion_threshold: float,
  command_name: str,
  contact_sensor_name: str,
) -> torch.Tensor:
  """Reward agreement between a biped walking clock and measured contacts."""
  if cycle_time_s <= 0.0:
    raise ValueError("cycle_time_s must be positive.")
  if not 0.0 < stance_fraction < 1.0:
    raise ValueError("stance_fraction must be between zero and one.")

  contact_sensor: ContactSensor = env.scene[contact_sensor_name]
  contact_time = contact_sensor.data.current_contact_time
  assert contact_time is not None
  if contact_time.shape[1] != len(phase_offsets):
    raise ValueError(
      "phase_offsets must contain one value per tracked contact site."
    )

  base_cycle = (env.episode_length_buf * env.step_dt / cycle_time_s).unsqueeze(-1)
  offsets = torch.tensor(
    phase_offsets,
    device=env.device,
    dtype=base_cycle.dtype,
  ).unsqueeze(0)
  scheduled_contact = torch.remainder(base_cycle + offsets, 1.0) < stance_fraction
  measured_contact = contact_time > 0.0
  contact_agreement = torch.eq(scheduled_contact, measured_contact).float().mean(-1)

  command = env.command_manager.get_command(command_name)
  planar_speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
  motion_level = planar_speed + torch.abs(command[:, 2])
  return contact_agreement * (motion_level > motion_threshold)
