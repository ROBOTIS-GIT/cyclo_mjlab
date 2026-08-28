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
# Additional notices:
# This module is adapted from HybridRobotics/whole_body_tracking, licensed under
# the MIT License. See THIRD_PARTY_LICENSES.md for details.

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import ReferenceTrajectoryCommand
from .rewards import _tracked_body_ids

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def reference_anchor_position_deviation(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(ReferenceTrajectoryCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def reference_anchor_height_deviation(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(ReferenceTrajectoryCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def reference_anchor_gravity_deviation(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(ReferenceTrajectoryCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def reference_body_position_deviation(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(ReferenceTrajectoryCommand, env.command_manager.get_term(command_name))

  body_indexes = _tracked_body_ids(command, body_names)
  error = torch.norm(
    command.aligned_body_position_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def reference_body_height_deviation(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(ReferenceTrajectoryCommand, env.command_manager.get_term(command_name))

  body_indexes = _tracked_body_ids(command, body_names)
  error = torch.abs(
    command.aligned_body_position_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)


def joint_velocity_limit_exceeded(
  env: ManagerBasedRlEnv,
  max_velocity: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Terminate environments whose joint velocity exceeds a manual limit."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.any(torch.abs(joint_vel) > max_velocity, dim=1)
