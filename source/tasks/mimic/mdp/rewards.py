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

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import ReferenceTrajectoryCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _tracked_body_ids(
  command: ReferenceTrajectoryCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def reference_anchor_position_tracking(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(ReferenceTrajectoryCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def reference_anchor_orientation_tracking(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(ReferenceTrajectoryCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def reference_body_position_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(ReferenceTrajectoryCommand, env.command_manager.get_term(command_name))
  body_indexes = _tracked_body_ids(command, body_names)
  error = torch.sum(
    torch.square(
      command.aligned_body_position_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def reference_body_orientation_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(ReferenceTrajectoryCommand, env.command_manager.get_term(command_name))
  body_indexes = _tracked_body_ids(command, body_names)
  error = (
    quat_error_magnitude(
      command.aligned_body_orientation_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def reference_body_linear_velocity_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(ReferenceTrajectoryCommand, env.command_manager.get_term(command_name))
  body_indexes = _tracked_body_ids(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def reference_body_angular_velocity_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(ReferenceTrajectoryCommand, env.command_manager.get_term(command_name))
  body_indexes = _tracked_body_ids(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 1.0,
) -> torch.Tensor:
  """Count bodies whose contact force exceeded the threshold recently."""
  sensor: ContactSensor = env.scene[sensor_name]
  force_history = sensor.data.force_history
  if force_history is None:
    force = sensor.data.force
    assert force is not None
    return (torch.linalg.norm(force, dim=-1) > threshold).sum(dim=1).float()

  # MJLab layout: [environment, primary body, history, xyz].
  max_force = torch.linalg.norm(force_history, dim=-1).amax(dim=2)
  return (max_force > threshold).sum(dim=1).float()
