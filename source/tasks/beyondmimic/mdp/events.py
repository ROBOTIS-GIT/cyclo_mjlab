# Copyright 2026 ROBOTIS CO., LTD.
# Copyright 2025, The mjlab Developers
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
#
# This file includes modifications by ROBOTIS CO., LTD. to code derived
# from mujocolab/mjlab.
#
# Additional notices:
# This module is adapted from HybridRobotics/whole_body_tracking, licensed under
# the MIT License. See THIRD_PARTY_LICENSES.md for details.

"""Event terms for Mimic tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from mjlab.envs.mdp.dr import body_com_offset, encoder_bias
from mjlab.envs.mdp.dr import geom_friction as randomize_rigid_body_material
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def apply_home_joint_offset_noise(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  pos_distribution_params: tuple[float, float] | None = None,
  operation: Literal["add", "scale", "abs"] = "abs",
  distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
  if pos_distribution_params is None:
    return
  if operation != "add" or distribution != "uniform":
    raise ValueError(
      "MJLab home joint offset noise supports operation='add' and "
      "distribution='uniform'."
    )
  encoder_bias(
    env,
    env_ids,
    bias_range=pos_distribution_params,
    asset_cfg=asset_cfg,
  )


def apply_link_com_offset_noise(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  com_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg,
) -> None:
  ranges = {
    index: com_range.get(key, (0.0, 0.0))
    for index, key in enumerate(("x", "y", "z"))
  }
  body_com_offset(
    env,
    env_ids,
    ranges=ranges,
    asset_cfg=asset_cfg,
    operation="add",
  )


apply_link_com_offset_noise.model_fields = body_com_offset.model_fields
apply_link_com_offset_noise.recompute = body_com_offset.recompute

__all__ = [
  "apply_home_joint_offset_noise",
  "apply_link_com_offset_noise",
  "randomize_rigid_body_material",
]
