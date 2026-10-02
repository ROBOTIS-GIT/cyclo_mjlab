# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Matched robot/terrain friction for low-friction motion tracking."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("geom_friction")
def randomize_support_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  terrain_cfg: SceneEntityCfg,
  low_range: tuple[float, float] = (0.2, 0.6),
  regular_range: tuple[float, float] = (0.3, 1.2),
  low_probability_start: float = 0.3,
  low_probability_end: float = 0.7,
  ramp_steps: int = 48_000,
  curriculum_command: str | None = None,
) -> None:
  """Sample one sliding coefficient per reset world for robot AND terrain.

  Matching both sides avoids MuJoCo's max-friction mixing rule masking low
  friction on hands (which do not have the feet's elevated geom priority).
  Other friction axes and all contact solver parameters remain unchanged.
  The ramp counts policy steps in this process, including when resuming a
  checkpoint; it starts again on each new training invocation.
  """
  if curriculum_command is not None:
    stage = env.command_manager.get_term(curriculum_command).difficulty_stage
    low_range = ((0.8, 1.2), (0.4, 0.6), (0.3, 0.6), (0.2, 0.6))[stage]
    regular_range = (0.8, 1.2)
    low_probability_start = low_probability_end = (0.0, 0.3, 0.5, 0.7)[stage]
  for bounds in (low_range, regular_range):
    if not all(math.isfinite(x) for x in bounds) or not 0 < bounds[0] <= bounds[1]:
      raise ValueError("Friction ranges must be finite, positive, and ordered.")
  if not (0 <= low_probability_start <= 1 and 0 <= low_probability_end <= 1):
    raise ValueError("Low-friction probabilities must be in [0, 1].")
  if ramp_steps < 0:
    raise ValueError("ramp_steps must be nonnegative.")
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  else:
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
  progress = min(env.common_step_counter / max(ramp_steps, 1), 1.0)
  if ramp_steps == 0:
    progress = 1.0
  probability = low_probability_start + progress * (
    low_probability_end - low_probability_start
  )
  low = torch.rand(len(env_ids), device=env.device) < probability
  unit = torch.rand(len(env_ids), device=env.device)
  coefficient = torch.where(
    low,
    low_range[0] + unit * (low_range[1] - low_range[0]),
    regular_range[0] + unit * (regular_range[1] - regular_range[0]),
  )
  for cfg in (asset_cfg, terrain_cfg):
    geom_ids = env.scene[cfg.name].indexing.geom_ids[cfg.geom_ids]
    env.sim.model.geom_friction[env_ids[:, None], geom_ids[None, :], 0] = (
      coefficient[:, None]
    )
