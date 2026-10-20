# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""50 Hz collision-corrected references for Taegeuk chapters 1–8."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from source.tasks.mimic.mdp import ReferenceTrajectoryCommandCfg
from source.tasks.mimic.mdp.pelvis_xy_rewards import reference_pelvis_xy
from source.tasks.mimic.motion_curriculum import TAEGUEK_TRAJECTORIES

from .base_env_cfg import k1_rev1_mimic_env_cfg


def k1_rev1_taeguek_env_cfg(
  chapter: int, play: bool = False
) -> ManagerBasedRlEnvCfg:
  if chapter not in TAEGUEK_TRAJECTORIES:
    raise ValueError(f"Taegeuk chapter must be 1–8, got {chapter}")
  cfg = k1_rev1_mimic_env_cfg(play=play)
  reference = cfg.commands["reference_trajectory"]
  assert isinstance(reference, ReferenceTrajectoryCommandCfg)
  reference.trajectory_file = str(TAEGUEK_TRAJECTORIES[chapter])
  cfg.rewards["reference_pelvis_xy"] = RewardTermCfg(
    func=reference_pelvis_xy,
    weight=1.0,
    params={"command_name": "reference_trajectory", "std": 0.5},
  )
  # The reference is resampled at 50 Hz without time scaling. Retain the base
  # 200 Hz physics / 50 Hz policy so one reference frame matches a policy step.
  return cfg
