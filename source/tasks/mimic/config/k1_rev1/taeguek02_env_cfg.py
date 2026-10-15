# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""K1 Rev.1 Taeguek 02 Mimic environment configuration."""

from source.tasks.mimic.motion_curriculum import MOTIONS
from .base_env_cfg import k1_rev1_mimic_env_cfg


def k1_rev1_taeguek02_env_cfg(play=False):
  cfg = k1_rev1_mimic_env_cfg(play=play)
  cfg.commands["reference_trajectory"].trajectory_file = str(MOTIONS["taeguek02"].trajectory)
  return cfg
