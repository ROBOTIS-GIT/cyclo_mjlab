# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Taeguek02 with the shared full-motion-gated taekwondo curriculum."""

from .koryo_curriculum_env_cfg import apply_taekwondo_curriculum
from .taeguek02_env_cfg import k1_rev1_taeguek02_env_cfg


def k1_rev1_taeguek02_curriculum_env_cfg(play=False):
  cfg = k1_rev1_taeguek02_env_cfg(play=play)
  return cfg if play else apply_taekwondo_curriculum(cfg, "taeguek02_difficulty")
