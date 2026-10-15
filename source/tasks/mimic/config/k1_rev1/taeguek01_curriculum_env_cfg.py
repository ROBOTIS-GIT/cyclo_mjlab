# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Taeguek01 with the same full-motion-gated curriculum as Koryo."""

from .koryo_curriculum_env_cfg import apply_taekwondo_curriculum
from .taeguek01_env_cfg import k1_rev1_taeguek01_env_cfg


def k1_rev1_taeguek01_curriculum_env_cfg(play=False):
  cfg = k1_rev1_taeguek01_env_cfg(play=play)
  return cfg if play else apply_taekwondo_curriculum(cfg, "taeguek01_difficulty")
