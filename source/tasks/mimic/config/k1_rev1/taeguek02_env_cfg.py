# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""K1 Rev.1 Taeguek 02 Mimic environment configuration."""

from .taeguek_env_cfg import k1_rev1_taeguek_env_cfg


def k1_rev1_taeguek02_env_cfg(play=False):
  return k1_rev1_taeguek_env_cfg(2, play=play)
