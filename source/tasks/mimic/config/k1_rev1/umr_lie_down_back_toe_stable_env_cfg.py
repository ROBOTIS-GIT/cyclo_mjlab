# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Penalize brief forefoot re-contact spikes in the soft-contact lie-down."""

from mjlab.envs import ManagerBasedRlEnvCfg

from .umr_lie_down_back_soft_contact_env_cfg import (
  k1_rev1_umr_lie_down_back_soft_contact_env_cfg,
)


def k1_rev1_umr_lie_down_back_toe_stable_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = k1_rev1_umr_lie_down_back_soft_contact_env_cfg(play=play)
  # A single 5 ms spike used to be averaged over four samples. Half mean,
  # half peak, with twice the weight, makes that isolated spike cost 5x as
  # much while retaining pressure against repeated load oscillations.
  # Both feet use the same objective; smooth rolling remains within the
  # existing 50 N deadband. No extra sensors or smaller physics step needed.
  toe_reward = cfg.rewards['toe_load_change']
  toe_reward.params['peak_fraction'] = 0.5
  toe_reward.weight = -0.2
  return cfg
