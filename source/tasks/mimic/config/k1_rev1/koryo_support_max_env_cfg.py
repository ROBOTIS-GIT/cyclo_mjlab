# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Scratch Koryo training: full disturbances with persistent support objectives."""
from mjlab.managers.reward_manager import RewardTermCfg
from source.tasks.mimic.mdp.support_contacts import FootSupportQuality
from .koryo_contact_max_env_cfg import k1_rev1_koryo_contact_max_env_cfg, k1_rev1_koryo_contact_max_ppo_cfg


def k1_rev1_koryo_support_max_env_cfg(play=False):
  cfg = k1_rev1_koryo_contact_max_env_cfg(play=play)
  if not play:
    cfg.scene.num_envs = 4096
  del cfg.rewards['performance_contact_transitions']
  del cfg.rewards['performance_stance_foot_motion']
  cfg.rewards['support_quality'] = RewardTermCfg(
    func=FootSupportQuality, weight=-1., params={
      'support_weight': 0.45, 'long_unload_weight': 0.15, 'liftoff_weight': 0.02,
      'slip_weight': 0.15, 'rotation_weight': 0.05,
      'single_support_multiplier': 1.5, 'ramp_steps': 24000,
    })
  return cfg


def k1_rev1_koryo_support_max_ppo_cfg():
  cfg = k1_rev1_koryo_contact_max_ppo_cfg()
  cfg.run_name = 'koryo_support_v3_max_from_scratch'
  cfg.resume = False
  return cfg
