# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Contact-transition refinement, preserving the Performance V1 baseline."""
from mjlab.managers.reward_manager import RewardTermCfg
from source.tasks.mimic.mdp.contact_transitions import FootContactTransitions
from .koryo_performance_env_cfg import k1_rev1_koryo_performance_env_cfg, k1_rev1_koryo_performance_ppo_cfg


def k1_rev1_koryo_contact_v2_env_cfg(play=False):
  cfg = k1_rev1_koryo_performance_env_cfg(play=play)
  del cfg.rewards['performance_foot_chatter']
  del cfg.rewards['performance_stance_foot_support']
  cfg.rewards['performance_contact_transitions'] = RewardTermCfg(
    func=FootContactTransitions, weight=-1.,
    params={'recontact_weight':0.2, 'support_weight':0.15, 'ramp_steps':24000})
  return cfg


def k1_rev1_koryo_contact_v2_ppo_cfg():
  cfg = k1_rev1_koryo_performance_ppo_cfg()
  cfg.algorithm.learning_rate = 1e-5
  cfg.run_name = 'koryo_contact_v2'
  cfg.save_interval = 500
  return cfg


def k1_rev1_koryo_contact_curriculum_env_cfg(play=False):
  cfg = k1_rev1_koryo_contact_v2_env_cfg(play=play)
  if not play:
    cfg.curriculum['koryo_difficulty'] = cfg.curriculum.pop('performance_fixed_difficulty')
  return cfg
