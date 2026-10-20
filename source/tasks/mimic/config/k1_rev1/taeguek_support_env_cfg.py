# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Taegeuk 2–8 support refinement; preserve reference, observations and events."""
from mjlab.managers.reward_manager import RewardTermCfg
from source.tasks.mimic.mdp.performance_contacts import add_foot_sensors
from source.tasks.mimic.mdp.taeguek_support import TaegeukFootSupportQuality
from .taeguek_env_cfg import k1_rev1_taeguek_env_cfg
from .agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg


def k1_rev1_taeguek_support_env_cfg(chapter, play=False):
  if chapter not in range(2,9):raise ValueError("SupportV1 is configured for Taegeuk 2–8.")
  cfg=add_foot_sensors(k1_rev1_taeguek_env_cfg(chapter, play=play))
  cfg.rewards['taeguek_support_quality']=RewardTermCfg(
    func=TaegeukFootSupportQuality,weight=-1.,params={
      'support_weight':0.45,'long_unload_weight':0.15,'liftoff_weight':0.02,
      'slip_weight':0.15,'rotation_weight':0.05,
      'single_support_multiplier':1.5,'ramp_steps':24000})
  return cfg


def k1_rev1_taeguek_support_ppo_cfg():
  cfg=k1_rev1_mimic_ppo_runner_cfg()
  cfg.run_name='taeguek_support_v1_scratch'
  cfg.resume=False
  return cfg
