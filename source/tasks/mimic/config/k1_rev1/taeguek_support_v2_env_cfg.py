# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Isolated SupportV2 fine-tuning tasks for Taegeuk 4 and 5."""
from source.tasks.mimic.mdp.taeguek_support_v2 import TaegeukFootSupportV2
from .taeguek_support_env_cfg import k1_rev1_taeguek_support_env_cfg
from .agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg


def k1_rev1_taeguek_support_v2_env_cfg(chapter, play=False):
  if chapter not in (4, 5):
    raise ValueError('SupportV2 is evaluated only for Taegeuk 4 and 5.')
  cfg = k1_rev1_taeguek_support_env_cfg(chapter, play=play)
  term = cfg.rewards['taeguek_support_quality']
  term.func = TaegeukFootSupportV2
  term.params = dict(support_weight=0.9, long_unload_weight=0.3,
    liftoff_weight=0.04, slip_weight=0.15, rotation_weight=0.05,
    single_support_multiplier=1.5, ramp_steps=1,
    clearance_weight=1.0, kick_height_weight=0.75, refinement_ramp_steps=12000)
  return cfg


def k1_rev1_taeguek_support_v2_ppo_cfg():
  cfg = k1_rev1_mimic_ppo_runner_cfg()
  cfg.run_name = 'taeguek_support_v2_finetune'
  cfg.algorithm.learning_rate = 1e-4
  cfg.algorithm.schedule = 'fixed'
  cfg.algorithm.entropy_coef = 0.001
  cfg.max_iterations = 10000
  cfg.save_interval = 500
  cfg.resume = False  # The driver supplies an explicit checkpoint per chapter.
  return cfg
