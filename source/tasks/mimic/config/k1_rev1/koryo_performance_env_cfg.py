# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""A separate contact-quality fine-tuning task; preserves original Koryo timing."""
from mjlab.managers.reward_manager import RewardTermCfg
from source.tasks.mimic.mdp.performance_contacts import add_foot_sensors, foot_contact_cost, stance_foot_motion, stance_foot_support
from .koryo_env_cfg import k1_rev1_koryo_env_cfg
from .koryo_curriculum_env_cfg import apply_taekwondo_curriculum
from .agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg


def k1_rev1_koryo_performance_env_cfg(play=False):
  cfg = k1_rev1_koryo_env_cfg(play=play)
  if not play:
    # Fixed stage 0: 25% noise, no pushes. This polishes an already robust policy.
    # The performance runner never imports the mastered curriculum's stage.
    cfg = apply_taekwondo_curriculum(cfg, 'performance_fixed_difficulty')
  add_foot_sensors(cfg)
  for kind,weight in [('impact',-0.2),('load_change',-0.15),('chatter',-0.1)]:
    cfg.rewards['performance_foot_'+kind] = RewardTermCfg(
      func=foot_contact_cost, weight=weight, params={'kind':kind})
  cfg.rewards['performance_stance_foot_motion'] = RewardTermCfg(func=stance_foot_motion,weight=-0.3)
  cfg.rewards['performance_stance_foot_support'] = RewardTermCfg(func=stance_foot_support,weight=-0.15)
  cfg.rewards['action_rate_l2'].weight = -0.2
  return cfg


def k1_rev1_koryo_performance_ppo_cfg():
  cfg = k1_rev1_mimic_ppo_runner_cfg()
  cfg.algorithm.learning_rate = 5e-5
  cfg.algorithm.schedule = 'fixed'
  cfg.algorithm.entropy_coef = 0.0001
  cfg.max_iterations = 5000
  cfg.run_name = 'koryo_performance'
  return cfg
