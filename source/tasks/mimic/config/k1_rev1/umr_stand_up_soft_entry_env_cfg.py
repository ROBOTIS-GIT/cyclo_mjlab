# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import fields
from mjlab.managers import RewardTermCfg
from source import SRC_PATH
from source.tasks.mimic.mdp.soft_entry import SoftEntryActionCfg, entry_dynamics
from .umr_stand_up_ai_sapiens_env_cfg import (
  k1_rev1_umr_stand_up_ai_sapiens_env_cfg, k1_rev1_umr_stand_up_ai_sapiens_ppo_cfg,
)

def soft_entry_env_cfg(play=False):
  cfg = k1_rev1_umr_stand_up_ai_sapiens_env_cfg(play=play)
  cfg.commands['reference_trajectory'].trajectory_file = str(SRC_PATH /
    'assets/motions/K1_rev1/umr_stand_up/k1_0008_stand_up_ai_sapiens_soft_entry.npz')
  old = cfg.actions['joint_pos']
  cfg.actions['joint_pos'] = SoftEntryActionCfg(**{f.name: getattr(old, f.name) for f in fields(old)})
  for name, weight in {'vertical_speed': -.3, 'angular_speed': -.2,
    'vertical_acceleration': -.05, 'angular_acceleration': -.05, 'height_overshoot': -.3}.items():
    cfg.rewards['soft_entry_' + name] = RewardTermCfg(func=entry_dynamics, weight=weight, params={'quantity': name})
  return cfg

def soft_entry_ppo_cfg():
  cfg = k1_rev1_umr_stand_up_ai_sapiens_ppo_cfg()
  cfg.algorithm.learning_rate = 5.e-5
  cfg.run_name = 'umr_stand_up_ai_sapiens_soft_entry'
  return cfg
