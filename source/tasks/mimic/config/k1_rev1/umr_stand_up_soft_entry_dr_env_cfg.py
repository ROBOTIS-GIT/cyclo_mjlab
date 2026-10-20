# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Keep SoftEntry while restoring the original StandUp domain randomization."""
from copy import deepcopy
from .umr_stand_up_soft_entry_env_cfg import soft_entry_env_cfg, soft_entry_ppo_cfg
from .umr_stand_up_soft_landing_env_cfg import k1_rev1_umr_stand_up_soft_landing_env_cfg

RESTORED_EVENTS = ('physics_material', 'joint_home_offset_noise', 'torso_com_offset_noise')


def soft_entry_dr_env_cfg(play=False):
  cfg = soft_entry_env_cfg(play=play)
  if not play:
    original = k1_rev1_umr_stand_up_soft_landing_env_cfg(play=False)
    for name in RESTORED_EVENTS:
      cfg.events[name] = deepcopy(original.events[name])
    # Clip completion resamples commands without necessarily resetting the env.
    # Restore this flag as well as the event, otherwise subsequent clips retain
    # the startup friction instead of the current curriculum stage's sample.
    cfg.commands['reference_trajectory'].resample_material_on_reset = True
  return cfg


def soft_entry_dr_ppo_cfg():
  cfg = soft_entry_ppo_cfg()
  cfg.run_name = 'umr_stand_up_ai_sapiens_soft_entry_dr'
  return cfg
