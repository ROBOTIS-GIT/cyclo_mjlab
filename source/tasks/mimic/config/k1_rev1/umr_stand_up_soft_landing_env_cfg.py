# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Retimed UMR stand-up with a pelvis-floor impact penalty."""
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from source import SRC_PATH
from source.tasks.mimic.mdp.impact_rewards import pelvis_impact
from .umr_stand_up_scratch_env_cfg import k1_rev1_umr_stand_up_scratch_env_cfg


def k1_rev1_umr_stand_up_soft_landing_env_cfg(play=False):
  cfg = k1_rev1_umr_stand_up_scratch_env_cfg(play=play)
  cfg.commands['reference_trajectory'].trajectory_file = str(
    SRC_PATH / 'assets/motions/K1_rev1/umr_stand_up'
    / 'k1_0008_stand_up_soft_landing_converted.npz'
  )
  if not play:
    # Same physical preparation poses as old [5, 7] seconds after retiming.
    cfg.commands['reference_trajectory'].preparation_start_seconds = (5.4875, 8.2)
  cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
    name='pelvis_floor_contact',
    primary=ContactMatch(mode='body', pattern='pelvis', entity='robot'),
    secondary=ContactMatch(mode='geom', pattern='terrain'),
    secondary_policy='error', fields=('force',), reduce='netforce',
    num_slots=1, history_length=cfg.decimation,
  ))
  # Nominal robot weight is 350 N; allow 1.2 body weights of support.
  cfg.rewards['pelvis_impact'] = RewardTermCfg(
    func=pelvis_impact, weight=-0.2,
    params={'sensor_name': 'pelvis_floor_contact', 'support_force_n': 420.0},
  )
  return cfg
