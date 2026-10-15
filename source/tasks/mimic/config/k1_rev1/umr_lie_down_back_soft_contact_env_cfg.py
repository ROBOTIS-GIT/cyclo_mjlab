# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Slower face-down landing with knee-impact and forefoot-stability objectives."""
from mjlab.envs.mdp.rewards import joint_acc_l2
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from source import SRC_PATH
from source.tasks.mimic.mdp.landing_contact_rewards import knee_impact, toe_load_change
from .umr_lie_down_back_env_cfg import k1_rev1_umr_lie_down_back_env_cfg


def k1_rev1_umr_lie_down_back_soft_contact_env_cfg(play=False):
  cfg = k1_rev1_umr_lie_down_back_env_cfg(play=play)
  cfg.commands['reference_trajectory'].trajectory_file = str(
    SRC_PATH / 'assets/motions/K1_rev1/umr_lie_down_back'
    / 'k1_lie_down_back_soft_contact_converted.npz'
  )
  floor = ContactMatch(mode='geom', pattern='terrain')
  sensors = [ContactSensorCfg(
    name='knee_floor_contact',
    primary=ContactMatch(mode='body', pattern=('left_knee_link','right_knee_link'), entity='robot'),
    secondary=floor, secondary_policy='error', fields=('force',),
    reduce='netforce', num_slots=1, history_length=cfg.decimation,
  )]
  for side in ('left','right'):
    sensors.append(ContactSensorCfg(
      name=f'{side}_toe_floor_contact',
      primary=ContactMatch(mode='geom', pattern=f'{side}_ankle_roll_link_collision_[4-8]', entity='robot'),
      secondary=floor, secondary_policy='error', fields=('force',),
      reduce='netforce', num_slots=1, history_length=cfg.decimation+1,
    ))
  # LieDown removes undesired_contacts: its full-body sensor is unused here.
  # Avoid computing those contact forces and rolling their history at 200 Hz.
  # Keep the dedicated knee/toe histories needed by the landing rewards.
  cfg.scene.sensors = (
    *(sensor for sensor in cfg.scene.sensors if sensor.name != 'contact_forces'),
    *sensors,
  )
  cfg.rewards['knee_impact'] = RewardTermCfg(
    func=knee_impact, weight=-0.5,
    params={'sensor_name':'knee_floor_contact', 'support_force_n':350.0},
  )
  cfg.rewards['toe_load_change'] = RewardTermCfg(
    func=toe_load_change, weight=-0.1,
    params={'sensor_names':('left_toe_floor_contact','right_toe_floor_contact'),
            'allowed_change_n':50.0, 'scale_n':175.0},
  )
  cfg.rewards['ankle_acc'] = RewardTermCfg(
    func=joint_acc_l2, weight=-2e-6,
    params={'asset_cfg':SceneEntityCfg('robot',joint_names=('.*_ankle_pitch_joint','.*_ankle_roll_joint'))},
  )
  return cfg
