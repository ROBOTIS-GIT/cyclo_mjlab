# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Fine-tune for the existing ai_sapiens controller without deployment changes."""
from dataclasses import fields, replace
import mujoco
from mjlab.managers import RewardTermCfg
from source.tasks.mimic.mdp.ai_sapiens_entry import (
  ASSET, CONTRACT, AiSapiensCommandCfg, AiSapiensActionCfg, entry_push, entry_foot_clearance,
)
from .umr_stand_up_ready_handoff_env_cfg import k1_rev1_umr_stand_up_ready_handoff_env_cfg
from .agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg


def deployment_robot_spec():
  spec = mujoco.MjSpec.from_file(str(ASSET / 'mujoco/k1/k1.xml'))
  for actuator in tuple(spec.actuators):
    spec.delete(actuator)
  for side in ('left', 'right'):
    spec.body(f'{side}_ankle_roll_link').add_site(name=f'{side}_foot', pos=(.0325, 0., -.0635), size=(.01, .01, .01), group=5)
  for name, sensor_type in (('imu_ang_vel', mujoco.mjtSensor.mjSENS_GYRO),
      ('imu_lin_vel', mujoco.mjtSensor.mjSENS_VELOCIMETER), ('imu_lin_acc', mujoco.mjtSensor.mjSENS_ACCELEROMETER)):
    spec.add_sensor(name=name, type=sensor_type, objtype=mujoco.mjtObj.mjOBJ_SITE, objname='imu')
  spec.add_sensor(name='root_angmom', type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY, objname='pelvis')
  return spec


def k1_rev1_umr_stand_up_ai_sapiens_env_cfg(play=False):
  cfg = k1_rev1_umr_stand_up_ready_handoff_env_cfg(play=play)
  old = cfg.commands['reference_trajectory']
  accepted = {f.name for f in fields(AiSapiensCommandCfg)}
  command = AiSapiensCommandCfg(**{f.name: getattr(old, f.name) for f in fields(old) if f.name in accepted})
  command.first_frame_env_fraction = 1. if play else .75
  command.deterministic_bank = play
  command.resample_material_on_reset = False
  cfg.commands['reference_trajectory'] = command
  action = cfg.actions['joint_pos']
  cfg.actions['joint_pos'] = AiSapiensActionCfg(**{f.name: getattr(action, f.name) for f in fields(action)})
  cfg.actions['joint_pos'].ready_kp = dict(zip(CONTRACT['joints'], CONTRACT['ready_kp']))
  cfg.actions['joint_pos'].ready_kd = dict(zip(CONTRACT['joints'], CONTRACT['ready_kd']))
  cfg.terminations.pop('ready_timeout')
  robot = cfg.scene.entities['robot']
  robot.spec_fn = deployment_robot_spec
  robot.collisions = tuple(replace(c, condim=3, priority=0, friction=(1., .005, .0001)) for c in robot.collisions)
  cfg.sim.mujoco.timestep = .002
  cfg.sim.mujoco.iterations = 100
  cfg.sim.mujoco.ls_iterations = 50
  cfg.decimation = 10
  # Match the tested deployment simulator first. Reset-bank variations retain
  # entry-pose diversity; later hardware DR should be evaluated separately.
  for key in ('joint_home_offset_noise', 'torso_com_offset_noise'):
    cfg.events.pop(key, None)
  material = cfg.events['physics_material']
  material.mode = 'startup'
  material.params.update(low_range=(1., 1.), regular_range=(1., 1.),
    low_probability_start=0., low_probability_end=0., curriculum_command=None)
  if 'push_robot' in cfg.events:
    cfg.events['push_robot'].func = entry_push
  cfg.scene.sensors = tuple(replace(s, history_length=cfg.decimation) if getattr(s, 'history_length', 0) else s for s in cfg.scene.sensors)
  cfg.rewards['entry_foot_clearance'] = RewardTermCfg(func=entry_foot_clearance, weight=-.2)
  return cfg


def k1_rev1_umr_stand_up_ai_sapiens_ppo_cfg():
  cfg = k1_rev1_mimic_ppo_runner_cfg()
  cfg.algorithm.learning_rate = 1.e-4
  cfg.algorithm.schedule = 'fixed'
  cfg.max_iterations = 10000
  cfg.save_interval = 250
  cfg.run_name = 'umr_stand_up_ai_sapiens'
  return cfg
