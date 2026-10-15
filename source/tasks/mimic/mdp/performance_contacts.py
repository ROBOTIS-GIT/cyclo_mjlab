# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Foot-floor measurements and soft-contact objectives for performance policies."""
import torch
from mjlab.sensor import ContactMatch, ContactSensorCfg

FOOT_BODIES = ('left_ankle_roll_link', 'right_ankle_roll_link')
SENSOR = 'performance_feet'


def add_foot_sensors(cfg):
  cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
    name=SENSOR,
    primary=ContactMatch(mode='body', pattern=FOOT_BODIES, entity='robot'),
    secondary=ContactMatch(mode='geom', pattern='terrain'),
    secondary_policy='error', fields=('force',), reduce='netforce',
    global_frame=True, num_slots=1, history_length=cfg.decimation + 1,
  ))
  return cfg


def foot_load_history(env):
  history = env.scene[SENSOR].data.force_history
  if history is None or history.shape[2] < 2:
    raise RuntimeError('Performance foot objectives require force history.')
  return history[..., 2].abs()


def contact_costs(load, force_limit=700.0, force_scale=350.0, jump_allowance=50.0):
  """Per-environment costs; steady support and one normal touchdown are allowed.

  Thresholds describe a simulation objective, not hardware safety limits.
  Histories are newest-first. Costs are invariant to their time direction.
  """
  peak = load.amax(dim=2)
  impact = ((peak-force_limit).clamp_min(0)/force_scale).square().clamp_max(25).sum(dim=1)
  jumps = (torch.diff(load,dim=2).abs()-jump_allowance).clamp_min(0)/force_scale
  changes = jumps.square().clamp_max(25)
  load_change = (0.5*changes.mean(dim=2)+0.5*changes.amax(dim=2)).sum(dim=1)
  a,b = load[:,:,:-1],load[:,:,1:]
  switches = (((a<=5)&(b>=20))|((a>=20)&(b<=5))).sum(dim=2)
  # Do not charge for a single liftoff/touchdown inside this policy interval.
  chatter = (switches-1).clamp_min(0).float().sum(dim=1)
  return impact,load_change,chatter


def foot_contact_cost(env, kind: str):
  load = foot_load_history(env)
  kinds = {'impact':0,'load_change':1,'chatter':2}
  if kind not in kinds:
    raise ValueError(f'Unknown foot contact cost: {kind}')
  cost = contact_costs(load)[kinds[kind]]
  command = env.command_manager.get_term('reference_trajectory')
  command.metrics[f'performance_foot_{kind}'] = cost
  command.metrics['performance_peak_foot_force_n'] = load.amax(dim=(1,2))
  # Reset establishes contacts instantaneously; do not train against that event.
  return cost * (env.episode_length_buf>1)


def quiet_stance_mask(ref_velocity, ref_angular_velocity, ref_height, min_height):
  """Exclude intended steps, kicks and pivots from the planted-foot objective."""
  return ((ref_velocity.norm(dim=-1)<0.1)
          & (ref_angular_velocity.norm(dim=-1)<1.0)
          & (ref_height<min_height+0.015))


def _quiet_reference_feet(cmd):
  if not hasattr(cmd,'_performance_feet'):
    ids = [cmd.cfg.body_names.index(n) for n in FOOT_BODIES]
    floor_height = cmd.reference.body_position_w[:,ids,2].amin(dim=0)
    cmd._performance_feet = (ids,floor_height)
  ids,floor_height = cmd._performance_feet
  ref_height = cmd.reference.body_position_w[cmd.frame_ids][:,ids,2]
  quiet = quiet_stance_mask(cmd.body_lin_vel_w[:,ids],cmd.body_ang_vel_w[:,ids],ref_height,floor_height)
  return ids,quiet


def stance_foot_motion(env):
  cmd = env.command_manager.get_term('reference_trajectory')
  ids,quiet = _quiet_reference_feet(cmd)
  loaded = foot_load_history(env)[:,:,0]>20
  speed = cmd.robot_body_lin_vel_w[:,ids,:2].norm(dim=-1)
  cost = (((speed-0.05).clamp_min(0)/0.2).square().clamp_max(25)*quiet*loaded).sum(dim=1)
  cmd.metrics['performance_stance_foot_motion'] = cost
  return cost*(env.episode_length_buf>1)


def stance_foot_support(env):
  """Prevent avoiding contact penalties by hovering a reference-planted foot."""
  cmd = env.command_manager.get_term('reference_trajectory')
  _,quiet = _quiet_reference_feet(cmd)
  peak = foot_load_history(env).amax(dim=2)
  cost = ((1-peak/20).clamp_min(0).square()*quiet).sum(dim=1)
  cmd.metrics['performance_stance_support_loss'] = cost
  return cost*(env.episode_length_buf>1)
