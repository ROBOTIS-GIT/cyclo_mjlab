# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Support continuity and reference-aware sole motion under full disturbances."""
import math

import mujoco
import torch
from mjlab.utils.lab_api.math import quat_apply

from .contact_transitions import build_reference_support
from .performance_contacts import FOOT_BODIES, foot_load_history


class SupportTracker:
  """Debounced contacts with no expiry for unexpected liftoffs/long unloads.

  Costs are soft and bounded: recovery steps are possible, but are not free.
  Reference flight clears support duration; ordinary first touchdowns are free.
  """
  def __init__(self, num_envs, device, dt, debounce_s=0.01):
    self.dt = dt
    self.debounce = max(1, math.ceil(debounce_s / dt - 1e-8))
    shape = (num_envs, 2)
    self.contact = torch.zeros(shape, dtype=torch.bool, device=device)
    self.pending_on = torch.zeros(shape, dtype=torch.long, device=device)
    self.pending_off = torch.zeros_like(self.pending_on)
    self.expected_age = torch.zeros(shape, device=device)
    self.unload_age = torch.zeros_like(self.expected_age)

  def reset(self, ids=None):
    for value in vars(self).values():
      if isinstance(value, torch.Tensor):
        value[slice(None) if ids is None else ids] = 0

  def update(self, loads, expected):
    """Consume only new substeps, oldest first; return per-foot measurements."""
    liftoffs = torch.zeros_like(self.expected_age)
    missing = torch.zeros_like(liftoffs)
    long_unload = torch.zeros_like(liftoffs)
    occupancy = torch.zeros_like(liftoffs)
    for load in loads.unbind(-1):
      self.expected_age = torch.where(expected, self.expected_age + self.dt, 0.)
      self.pending_on = torch.where(~self.contact & (load >= 20), self.pending_on + 1, 0)
      self.pending_off = torch.where(self.contact & (load <= 5), self.pending_off + 1, 0)
      on = ~self.contact & (self.pending_on >= self.debounce)
      off = self.contact & (self.pending_off >= self.debounce)
      eligible = expected & (self.expected_age >= 0.05)
      liftoffs += (off & eligible).float()
      self.contact = (self.contact | on) & ~off
      self.unload_age = torch.where(expected & ~self.contact, self.unload_age + self.dt, 0.)
      absent = eligible & ~self.contact
      missing += absent.float()
      long_unload += ((self.unload_age - 0.05).clamp_min(0) / 0.15).square().clamp_max(4) * absent
      occupancy += self.contact.float()
    n = loads.shape[-1]
    return liftoffs, missing / n, long_unload / n, occupancy / n


def sole_horizontal_speed(quat, velocity_com, angular_velocity, offsets, radii, com):
  """Mean speed of low sphere-bottom points, not the moving ankle/COM origin.

  velocity_com and angular_velocity are world-frame velocities. Rotation about a
  stationary contact has zero point speed even when the ankle origin is moving.
  Low spheres approximate the support patch; reference speed permits pivots.
  """
  n, feet, _ = quat.shape
  count = offsets.shape[1]
  q = quat[:, :, None, :].expand(n, feet, count, 4)
  rotated = quat_apply(q, offsets[None].expand(n, -1, -1, -1))
  bottom = rotated.clone()
  bottom[..., 2] -= radii[None]
  com_w = quat_apply(quat, com[None].expand(n, -1, -1))
  lever = bottom - com_w[:, :, None, :]
  velocity = velocity_com[:, :, None, :] + torch.cross(
    angular_velocity[:, :, None, :].expand_as(lever), lever, dim=-1)
  low = bottom[..., 2] <= bottom[..., 2].amin(-1, keepdim=True) + 0.005
  return (velocity[..., :2].norm(dim=-1) * low).sum(-1) / low.sum(-1).clamp_min(1)


def excess_motion_cost(speed, ref_speed, angular_velocity, ref_angular_velocity):
  """Allow reference sliding/pivots plus tolerance; bound recovery costs."""
  slip = ((speed - ref_speed - 0.05).clamp_min(0) / 0.2).square().clamp_max(4)
  tilt = (angular_velocity[..., :2].norm(dim=-1)
          - ref_angular_velocity[..., :2].norm(dim=-1) - 0.5).clamp_min(0)
  yaw = (angular_velocity[..., 2].abs() - ref_angular_velocity[..., 2].abs() - 1.).clamp_min(0)
  rotation = ((tilt / 2).square() + (yaw / 2).square()).clamp_max(4)
  return slip, rotation


class FootSupportQuality:
  def __init__(self, cfg, env):
    self.cmd = env.command_manager.get_term('reference_trajectory')
    self.feet = [self.cmd.cfg.body_names.index(n) for n in FOOT_BODIES]
    self.reference_support = build_reference_support(env, self.cmd)
    self.tracker = SupportTracker(env.num_envs, env.device, env.cfg.sim.mujoco.timestep)
    self.last_frame = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    self.age = torch.zeros(env.num_envs, device=env.device)
    self.cached = torch.zeros_like(self.age)
    self.last_step = -1
    offsets, radii, com = [], [], []
    model = env.sim.mj_model
    for name in FOOT_BODIES:
      bodies = [i for i in range(model.nbody)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or '').split('/')[-1] == name]
      if len(bodies) != 1:
        raise ValueError(f'Expected one foot body: {name}')
      body = bodies[0]
      geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
               and (model.geom_contype[g] or model.geom_conaffinity[g])]
      if len(geoms) != 9 or any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE for g in geoms):
        raise ValueError('Support quality expects nine K1 foot spheres per side.')
      offsets.append(torch.tensor(model.geom_pos[geoms].copy(), dtype=torch.float32, device=env.device))
      radii.append(torch.tensor(model.geom_size[geoms, 0].copy(), dtype=torch.float32, device=env.device))
      com.append(torch.tensor(model.body_ipos[body].copy(), dtype=torch.float32, device=env.device))
    self.offsets, self.radii, self.com = map(torch.stack, (offsets, radii, com))
    ref = self.cmd.reference
    self.ref_speed = sole_horizontal_speed(
      ref.body_orientation_w[:, self.feet], ref.body_linear_velocity_w[:, self.feet],
      ref.body_angular_velocity_w[:, self.feet], self.offsets, self.radii, self.com)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.tracker.reset(env_ids)
    self.last_frame[ids] = -1
    self.age[ids] = 0
    self.cached[ids] = 0
    self.last_step = -1

  def __call__(self, env, support_weight=0.45, long_unload_weight=0.15,
               liftoff_weight=0.02, slip_weight=0.15, rotation_weight=0.05,
               single_support_multiplier=1.5, ramp_steps=24000):
    if self.last_step == env.common_step_counter:
      return self.cached
    frame = self.cmd.frame_ids
    discontinuity = (frame != self.last_frame + 1) | (env.episode_length_buf <= 1)
    self.reset(discontinuity.nonzero(as_tuple=False).flatten())
    self.last_frame.copy_(frame)
    expected = self.reference_support[frame]
    loads = foot_load_history(env)[:, :, :env.cfg.decimation].flip(-1)
    liftoffs, missing, long_unload, occupancy = self.tracker.update(loads, expected)
    self.age += env.step_dt
    valid = (self.age > 0.15).float()
    multiplier = torch.where(expected.sum(-1) == 1, single_support_multiplier, 1.)
    speed = sole_horizontal_speed(
      self.cmd.robot_body_quat_w[:, self.feet], self.cmd.robot_body_lin_vel_w[:, self.feet],
      self.cmd.robot_body_ang_vel_w[:, self.feet], self.offsets, self.radii, self.com)
    slip, rotation = excess_motion_cost(
      speed, self.ref_speed[frame], self.cmd.robot_body_ang_vel_w[:, self.feet],
      self.cmd.body_ang_vel_w[:, self.feet])
    # Motion costs need physical support, while missing/long-unload costs do not.
    slip = slip * expected * occupancy
    rotation = rotation * expected * occupancy
    # RewardManager multiplies by step_dt: normalize event counts to make each
    # unexpected liftoff cost liftoff_weight, independent of policy frequency.
    cost = (support_weight * missing + long_unload_weight * long_unload
            + liftoff_weight * liftoffs / env.step_dt
            + slip_weight * slip + rotation_weight * rotation).sum(-1)
    ramp = min(1., 0.2 + 0.8 * env.common_step_counter / max(1, ramp_steps))
    for name, value in [('liftoffs', liftoffs), ('missing', missing),
                        ('long_unload', long_unload), ('sole_slip', slip), ('rotation', rotation)]:
      self.cmd.metrics['support_v3_' + name] = value.sum(-1) * valid
    self.cmd.metrics['support_v3_expected_feet'] = expected.float().sum(-1)
    self.cmd.metrics['support_v3_ramp'] = torch.full_like(valid, ramp)
    self.cached = ramp * cost * multiplier * valid
    self.last_step = env.common_step_counter
    return self.cached
