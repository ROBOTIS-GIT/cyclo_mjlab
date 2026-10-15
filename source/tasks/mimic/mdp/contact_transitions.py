# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Persistent, debounced foot contacts for the Koryo ContactV2 task."""
import math
import mujoco
import torch
from mjlab.utils.lab_api.math import quat_apply
from .performance_contacts import FOOT_BODIES, foot_load_history


class ContactTracker:
  """Consume chronological *new* substeps once; never reconsume history overlap.

  Reference flight clears recontact eligibility, but physical contact state stays
  hysteretic. A normal first touchdown is free. All state is reset per environment.
  """
  def __init__(self, num_envs, device, dt, window_s=0.15, debounce_s=0.01):
    self.dt = dt
    self.window_s = window_s
    self.debounce = max(1, math.ceil(debounce_s / dt - 1e-8))
    self.width = max(1, math.ceil(window_s / dt))
    shape = (num_envs, 2)
    self.contact = torch.zeros(shape, device=device, dtype=torch.bool)
    self.seen = torch.zeros_like(self.contact)
    self.pending_on = torch.zeros(shape, device=device, dtype=torch.long)
    self.pending_off = torch.zeros_like(self.pending_on)
    self.off_age = torch.zeros(shape, device=device)
    self.expected_age = torch.zeros_like(self.off_age)
    self.contact_history = torch.zeros((*shape, self.width), device=device, dtype=torch.bool)
    self.expected_history = torch.zeros_like(self.contact_history)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    for value in vars(self).values():
      if isinstance(value, torch.Tensor):
        value[ids] = 0

  def update(self, loads, expected):
    """loads: [env, foot, substep], expected: [env, foot] reference support."""
    self.seen &= expected
    self.expected_history &= expected[..., None]
    events = torch.zeros_like(self.off_age)
    for f in loads.unbind(-1):
      self.expected_age = torch.where(expected, self.expected_age + self.dt, 0.)
      self.pending_on = torch.where(~self.contact & (f >= 20), self.pending_on + 1, 0)
      self.pending_off = torch.where(self.contact & (f <= 5), self.pending_off + 1, 0)
      touchdown = ~self.contact & (self.pending_on >= self.debounce)
      liftoff = self.contact & (self.pending_off >= self.debounce)
      # Count a completed short unload, including when it spans policy intervals.
      events += (touchdown & self.seen & expected & (self.off_age <= self.window_s)).float()
      self.contact = (self.contact | touchdown) & ~liftoff
      self.off_age = torch.where(self.contact, 0., self.off_age + self.dt)
      self.seen |= self.contact & expected
      self.contact_history = self.contact_history.roll(-1, dims=2)
      self.expected_history = self.expected_history.roll(-1, dims=2)
      self.contact_history[..., -1] = self.contact
      self.expected_history[..., -1] = expected
    missing = (self.expected_history & ~self.contact_history).float().sum(-1)
    duration = self.expected_history.float().sum(-1).clamp_min(1)
    # Allow a brief landing transition; sustained hovering still incurs a cost.
    deficit = missing / duration * expected * (self.expected_age >= 0.05)
    return events.sum(-1), deficit.sum(-1)


def reference_support_mask(bottom, vertical_speed):
  """Heuristic support labels, including pivots; not manually annotated truth.

  The lowest colliding sole sphere handles foot rotation better than ankle height.
  Hysteresis limits label flicker; clear upward motion permits intentional liftoff.
  """
  floor = bottom.amin(dim=0)
  height = bottom - floor
  state = torch.zeros(2, device=bottom.device, dtype=torch.bool)
  labels = []
  for z, vz in zip(height, vertical_speed):
    state = torch.where(state, z < 0.025, z < 0.012) & (vz < 0.2)
    labels.append(state)
  return torch.stack(labels)


def build_reference_support(env, cmd):
  """Compute fixed labels once from the actual MuJoCo foot collision geometry."""
  model = env.sim.mj_model
  ids = [cmd.cfg.body_names.index(n) for n in FOOT_BODIES]
  bottoms = []
  for name, ref_id in zip(FOOT_BODIES, ids):
    matches = [i for i in range(model.nbody)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or '').split('/')[-1] == name]
    if len(matches) != 1:
      raise ValueError(f'Expected one collision body for {name}, got {matches}')
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == matches[0]
             and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not geoms or any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE for g in geoms):
      raise ValueError('ContactV2 expects K1 sphere foot collision geometry.')
    p = cmd.reference.body_position_w[:, ref_id]
    q = cmd.reference.body_orientation_w[:, ref_id]
    offset = torch.as_tensor(model.geom_pos[geoms].copy(), device=p.device, dtype=p.dtype)
    radius = torch.as_tensor(model.geom_size[geoms, 0].copy(), device=p.device, dtype=p.dtype)
    rotated = quat_apply(q[:, None, :].expand(-1, len(geoms), -1), offset[None].expand(len(p), -1, -1))
    bottoms.append((p[:, None, 2] + rotated[..., 2] - radius).amin(-1))
  bottom = torch.stack(bottoms, dim=1)
  # Sole vertical motion includes rotation, unlike ankle-origin vertical velocity.
  velocity = torch.gradient(bottom, spacing=(1 / cmd.reference.fps,), dim=(0,))[0]
  return reference_support_mask(bottom.cpu(), velocity.cpu()).to(bottom.device)


class FootContactTransitions:
  """One manager term owns all contact state and its per-environment reset hook."""
  def __init__(self, cfg, env):
    self.env = env
    self.cmd = env.command_manager.get_term('reference_trajectory')
    self.tracker = ContactTracker(env.num_envs, env.device, env.cfg.sim.mujoco.timestep)
    self.reference_support = build_reference_support(env, self.cmd)
    self.last_frame = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    self.age = torch.zeros(env.num_envs, device=env.device)
    self.last_step = -1
    self.cached = torch.zeros_like(self.age)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.tracker.reset(ids)
    self.last_frame[ids] = -1
    self.age[ids] = 0
    self.cached[ids] = 0
    self.last_step = -1

  def __call__(self, env, recontact_weight=0.2, support_weight=0.15, ramp_steps=24000):
    if self.last_step == env.common_step_counter:
      return self.cached
    frame = self.cmd.frame_ids
    # Reference wrapping/resampling teleports also invalidate contact memory.
    discontinuity = (frame != self.last_frame + 1) | (env.episode_length_buf <= 1)
    ids = discontinuity.nonzero(as_tuple=False).flatten()
    self.reset(ids)
    self.last_frame.copy_(frame)
    history = foot_load_history(env)
    loads = history[:, :, :env.cfg.decimation].flip(-1)
    expected = self.reference_support[frame]
    events, deficit = self.tracker.update(loads, expected)
    self.age += env.step_dt
    valid = self.age > self.tracker.window_s
    events, deficit = events * valid, deficit * valid
    ramp = min(1., 0.2 + 0.8 * env.common_step_counter / max(1, ramp_steps))
    self.cmd.metrics['contact_v2_recontacts'] = events
    self.cmd.metrics['contact_v2_support_deficit'] = deficit
    self.cmd.metrics['contact_v2_expected_support_feet'] = expected.float().sum(-1)
    self.cmd.metrics['contact_v2_ramp'] = torch.full_like(events, ramp)
    self.cached = ramp * (recontact_weight * events + support_weight * deficit)
    self.last_step = env.common_step_counter
    return self.cached
