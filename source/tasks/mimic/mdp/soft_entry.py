# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Entry-only body dynamics penalties with 2 ms sampling and reset-safe history."""
from dataclasses import dataclass
import torch
from .ai_sapiens_entry import AiSapiensAction, AiSapiensActionCfg, CONTRACT
from .ready_handoff import ReadyHandoffAction

# Excess thresholds, normalization scales. Velocity uses COM, matching NPZ.
LIMITS = {'vertical_speed': (.10, .15), 'angular_speed': (.60, .80),
  'vertical_acceleration': (3., 5.), 'angular_acceleration': (15., 25.),
  'height_overshoot': (.015, .030)}

@dataclass(kw_only=True)
class SoftEntryActionCfg(AiSapiensActionCfg):
  def build(self, env):
    return SoftEntryAction(self, env)

class SoftEntryAction(AiSapiensAction):
  def __init__(self, cfg, env):
    # The reference starts at the MEASURED settled pose. The controller starts
    # from the unchanged ReadyPose COMMAND, including compliance under gravity.
    ReadyHandoffAction.__init__(self, cfg, env)
    indices = [CONTRACT['joints'].index(j) for j in self.target_names]
    self.ready_target.copy_(torch.tensor(CONTRACT['ready_target'], device=self.device)[indices])
    self.applied_target[:] = self.ready_target
    self.previous_target[:] = self.ready_target
    self.actor_target = self.applied_target.clone()
    names = ('pelvis', 'torso_link')
    self.body_ids = [self._entity.body_names.index(n) for n in names]
    self.ref_ids = [self.command.cfg.body_names.index(n) for n in names]
    self.prev_velocity = torch.zeros(self.num_envs, 2, 4, device=self.device)
    self.velocity_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.interval_peaks = {k: torch.zeros(self.num_envs, device=self.device) for k in LIMITS}
    self.finalized = False
    for k in LIMITS:
      self.command.metrics['soft_entry_peak_' + k] = torch.zeros(self.num_envs, device=self.device)

  def reset_handoff(self, ids):
    super().reset_handoff(ids)
    self.velocity_valid[ids] = False
    self.prev_velocity[ids] = 0
    for k, v in self.interval_peaks.items():
      v[ids] = 0
      self.command.metrics['soft_entry_peak_' + k][ids] = 0

  def process_actions(self, actions):
    super().process_actions(actions)
    for v in self.interval_peaks.values():
      v.zero_()
    self.finalized = False
    # Previous interval already sampled this boundary in the reward call.
    # Start here afresh; include acceleration from this boundary to substep 1.
    self.velocity_valid.zero_()

  def sample_dynamics(self):
    c = self.command
    d = self._entity.data
    vz = d.body_com_lin_vel_w[:, self.body_ids, 2]
    omega = d.body_link_ang_vel_w[:, self.body_ids]
    vel = torch.cat((vz[..., None], omega), -1)
    accel = (vel - self.prev_velocity) / self._env.physics_dt
    accel *= self.velocity_valid[:, None, None]
    self.prev_velocity.copy_(vel)
    self.velocity_valid[:] = True
    # Matching COM quantities avoids treating the link-to-COM lever arm as error.
    ref_vz = c.body_lin_vel_w[:, self.ref_ids, 2]
    ref_omega = c.body_ang_vel_w[:, self.ref_ids]
    height_error = d.body_link_pos_w[:, self.body_ids, 2] - c.body_pos_w[:, self.ref_ids, 2]
    values = {'vertical_speed': (vz-ref_vz).abs().amax(-1),
      'angular_speed': torch.linalg.vector_norm(omega-ref_omega, dim=-1).amax(-1),
      'vertical_acceleration': accel[..., 0].abs().amax(-1),
      'angular_acceleration': torch.linalg.vector_norm(accel[..., 1:], dim=-1).amax(-1),
      'height_overshoot': height_error.clamp_min(0.).amax(-1)}
    for key, value in values.items():
      self.interval_peaks[key].copy_(torch.maximum(self.interval_peaks[key], value))

  def apply_actions(self):
    self.sample_dynamics()
    super().apply_actions()


def entry_dynamics(env, quantity):
  a = env.action_manager.get_term('joint_pos')
  c = a.command
  if not a.finalized:
    # Include the final physics state before env resets or reference advances.
    a.sample_dynamics()
    a.finalized = True
    for key, value in a.interval_peaks.items():
      metric = c.metrics['soft_entry_peak_' + key]
      metric.copy_(torch.maximum(metric, value * c.penalty_mask))
  limit, scale = LIMITS[quantity]
  # Full weight to 1 s, fading to zero by 1.5 s. Later rising is unaffected.
  taper = ((c.cfg.entry_penalty_seconds-c.elapsed)/.5).clamp(0., 1.)
  return ((a.interval_peaks[quantity]-limit).clamp_min(0.)/scale).square().clamp_max(25.) * c.penalty_mask * taper
