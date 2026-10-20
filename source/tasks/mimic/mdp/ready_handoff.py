# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""StandUp-only ReadyPose settling and PD handoff; see docs/stand_up_ready_handoff.md."""
from dataclasses import dataclass, field
import math
import re

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.event_manager import requires_model_fields

from .commands import ReferenceTrajectoryCommand
from .stand_up_curriculum import StandUpCurriculumCommand, StandUpCurriculumCommandCfg, curriculum_push


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def handoff_model_fields(env, env_ids):
  """Declare per-world gains before command/action managers are constructed."""


@dataclass(kw_only=True)
class ReadyHandoffCommandCfg(StandUpCurriculumCommandCfg):
  ready_hold_seconds: tuple[float, float] = (0.4, 0.8)
  handoff_seconds: tuple[float, float] = (0.6, 1.0)
  ready_timeout_seconds: float = 3.0
  stable_seconds: float = 0.1
  stable_joint_error_rad: float = 0.2
  stable_joint_speed_rad_s: float = 0.5
  stable_body_speed_rad_s: float = 0.4
  post_handoff_penalty_seconds: float = 0.3

  def build(self, env):
    for bounds in (self.ready_hold_seconds, self.handoff_seconds):
      if not all(math.isfinite(x) for x in bounds) or not 0 < bounds[0] <= bounds[1]:
        raise ValueError("Handoff durations must be positive finite ordered ranges.")
    for value in (self.ready_timeout_seconds, self.stable_seconds, self.stable_joint_error_rad,
                  self.stable_joint_speed_rad_s, self.stable_body_speed_rad_s,
                  self.post_handoff_penalty_seconds):
      if not math.isfinite(value) or value <= 0:
        raise ValueError("Handoff thresholds must be positive and finite.")
    if self.ready_timeout_seconds <= self.ready_hold_seconds[1]:
      raise ValueError("Ready timeout must exceed minimum hold duration.")
    if not 0 < self.first_frame_env_fraction <= 1:
      raise ValueError("Handoff requires a nonempty first-frame fraction.")
    return ReadyHandoffCommand(self, env)


class ReadyHandoffCommand(StandUpCurriculumCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.freeze_reference = self.enabled.clone()
    self.phase = torch.full((self.num_envs,), 2, dtype=torch.long, device=self.device)
    self.ready_elapsed = torch.zeros(self.num_envs, device=self.device)
    self.stable_elapsed = torch.zeros_like(self.ready_elapsed)
    self.blend_elapsed = torch.zeros_like(self.ready_elapsed)
    self.motion_elapsed = torch.zeros_like(self.ready_elapsed)
    self.hold_duration = torch.zeros_like(self.ready_elapsed)
    self.blend_duration = torch.ones_like(self.ready_elapsed)
    self.timeout = self.enabled.clone()
    self.penalty_mask = self.enabled.clone()
    for key in ("handoff_alpha", "handoff_phase", "handoff_ready_timeout",
                "handoff_target_delta_rad", "handoff_torque_delta_nm",
                "handoff_peak_target_delta_rad", "handoff_peak_torque_delta_nm"):
      self.metrics[key] = torch.zeros_like(self.ready_elapsed)

  def _resample_command(self, env_ids):
    ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
    if not len(ids):
      return
    # Do not carry a held velocity target into random intermediate-frame resets.
    self.freeze_reference[ids] = False
    super()._resample_command(ids)
    self.enabled[ids] = (ids < int(self.num_envs * self.cfg.first_frame_env_fraction)) | self.cfg.start_from_zero
    self.freeze_reference[ids] = self.enabled[ids]
    self.phase[ids] = torch.where(self.enabled[ids], 0, 2)
    for tensor in (self.ready_elapsed, self.stable_elapsed, self.blend_elapsed, self.motion_elapsed):
      tensor[ids] = 0
    self.timeout[ids] = False
    self.penalty_mask[ids] = self.enabled[ids]
    for tensor, bounds in ((self.hold_duration, self.cfg.ready_hold_seconds),
                           (self.blend_duration, self.cfg.handoff_seconds)):
      tensor[ids] = bounds[0] + torch.rand(len(ids), device=self.device) * (bounds[1] - bounds[0])
    for key in self.metrics:
      if key.startswith("handoff_"):
        self.metrics[key][ids] = 0
    # Also reset transition history on clip-end teleports, not only env resets.
    self._env.action_manager.reset(ids)
    self._env.action_manager.get_term("joint_pos").reset_handoff(ids)

  def _update_command(self):
    self.frame_ids -= self.freeze_reference.long()
    super()._update_command()

  def refresh_reference(self):
    """Refresh alignment after explicit reset without advancing either clock."""
    self.frame_ids -= 1
    ReferenceTrajectoryCommand._update_command(self)

  def _held_velocity(self, value):
    return torch.where(self.freeze_reference.reshape((-1,) + (1,) * (value.ndim - 1)), 0., value)

  @property
  def joint_vel(self):
    return self._held_velocity(super().joint_vel)

  @property
  def body_lin_vel_w(self):
    return self._held_velocity(super().body_lin_vel_w)

  @property
  def body_ang_vel_w(self):
    return self._held_velocity(super().body_ang_vel_w)

  @property
  def anchor_lin_vel_w(self):
    return self._held_velocity(super().anchor_lin_vel_w)

  @property
  def anchor_ang_vel_w(self):
    return self._held_velocity(super().anchor_ang_vel_w)

  def advance_handoff(self):
    """Once per control step, before applying the action. Alpha includes endpoints."""
    dt = self._env.step_dt
    ready = self.enabled & (self.phase == 0)
    stable = ((self.robot_joint_pos - self.reference.joint_position[0]).abs().amax(-1) < self.cfg.stable_joint_error_rad)
    stable &= self.robot_joint_vel.abs().amax(-1) < self.cfg.stable_joint_speed_rad_s
    stable &= self.robot_anchor_ang_vel_w.norm(dim=-1) < self.cfg.stable_body_speed_rad_s
    self.stable_elapsed = torch.where(ready & stable, self.stable_elapsed + dt, 0.)
    enter = ready & (self.ready_elapsed >= self.hold_duration) & (self.stable_elapsed >= self.cfg.stable_seconds)
    self.phase[enter] = 1
    self.timeout = ready & ~enter & (self.ready_elapsed >= self.cfg.ready_timeout_seconds)
    blend = self.enabled & (self.phase == 1)
    alpha = torch.where(self.enabled & (self.phase == 0), 0., 1.)
    alpha = torch.where(blend, (self.blend_elapsed / self.blend_duration).clamp(0., 1.), alpha)
    # Hold through the first full-policy step; advance on the following step.
    self.freeze_reference = self.enabled & (self.phase != 2)
    self.penalty_mask = self.enabled & ((self.phase != 2) | (self.motion_elapsed < self.cfg.post_handoff_penalty_seconds))
    self.metrics["handoff_alpha"][:] = alpha
    self.metrics["handoff_phase"][:] = self.phase
    self.metrics["handoff_ready_timeout"][:] = self.timeout.float()
    self.ready_elapsed += ready * dt
    self.blend_elapsed += blend * dt
    self.motion_elapsed += (self.enabled & (self.phase == 2)) * dt
    self.phase[blend & (alpha >= 1.)] = 2
    return alpha


@dataclass(kw_only=True)
class ReadyHandoffActionCfg(JointPositionActionCfg):
  # Reported hardware values; all unlisted joints retain policy gains.
  ready_kp: dict[str, float] = field(default_factory=lambda: {
    ".*_knee_joint": 150., ".*_shoulder_.*_joint": 40., ".*_elbow_joint": 40.,
  })
  ready_kd: dict[str, float] = field(default_factory=lambda: {
    ".*_shoulder_.*_joint": 10., ".*_elbow_joint": 10.,
  })

  def build(self, env):
    return ReadyHandoffAction(self, env)


class ReadyHandoffAction(JointPositionAction):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.command = env.command_manager.get_term("reference_trajectory")
    model = env.sim.mj_model
    joint_ids = self._entity.indexing.joint_ids[self.target_ids].cpu().tolist()
    # Resolve by transmission joint, not by actuator-array order.
    ids = []
    for joint in joint_ids:
      matches = [int(i) for i in self._entity.indexing.ctrl_ids.cpu().tolist()
                 if int(model.actuator_trnid[i, 0]) == joint]
      if len(matches) != 1:
        raise ValueError("Ready handoff requires one position actuator per joint.")
      ids.append(matches[0])
    self.ctrl_ids = torch.tensor(ids, device=self.device)
    self.policy_kp = env.sim.model.actuator_gainprm[0, self.ctrl_ids, 0].clone()
    self.policy_kd = -env.sim.model.actuator_biasprm[0, self.ctrl_ids, 2].clone()
    self.ready_kp, self.ready_kd = self.policy_kp.clone(), self.policy_kd.clone()
    for target, overrides in ((self.ready_kp, cfg.ready_kp), (self.ready_kd, cfg.ready_kd)):
      for pattern, value in overrides.items():
        selected = [i for i, name in enumerate(self.target_names) if re.fullmatch(pattern, name)]
        if not selected or not math.isfinite(value) or value <= 0:
          raise ValueError(f"Invalid ReadyPose gain override: {pattern}={value}")
        target[selected] = value
    self.ready_target = self.command.reference.joint_position[0, self.target_ids].clone()
    self.applied_target = self.ready_target.repeat(self.num_envs, 1)
    self.previous_target = self.applied_target.clone()
    self.target_delta = torch.zeros_like(self.applied_target)
    self.previous_torque = torch.zeros_like(self.applied_target)
    self.torque_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def reset_handoff(self, ids):
    self.applied_target[ids] = self.ready_target
    self.previous_target[ids] = self.ready_target
    self.target_delta[ids] = 0
    self.previous_torque[ids] = 0
    self.torque_valid[ids] = False

  def process_actions(self, actions):
    super().process_actions(actions)
    alpha = self.command.advance_handoff()[:, None]
    self.previous_target.copy_(self.applied_target)
    self.applied_target = torch.lerp(self.ready_target.expand_as(self._processed_actions), self._processed_actions, alpha)
    self.target_delta = self.applied_target - self.previous_target
    self._processed_actions = self.applied_target
    kp = torch.lerp(self.ready_kp.expand_as(self.applied_target), self.policy_kp.expand_as(self.applied_target), alpha)
    kd = torch.lerp(self.ready_kd.expand_as(self.applied_target), self.policy_kd.expand_as(self.applied_target), alpha)
    self._env.sim.model.actuator_gainprm[:, self.ctrl_ids, 0] = kp
    self._env.sim.model.actuator_biasprm[:, self.ctrl_ids, 1] = -kp
    self._env.sim.model.actuator_biasprm[:, self.ctrl_ids, 2] = -kd


def handoff_timeout(env, command_name="reference_trajectory"):
  return env.command_manager.get_term(command_name).timeout


def handoff_push(env, env_ids, velocity_range):
  c = env.command_manager.get_term("reference_trajectory")
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  ids = ids[~c.freeze_reference[ids]]
  if len(ids):
    curriculum_push(env, ids, velocity_range)


def handoff_target_rate(env, action_name="joint_pos", scale_rad=0.05):
  a = env.action_manager.get_term(action_name)
  c = a.command
  delta = a.target_delta.abs().amax(-1) * c.penalty_mask
  c.metrics["handoff_target_delta_rad"][:] = delta
  c.metrics["handoff_peak_target_delta_rad"][:] = torch.maximum(c.metrics["handoff_peak_target_delta_rad"], delta)
  return (a.target_delta / scale_rad).square().mean(-1).clamp_max(25.) * c.penalty_mask


def handoff_torque_rate(env, action_name="joint_pos", scale_nm=10.):
  a = env.action_manager.get_term(action_name)
  c = a.command
  torque = env.sim.data.actuator_force[:, a.ctrl_ids]
  delta = (torque - a.previous_torque) * a.torque_valid[:, None]
  a.previous_torque.copy_(torque)
  a.torque_valid[:] = True
  peak = delta.abs().amax(-1) * c.penalty_mask
  c.metrics["handoff_torque_delta_nm"][:] = peak
  c.metrics["handoff_peak_torque_delta_nm"][:] = torch.maximum(c.metrics["handoff_peak_torque_delta_nm"], peak)
  return (delta / scale_nm).square().mean(-1).clamp_max(25.) * c.penalty_mask
