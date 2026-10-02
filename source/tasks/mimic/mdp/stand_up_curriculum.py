# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Success-gated stand-up curriculum for learning from random initialization."""

from dataclasses import dataclass
import math

import torch
from mjlab.envs.mdp.events import push_by_setting_velocity

from .commands import ReferenceTrajectoryCommand, ReferenceTrajectoryCommandCfg


class StandUpCurriculumCommand(ReferenceTrajectoryCommand):
  """Count complete attempts of the fixed frame-zero group before teleporting.

  Partial-motion starts, initial resets and interrupted attempts are excluded.
  """

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.difficulty_stage = 0
    self.attempts = 0
    self.successes = 0
    self.last_success_rate = 0.0
    self.completed_successes = 0
    self.failed_attempts = 0
    self.interrupted_attempts = 0
    self.trial_stage = torch.full((self.num_envs,), -1, device=self.device)
    self.base_pose_noise = dict(cfg.reset_pose_noise)
    self.base_velocity_noise = dict(cfg.reset_velocity_noise)
    for key in ("full_motion_success_rate", "difficulty_stage", "full_motion_attempts"):
      self.metrics[key] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["preparation_start_fraction"] = torch.zeros(self.num_envs, device=self.device)
    for key in ("full_motion_successes_total", "full_motion_failures_total", "full_motion_interruptions_total"):
      self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

  def curriculum_state_dict(self):
    return {key: getattr(self, key) for key in (
      "difficulty_stage", "attempts", "successes", "last_success_rate",
      "completed_successes", "failed_attempts", "interrupted_attempts",
    )}

  def load_curriculum_state_dict(self, state):
    stage = int(state["difficulty_stage"])
    if not 0 <= stage <= 3:
      raise ValueError("Invalid saved curriculum stage.")
    for key in self.curriculum_state_dict():
      if key in state:
        setattr(self, key, state[key])
    # Physics state is not restored by PPO: discard in-flight attempt labels.
    self.trial_stage.fill_(-1)

  def _sample_start_frames(self, env_ids):
    super()._sample_start_frames(env_ids)
    if self.cfg.start_from_zero:
      return
    ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
    ids = ids[ids >= int(self.num_envs * self.cfg.first_frame_env_fraction)]
    chosen = ids[torch.rand(len(ids), device=self.device) < self.cfg.preparation_start_fraction]
    low, high = self.cfg.preparation_start_seconds
    first = math.ceil(low / self._env.step_dt)
    last = min(math.floor(high / self._env.step_dt), self.reference.num_frames - 1)
    self.frame_ids[chosen] = torch.randint(first, last + 1, (len(chosen),), device=self.device)
    self.metrics["preparation_start_fraction"][:] = self.cfg.preparation_start_fraction

  def _record_outcomes(self, env_ids):
    fixed = env_ids[env_ids < int(self.num_envs * self.cfg.first_frame_env_fraction)]
    eligible = fixed[self.trial_stage[fixed] == self.difficulty_stage]
    if len(eligible) == 0:
      return
    complete = self.frame_ids[eligible] >= self.reference.num_frames
    # The reference index is already past the end when a clip completes.
    target_z = self.reference.body_position_w[-1, self.reference_anchor_body_id, 2]
    actual_z = self.robot_anchor_pos_w[eligible, 2] - self._env.scene.env_origins[eligible, 2]
    q = self.robot_anchor_quat_w[eligible]
    upright = 1.0 - 2.0 * (q[:, 1].square() + q[:, 2].square())
    terminated = self._env.termination_manager.terminated[eligible]
    interrupted = ~complete & ~terminated
    success = (complete & ~terminated
               & ((actual_z - target_z).abs() < 0.1) & (upright > 0.93969262))
    failures = ~success & ~interrupted
    self.completed_successes += int(success.sum())
    self.failed_attempts += int(failures.sum())
    self.interrupted_attempts += int(interrupted.sum())
    self.attempts += int((~interrupted).sum())
    self.successes += int(success.sum())
    if self.attempts >= self.cfg.curriculum_min_attempts:
      self.last_success_rate = self.successes / self.attempts
      if self.last_success_rate >= self.cfg.curriculum_success_threshold:
        self.difficulty_stage = min(self.difficulty_stage + 1, 3)
      self.attempts = self.successes = 0

  def _resample_command(self, env_ids):
    env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
    if len(env_ids) == 0:
      return
    self._record_outcomes(env_ids)
    stage = self.difficulty_stage
    scale = (0.25, 0.5, 0.75, 1.0)[stage]
    self.cfg.reset_pose_noise = {k: (a * scale, b * scale) for k, (a, b) in self.base_pose_noise.items()}
    self.cfg.reset_velocity_noise = {k: (a * scale, b * scale) for k, (a, b) in self.base_velocity_noise.items()}
    noise = (0.05, 0.1, 0.15, 0.2)[stage]
    self.cfg.first_frame_joint_position_noise = (-noise, noise)
    noise = (0.05, 0.075, 0.1, 0.1)[stage]
    self.cfg.reset_joint_position_noise = (-noise, noise)
    super()._resample_command(env_ids)
    self.trial_stage[env_ids] = stage
    # Clip-end teleportation does not trigger an environment reset event.
    # Resample the material here as well so every new attempt uses its stage.
    material = self._env.event_manager.get_term_cfg("physics_material")
    material.func(self._env, env_ids, **material.params)
    self.metrics["full_motion_success_rate"][:] = self.last_success_rate
    self.metrics["difficulty_stage"][:] = stage
    self.metrics["full_motion_attempts"][:] = self.attempts
    self.metrics["full_motion_successes_total"][:] = self.completed_successes
    self.metrics["full_motion_failures_total"][:] = self.failed_attempts
    self.metrics["full_motion_interruptions_total"][:] = self.interrupted_attempts


@dataclass(kw_only=True)
class StandUpCurriculumCommandCfg(ReferenceTrajectoryCommandCfg):
  preparation_start_fraction: float = 0.4
  """Fraction of non-fixed resets replaced with uniform preparation-window starts."""
  preparation_start_seconds: tuple[float, float] = (5.0, 7.0)
  curriculum_min_attempts: int = 1024
  curriculum_success_threshold: float = 0.8

  def build(self, env):
    low, high = self.preparation_start_seconds
    if not 0 <= self.preparation_start_fraction <= 1 or not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
      raise ValueError("Invalid preparation-start range or fraction.")
    if self.curriculum_min_attempts < 1 or not 0 < self.curriculum_success_threshold <= 1:
      raise ValueError("Invalid stand-up curriculum gate.")
    if not 0 < self.first_frame_env_fraction <= 1 or not 0 <= self.start_sampling_uniform_mix <= 1:
      raise ValueError("Invalid stand-up sampling fractions.")
    command = StandUpCurriculumCommand(self, env)
    if math.ceil(low / env.step_dt) > min(math.floor(high / env.step_dt), command.reference.num_frames - 1):
      raise ValueError("Preparation-start window does not contain a reference frame.")
    return command


def curriculum_push(env, env_ids, velocity_range, command_name="reference_trajectory"):
  scale = (0.0, 0.25, 0.5, 1.0)[env.command_manager.get_term(command_name).difficulty_stage]
  if scale == 0:
    return
  push_by_setting_velocity(env, env_ids, velocity_range={k: (a * scale, b * scale) for k, (a, b) in velocity_range.items()})
