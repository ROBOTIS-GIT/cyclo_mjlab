# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Success-gated stand-up curriculum for learning from random initialization."""

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.events import push_by_setting_velocity

from .commands import ReferenceTrajectoryCommand, ReferenceTrajectoryCommandCfg


class StandUpCurriculumCommand(ReferenceTrajectoryCommand):
  """Count complete attempts of the fixed frame-zero group before teleporting.

  Partial-motion starts and initial resets never count as successes. Stages
  restart at zero in a new process; this task is intended for scratch training.
  """

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.difficulty_stage = 0
    self.attempts = 0
    self.successes = 0
    self.last_success_rate = 0.0
    self.trial_stage = torch.full((self.num_envs,), -1, device=self.device)
    self.base_pose_noise = dict(cfg.reset_pose_noise)
    self.base_velocity_noise = dict(cfg.reset_velocity_noise)
    for key in ("full_motion_success_rate", "difficulty_stage", "full_motion_attempts"):
      self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

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
    success = (complete & ~self._env.termination_manager.terminated[eligible]
               & ((actual_z - target_z).abs() < 0.1) & (upright > 0.93969262))
    self.attempts += len(eligible)
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


@dataclass(kw_only=True)
class StandUpCurriculumCommandCfg(ReferenceTrajectoryCommandCfg):
  curriculum_min_attempts: int = 1024
  curriculum_success_threshold: float = 0.8

  def build(self, env):
    if self.curriculum_min_attempts < 1 or not 0 < self.curriculum_success_threshold <= 1:
      raise ValueError("Invalid stand-up curriculum gate.")
    if not 0 < self.first_frame_env_fraction <= 1 or not 0 <= self.start_sampling_uniform_mix <= 1:
      raise ValueError("Invalid stand-up sampling fractions.")
    return StandUpCurriculumCommand(self, env)


def curriculum_push(env, env_ids, velocity_range, command_name="reference_trajectory"):
  scale = (0.0, 0.25, 0.5, 1.0)[env.command_manager.get_term(command_name).difficulty_stage]
  if scale == 0:
    return
  push_by_setting_velocity(env, env_ids, velocity_range={k: (a * scale, b * scale) for k, (a, b) in velocity_range.items()})
