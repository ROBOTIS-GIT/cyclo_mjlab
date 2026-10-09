# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Koryo perturbation curriculum gated by frame-zero, 30-second attempts."""

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.events import push_by_setting_velocity
from mjlab.utils.noise import UniformNoiseCfg

from .commands import ReferenceTrajectoryCommand, ReferenceTrajectoryCommandCfg
from .events import (
  apply_home_joint_offset_noise,
  apply_link_com_offset_noise,
  randomize_rigid_body_material,
)

NOISE_SCALES = (0.25, 0.5, 0.75, 1.0)
PUSH_SCALES = (0.0, 0.25, 0.5, 1.0)


def scaled_ranges(ranges, scale):
  return {key: (low * scale, high * scale) for key, (low, high) in ranges.items()}


@dataclass
class KoryoObservationNoise(UniformNoiseCfg):
  """Scale the sampled noise, keeping cached base bounds unchanged."""
  curriculum_scale: float = 0.25

  def apply(self, data):
    if self.operation != "add":
      raise ValueError("Koryo observation noise requires operation='add'.")
    return data + self.curriculum_scale * (super().apply(data) - data)


class KoryoCurriculumCommand(ReferenceTrajectoryCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.difficulty_stage = 0
    self.attempts = 0
    self.successes = 0
    self.last_success_rate = 0.0
    self.trial_stage = torch.full((self.num_envs,), -1, device=self.device)
    self.base_pose_noise = dict(cfg.reset_pose_noise)
    self.base_velocity_noise = dict(cfg.reset_velocity_noise)
    self.base_joint_noise = cfg.reset_joint_position_noise

  def curriculum_state_dict(self):
    return {key: getattr(self, key) for key in (
      "difficulty_stage", "attempts", "successes", "last_success_rate",
    )}

  def load_curriculum_state_dict(self, state):
    stage, attempts, successes = (
      int(state[key]) for key in ("difficulty_stage", "attempts", "successes")
    )
    rate = float(state["last_success_rate"])
    if not (0 <= stage < len(NOISE_SCALES) and 0 <= successes <= attempts
            and 0 <= rate <= 1):
      raise ValueError("Invalid Koryo curriculum checkpoint state.")
    self.difficulty_stage, self.attempts, self.successes = stage, attempts, successes
    self.last_success_rate = rate
    # Checkpoints contain no in-flight physics state.
    self.trial_stage.fill_(-1)

  def record_outcomes(self, env_ids):
    """Called before physics/reset events, never on clip-end resampling.

    Count only completed frame-zero attempts from the current stage. Ignore
    initial/manual resets, partial starts, and attempts begun at an older stage.
    A failure coinciding with timeout is still a failure.
    """
    ids = torch.arange(self.num_envs, device=self.device)[env_ids]
    ids = ids[(ids < int(self.num_envs * self.cfg.first_frame_env_fraction))
              & (self.trial_stage[ids] == self.difficulty_stage)]
    terminated = self._env.termination_manager.terminated[ids]
    complete = self._env.episode_length_buf[ids] >= self._env.max_episode_length
    eligible = (self._env.episode_length_buf[ids] > 0) & (terminated | complete)
    self.attempts += int(eligible.sum())
    self.successes += int((eligible & complete & ~terminated).sum())
    # Prevent duplicate accounting if an explicit reset repeats.
    self.trial_stage[ids] = -1
    if self.attempts >= self.cfg.curriculum_min_attempts:
      self.last_success_rate = self.successes / self.attempts
      if self.last_success_rate >= self.cfg.curriculum_success_threshold:
        self.difficulty_stage = min(self.difficulty_stage + 1, len(NOISE_SCALES) - 1)
      self.attempts = self.successes = 0

  def _resample_command(self, env_ids):
    scale = NOISE_SCALES[self.difficulty_stage]
    self.cfg.reset_pose_noise = scaled_ranges(self.base_pose_noise, scale)
    self.cfg.reset_velocity_noise = scaled_ranges(self.base_velocity_noise, scale)
    self.cfg.reset_joint_position_noise = tuple(x * scale for x in self.base_joint_noise)
    super()._resample_command(env_ids)
    self.trial_stage[env_ids] = self.difficulty_stage


@dataclass(kw_only=True)
class KoryoCurriculumCommandCfg(ReferenceTrajectoryCommandCfg):
  curriculum_min_attempts: int = 1024
  curriculum_success_threshold: float = 0.8

  def build(self, env):
    if not (0 < self.first_frame_env_fraction <= 1
            and int(env.num_envs * self.first_frame_env_fraction) >= 1
            and 0 <= self.start_sampling_uniform_mix <= 1
            and self.curriculum_min_attempts >= 1
            and 0 < self.curriculum_success_threshold <= 1):
      raise ValueError("Invalid Koryo curriculum gate or fixed-start group size.")
    if self.start_from_zero or self.first_frame_joint_position_noise is not None:
      raise ValueError("Koryo curriculum requires mixed starts and shared reset noise.")
    command = KoryoCurriculumCommand(self, env)
    if command.reference.num_frames <= env.max_episode_length:
      raise ValueError("Koryo gate requires a reference longer than the episode cap.")
    return command


def update_koryo_difficulty(env, env_ids):
  command = env.command_manager.get_term("reference_trajectory")
  command.record_outcomes(env_ids)
  scale = NOISE_SCALES[command.difficulty_stage]
  for name in env.observation_manager.active_terms["actor"]:
    noise = env.observation_manager.get_term_cfg("actor", name).noise
    if isinstance(noise, KoryoObservationNoise):
      noise.curriculum_scale = scale
  return {
    "stage": command.difficulty_stage,
    "noise_scale": scale,
    "push_scale": PUSH_SCALES[command.difficulty_stage],
    "frame_zero_30s_success_rate": command.last_success_rate,
    "attempts_in_window": command.attempts,
  }


def _scale(env):
  return NOISE_SCALES[env.command_manager.get_term("reference_trajectory").difficulty_stage]


def koryo_material(env, env_ids, ranges, **kwargs):
  midpoint = sum(ranges) / 2
  bounds = tuple(midpoint + (x - midpoint) * _scale(env) for x in ranges)
  randomize_rigid_body_material(env, env_ids, ranges=bounds, **kwargs)


def koryo_com_offset(env, env_ids, com_range, **kwargs):
  apply_link_com_offset_noise(env, env_ids, com_range=scaled_ranges(com_range, _scale(env)), **kwargs)


def koryo_home_offset(env, env_ids, pos_distribution_params, **kwargs):
  bounds = tuple(x * _scale(env) for x in pos_distribution_params)
  apply_home_joint_offset_noise(env, env_ids, pos_distribution_params=bounds, **kwargs)


def koryo_push(env, env_ids, velocity_range):
  scale = PUSH_SCALES[env.command_manager.get_term("reference_trajectory").difficulty_stage]
  if scale:
    push_by_setting_velocity(env, env_ids, velocity_range=scaled_ranges(velocity_range, scale))


# Preserve MJLab's per-environment field expansion and recomputation contracts.
koryo_material.model_fields = randomize_rigid_body_material.model_fields
koryo_material.recompute = randomize_rigid_body_material.recompute
koryo_com_offset.model_fields = apply_link_com_offset_noise.model_fields
koryo_com_offset.recompute = apply_link_com_offset_noise.recompute
