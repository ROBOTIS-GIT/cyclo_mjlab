# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Scratch training: balanced starts and success-gated domain randomization."""

from dataclasses import fields
from source.tasks.mimic.mdp.pose_gated_rewards import (
  pose_gated_linear_velocity_tracking,
  pose_gated_angular_velocity_tracking,
)

from source.tasks.mimic.mdp.stand_up_curriculum import (
  StandUpCurriculumCommandCfg,
  curriculum_push,
)
from .umr_stand_up_low_friction_env_cfg import k1_rev1_umr_stand_up_low_friction_env_cfg


def k1_rev1_umr_stand_up_scratch_env_cfg(play=False):
  cfg = k1_rev1_umr_stand_up_low_friction_env_cfg(play=play)
  cfg.rewards["reference_body_linear_velocity"].func = pose_gated_linear_velocity_tracking
  cfg.rewards["reference_body_angular_velocity"].func = pose_gated_angular_velocity_tracking
  if play:
    return cfg
  old = cfg.commands["reference_trajectory"]
  command = StandUpCurriculumCommandCfg(**{f.name: getattr(old, f.name) for f in fields(old)})
  command.start_sampling_uniform_mix = 0.5
  command.first_frame_joint_position_noise = (-0.05, 0.05)
  command.reset_joint_position_noise = (-0.05, 0.05)
  cfg.commands["reference_trajectory"] = command
  material = cfg.events["physics_material"]
  material.mode = "startup"
  material.params["curriculum_command"] = "reference_trajectory"
  cfg.events["push_robot"].func = curriculum_push
  return cfg
