# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""K1 Rev.1 UMR StandUpBack task using the v2 reference motion."""
from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from source import SRC_PATH
from source.tasks.mimic.mdp.pose_gated_rewards import (
  pose_gated_angular_velocity_tracking,
  pose_gated_linear_velocity_tracking,
)
from source.tasks.mimic.mdp.stand_up_curriculum import (
  StandUpCurriculumCommandCfg,
  curriculum_push,
)
from .umr_stand_up_env_cfg import k1_rev1_umr_stand_up_env_cfg


TRAJECTORY_FILE = (
  SRC_PATH / "assets" / "motions" / "K1_rev1" / "umr_stand_up_back"
  / "k1_stand_up_back_v2_converted.npz"
)


def k1_rev1_umr_stand_up_back_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # Retain stand-up contact handling, hard-limit-aligned soft limits, and
  # the 30% frame-zero reset group with joint-angle randomization.
  cfg = k1_rev1_umr_stand_up_env_cfg(play=play)
  cfg.commands["reference_trajectory"].trajectory_file = str(TRAJECTORY_FILE)
  cfg.rewards["reference_body_linear_velocity"].func = pose_gated_linear_velocity_tracking
  cfg.rewards["reference_body_angular_velocity"].func = pose_gated_angular_velocity_tracking
  if play:
    return cfg

  old = cfg.commands["reference_trajectory"]
  command = StandUpCurriculumCommandCfg(**{f.name: getattr(old, f.name) for f in fields(old)})
  # Train the approach to the failed rise, rather than mostly teleporting into
  # the already-rising pose at 9.7 s. Keep the fixed 30% full-motion group.
  command.preparation_start_seconds = (7.5, 9.5)
  command.preparation_start_fraction = 0.4
  command.start_sampling_uniform_mix = 0.5
  command.first_frame_joint_position_noise = (-0.05, 0.05)
  command.reset_joint_position_noise = (-0.05, 0.05)
  # This task keeps its existing startup friction DR. Only reset noise and
  # pushes grow with full-motion success; no per-attempt material resampling.
  command.resample_material_on_reset = False
  cfg.commands["reference_trajectory"] = command
  cfg.events["push_robot"].func = curriculum_push
  return cfg
