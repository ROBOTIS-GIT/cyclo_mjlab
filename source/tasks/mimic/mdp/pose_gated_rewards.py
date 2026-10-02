# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Velocity tracking rewards gated by the quality of the tracked body pose."""

from .rewards import (
  reference_body_position_tracking,
  reference_body_orientation_tracking,
  reference_body_linear_velocity_tracking,
  reference_body_angular_velocity_tracking,
)


def _pose_gate(env, command_name, position_std, orientation_std, body_names):
  return reference_body_position_tracking(
    env, command_name, position_std, body_names
  ) * reference_body_orientation_tracking(
    env, command_name, orientation_std, body_names
  )


def pose_gated_linear_velocity_tracking(
  env, command_name, std, body_names=None,
  position_std=0.3, orientation_std=0.4,
):
  """Preserve velocity reward at the target pose; suppress it away from it."""
  return reference_body_linear_velocity_tracking(
    env, command_name, std, body_names
  ) * _pose_gate(env, command_name, position_std, orientation_std, body_names)


def pose_gated_angular_velocity_tracking(
  env, command_name, std, body_names=None,
  position_std=0.3, orientation_std=0.4,
):
  return reference_body_angular_velocity_tracking(
    env, command_name, std, body_names
  ) * _pose_gate(env, command_name, position_std, orientation_std, body_names)
