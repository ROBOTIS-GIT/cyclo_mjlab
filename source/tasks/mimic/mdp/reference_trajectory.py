# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Additional notices:
# This module is adapted from HybridRobotics/whole_body_tracking, licensed under
# the MIT License. See THIRD_PARTY_LICENSES.md for details.

"""Reference trajectory storage for Mimic tasks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class ReferenceTrajectory:
  """Tensor view over a named MuJoCo reference trajectory NPZ file."""

  REQUIRED_KEYS = (
    "fps",
    "joint_names",
    "body_names",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  )

  def __init__(
    self,
    trajectory_file: str,
    robot_joint_names: tuple[str, ...],
    robot_body_names: tuple[str, ...],
    tracked_body_names: tuple[str, ...],
    device: str = "cpu",
  ):
    path = Path(trajectory_file)
    if not path.is_file():
      raise FileNotFoundError(f"Reference trajectory file does not exist: {path}")

    try:
      with np.load(path, allow_pickle=False) as data:
        missing = [key for key in self.REQUIRED_KEYS if key not in data]
        if missing:
          raise KeyError(f"Reference trajectory file is missing keys: {missing}")

        trajectory_joint_names = self._read_names(
          data["joint_names"], "joint_names"
        )
        trajectory_body_names = self._read_names(data["body_names"], "body_names")
        self._validate_name_set(
          trajectory_joint_names,
          robot_joint_names,
          "joint_names",
          "MuJoCo joints",
        )
        self._validate_name_set(
          trajectory_body_names,
          robot_body_names,
          "body_names",
          "MuJoCo bodies",
        )
        self._validate_unique_names(tracked_body_names, "tracked body names")
        unknown_tracked_bodies = sorted(
          set(tracked_body_names) - set(robot_body_names)
        )
        if unknown_tracked_bodies:
          raise ValueError(
            "Tracked bodies are not present in the MuJoCo model: "
            f"{unknown_tracked_bodies}"
          )

        fps = np.asarray(data["fps"])
        if fps.size != 1:
          raise ValueError(f"fps must contain one value, got shape {fps.shape}")
        self.fps = float(fps.reshape(-1)[0])
        if not np.isfinite(self.fps) or self.fps <= 0.0:
          raise ValueError(f"fps must be positive and finite, got {self.fps}")

        joint_position = np.asarray(data["joint_pos"])
        joint_velocity = np.asarray(data["joint_vel"])
        body_position_w = np.asarray(data["body_pos_w"])
        body_orientation_w = np.asarray(data["body_quat_w"])
        body_linear_velocity_w = np.asarray(data["body_lin_vel_w"])
        body_angular_velocity_w = np.asarray(data["body_ang_vel_w"])
    except (OSError, ValueError, KeyError) as error:
      raise ValueError(f"Invalid reference trajectory '{path}': {error}") from error

    arrays = {
      "joint_pos": joint_position,
      "joint_vel": joint_velocity,
      "body_pos_w": body_position_w,
      "body_quat_w": body_orientation_w,
      "body_lin_vel_w": body_linear_velocity_w,
      "body_ang_vel_w": body_angular_velocity_w,
    }
    self._validate_array_shapes(
      arrays,
      joint_count=len(trajectory_joint_names),
      body_count=len(trajectory_body_names),
    )
    non_finite = [
      name for name, values in arrays.items() if not np.all(np.isfinite(values))
    ]
    if non_finite:
      raise ValueError(
        f"Invalid reference trajectory '{path}': non-finite values in {non_finite}"
      )

    joint_index = {
      name: index for index, name in enumerate(trajectory_joint_names)
    }
    body_index = {
      name: index for index, name in enumerate(trajectory_body_names)
    }
    joint_order = [joint_index[name] for name in robot_joint_names]
    tracked_body_ids = [body_index[name] for name in tracked_body_names]

    self.joint_names = robot_joint_names
    self.body_names = tracked_body_names
    self.joint_position = torch.as_tensor(
      joint_position[:, joint_order], dtype=torch.float32, device=device
    )
    self.joint_velocity = torch.as_tensor(
      joint_velocity[:, joint_order], dtype=torch.float32, device=device
    )
    self._body_position_w = torch.as_tensor(
      body_position_w, dtype=torch.float32, device=device
    )
    self._body_orientation_w = torch.as_tensor(
      body_orientation_w, dtype=torch.float32, device=device
    )
    self._body_linear_velocity_w = torch.as_tensor(
      body_linear_velocity_w, dtype=torch.float32, device=device
    )
    self._body_angular_velocity_w = torch.as_tensor(
      body_angular_velocity_w, dtype=torch.float32, device=device
    )
    self._tracked_body_ids = tracked_body_ids
    self.num_frames = self.joint_position.shape[0]

  @property
  def body_position_w(self) -> torch.Tensor:
    return self._body_position_w[:, self._tracked_body_ids]

  @property
  def body_orientation_w(self) -> torch.Tensor:
    return self._body_orientation_w[:, self._tracked_body_ids]

  @property
  def body_linear_velocity_w(self) -> torch.Tensor:
    return self._body_linear_velocity_w[:, self._tracked_body_ids]

  @property
  def body_angular_velocity_w(self) -> torch.Tensor:
    return self._body_angular_velocity_w[:, self._tracked_body_ids]

  @classmethod
  def _read_names(cls, values: np.ndarray, field_name: str) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
      raise ValueError(
        f"{field_name} must be a one-dimensional string array, "
        f"got shape {array.shape} and dtype {array.dtype}"
      )
    names = tuple(
      value.decode("utf-8") if isinstance(value, bytes) else str(value)
      for value in array
    )
    if not names or any(not name for name in names):
      raise ValueError(f"{field_name} must contain non-empty names")
    cls._validate_unique_names(names, field_name)
    return names

  @staticmethod
  def _validate_unique_names(names: tuple[str, ...], field_name: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
      if name in seen:
        duplicates.add(name)
      seen.add(name)
    if duplicates:
      raise ValueError(f"{field_name} contains duplicate names: {sorted(duplicates)}")

  @classmethod
  def _validate_name_set(
    cls,
    trajectory_names: tuple[str, ...],
    model_names: tuple[str, ...],
    trajectory_field_name: str,
    model_label: str,
  ) -> None:
    cls._validate_unique_names(model_names, model_label)
    missing = sorted(set(model_names) - set(trajectory_names))
    extra = sorted(set(trajectory_names) - set(model_names))
    if missing or extra:
      raise ValueError(
        f"{trajectory_field_name} does not match {model_label}: "
        f"missing={missing}, extra={extra}"
      )

  @staticmethod
  def _validate_array_shapes(
    arrays: dict[str, np.ndarray], joint_count: int, body_count: int
  ) -> None:
    joint_position = arrays["joint_pos"]
    if joint_position.ndim != 2:
      raise ValueError(
        f"joint_pos must be 2D, got shape {joint_position.shape}"
      )
    num_frames = joint_position.shape[0]
    if num_frames < 1:
      raise ValueError("Reference trajectory must contain at least one frame")

    expected_shapes = {
      "joint_pos": (num_frames, joint_count),
      "joint_vel": (num_frames, joint_count),
      "body_pos_w": (num_frames, body_count, 3),
      "body_quat_w": (num_frames, body_count, 4),
      "body_lin_vel_w": (num_frames, body_count, 3),
      "body_ang_vel_w": (num_frames, body_count, 3),
    }
    invalid_shapes = {
      name: (values.shape, expected_shapes[name])
      for name, values in arrays.items()
      if values.shape != expected_shapes[name]
    }
    if invalid_shapes:
      details = ", ".join(
        f"{name}={actual} (expected {expected})"
        for name, (actual, expected) in invalid_shapes.items()
      )
      raise ValueError(f"Reference trajectory array shape mismatch: {details}")
