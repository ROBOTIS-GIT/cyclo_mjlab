# Copyright 2026 ROBOTIS CO., LTD.
# Copyright 2025, The mjlab Developers
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
# Author: Insu Park
#
# This file includes modifications by ROBOTIS CO., LTD. to code derived
# from mujocolab/mjlab.
#
# Additional notices:
# This module is adapted from HybridRobotics/whole_body_tracking, licensed under
# the MIT License. See THIRD_PARTY_LICENSES.md for details.

"""Reference-trajectory command term for Mimic tasks."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from .reference_trajectory import ReferenceTrajectory

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class ReferenceTrajectoryCommand(CommandTerm):
  """Command term that writes sampled reference states and exposes tracking targets."""

  cfg: ReferenceTrajectoryCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(
    self, cfg: ReferenceTrajectoryCommandCfg, env: ManagerBasedRlEnv
  ):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.asset_name]
    self.robot_anchor_body_id = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.reference_anchor_body_id = self.cfg.body_names.index(
      self.cfg.anchor_body_name
    )
    self.tracked_body_ids = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.reference = ReferenceTrajectory(
      trajectory_file=self.cfg.trajectory_file,
      robot_joint_names=self.robot.joint_names,
      robot_body_names=self.robot.body_names,
      tracked_body_names=self.cfg.body_names,
      device=self.device,
    )
    self.frame_ids = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.aligned_body_position_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.aligned_body_orientation_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.aligned_body_orientation_w[:, :, 0] = 1.0
    self.aligned_body_linear_velocity_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.aligned_body_angular_velocity_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )

    reference_frames_per_policy_step = 1 / env.step_dt
    self.start_bin_count = (
      int(self.reference.num_frames // reference_frames_per_policy_step) + 1
    )
    self.failure_score_by_bin = torch.zeros(
      self.start_bin_count, dtype=torch.float, device=self.device
    )
    self._episode_failures_by_bin = torch.zeros(
      self.start_bin_count, dtype=torch.float, device=self.device
    )
    self.failure_smoothing_kernel = torch.tensor(
      [
        self.cfg.failure_sampling_decay**i
        for i in range(self.cfg.failure_sampling_kernel_size)
      ],
      device=self.device,
    )
    self.failure_smoothing_kernel = (
      self.failure_smoothing_kernel / self.failure_smoothing_kernel.sum()
    )

    self.metrics["error_anchor_pos"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_rot"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_rot"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_joint_pos"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_joint_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["start_sampling_entropy"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["start_sampling_top1_prob"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["start_sampling_top1_bin"] = torch.zeros(
      self.num_envs, device=self.device
    )

    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._ghost_offset = np.array(cfg.viz.ghost_offset, dtype=np.float64)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.reference.joint_position[self.frame_ids]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.reference.joint_velocity[self.frame_ids]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.reference.body_position_w[self.frame_ids]
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.reference.body_orientation_w[self.frame_ids]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.reference.body_linear_velocity_w[self.frame_ids]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.reference.body_angular_velocity_w[self.frame_ids]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.reference.body_position_w[
        self.frame_ids, self.reference_anchor_body_id
      ]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.reference.body_orientation_w[
      self.frame_ids, self.reference_anchor_body_id
    ]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.reference.body_linear_velocity_w[
      self.frame_ids, self.reference_anchor_body_id
    ]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.reference.body_angular_velocity_w[
      self.frame_ids, self.reference_anchor_body_id
    ]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.tracked_body_ids]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.tracked_body_ids]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_com_lin_vel_w[:, self.tracked_body_ids]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.tracked_body_ids]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_id]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_id]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_com_lin_vel_w[:, self.robot_anchor_body_id]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_id]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )
    self.metrics["error_body_pos"] = torch.norm(
      self.aligned_body_position_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.aligned_body_orientation_w, self.robot_body_quat_w
    ).mean(dim=-1)
    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _sample_start_frames(self, env_ids: Sequence[int]):
    if self.cfg.start_from_zero:
      self.frame_ids[env_ids] = 0
      return

    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_start_bin = torch.clamp(
        (self.frame_ids * self.start_bin_count)
        // max(self.reference.num_frames, 1),
        0,
        self.start_bin_count - 1,
      )
      failed_start_bins = current_start_bin[env_ids][episode_failed]
      self._episode_failures_by_bin[:] = torch.bincount(
        failed_start_bins, minlength=self.start_bin_count
      )

    start_bin_probabilities = (
      self.failure_score_by_bin
      + self.cfg.failure_sampling_uniform_ratio / float(self.start_bin_count)
    )
    start_bin_probabilities = torch.nn.functional.pad(
      start_bin_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.failure_sampling_kernel_size - 1),
      mode="replicate",
    )
    start_bin_probabilities = torch.nn.functional.conv1d(
      start_bin_probabilities, self.failure_smoothing_kernel.view(1, 1, -1)
    ).view(-1)
    start_bin_probabilities = (
      start_bin_probabilities / start_bin_probabilities.sum()
    )

    sampled_start_bins = torch.multinomial(
      start_bin_probabilities, len(env_ids), replacement=True
    )
    self.frame_ids[env_ids] = (
      (
        sampled_start_bins
        + sample_uniform(
          0.0, 1.0, (len(env_ids),), device=self.device
        )
      )
      / self.start_bin_count
      * (self.reference.num_frames - 1)
    ).long()

    entropy = -(
      start_bin_probabilities * (start_bin_probabilities + 1e-12).log()
    ).sum()
    entropy_norm = (
      entropy / math.log(self.start_bin_count)
      if self.start_bin_count > 1
      else 1.0
    )
    top1_prob, top1_bin = start_bin_probabilities.max(dim=0)
    self.metrics["start_sampling_entropy"][:] = entropy_norm
    self.metrics["start_sampling_top1_prob"][:] = top1_prob
    self.metrics["start_sampling_top1_bin"][:] = (
      top1_bin.float() / self.start_bin_count
    )

  def _resample_command(self, env_ids: Sequence[int]):
    if len(env_ids) == 0:
      return
    self._sample_start_frames(env_ids)

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.reset_pose_noise.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

    range_list = [
      self.cfg.reset_velocity_noise.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()
    joint_pos += sample_uniform(
      *self.cfg.reset_joint_position_noise,
      joint_pos.shape,
      joint_pos.device,
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids],
      soft_joint_pos_limits[:, :, 0],
      soft_joint_pos_limits[:, :, 1],
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )
    self.robot.write_root_link_pose_to_sim(
      torch.cat([root_pos[env_ids], root_ori[env_ids]], dim=-1),
      env_ids=env_ids,
    )
    root_body_id = self.robot.indexing.root_body_id
    com_offset_b = self.robot.data.model.body_ipos[:, root_body_id]
    if com_offset_b.ndim == 3:
      com_offset_b = com_offset_b.squeeze(1)
    com_offset_w = quat_apply(root_ori[env_ids], com_offset_b[env_ids])
    root_link_lin_vel = root_lin_vel[env_ids] - torch.cross(
      root_ang_vel[env_ids], com_offset_w, dim=-1
    )
    self.robot.write_root_link_velocity_to_sim(
      torch.cat([root_link_lin_vel, root_ang_vel[env_ids]], dim=-1),
      env_ids=env_ids,
    )
    self.robot.clear_state(env_ids=env_ids)

  def _update_command(self):
    self.frame_ids += 1
    env_ids = torch.where(self.frame_ids >= self.reference.num_frames)[0]
    self._resample_command(env_ids)

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.aligned_body_orientation_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.aligned_body_position_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

    anchor_lin_vel_w_repeat = self.anchor_lin_vel_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_ang_vel_w_repeat = self.anchor_ang_vel_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_lin_vel_w_repeat = self.robot_anchor_lin_vel_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_ang_vel_w_repeat = self.robot_anchor_ang_vel_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    self.aligned_body_linear_velocity_w = (
      robot_anchor_lin_vel_w_repeat
      + quat_apply(delta_ori_w, self.body_lin_vel_w - anchor_lin_vel_w_repeat)
    )
    self.aligned_body_angular_velocity_w = (
      robot_anchor_ang_vel_w_repeat
      + quat_apply(delta_ori_w, self.body_ang_vel_w - anchor_ang_vel_w_repeat)
    )

    self.failure_score_by_bin = (
      self.cfg.failure_sampling_ema_alpha * self._episode_failures_by_bin
      + (1 - self.cfg.failure_sampling_ema_alpha) * self.failure_score_by_bin
    )
    self._episode_failures_by_bin.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw a ghost robot or frames using MJLab's debug visualizer."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.asset_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = (
          self.body_pos_w[batch, 0].cpu().numpy() + self._ghost_offset
        )
        qpos[free_joint_q_adr[3:7]] = (
          self.body_quat_w[batch, 0].cpu().numpy()
        )
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()
        visualizer.add_ghost_mesh(
          qpos, model=self._ghost_model, label=f"ghost_{batch}"
        )

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.aligned_body_position_w[batch].cpu().numpy()
        desired_body_rotm = matrix_from_quat(
          self.aligned_body_orientation_w[batch]
        ).cpu().numpy()
        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_rotm = matrix_from_quat(
          self.robot_body_quat_w[batch]
        ).cpu().numpy()

        for index, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[index],
            rotation_matrix=desired_body_rotm[index],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[index],
            rotation_matrix=current_body_rotm[index],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        visualizer.add_frame(
          position=self.anchor_pos_w[batch].cpu().numpy(),
          rotation_matrix=matrix_from_quat(
            self.anchor_quat_w[batch]
          ).cpu().numpy(),
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )
        visualizer.add_frame(
          position=self.robot_anchor_pos_w[batch].cpu().numpy(),
          rotation_matrix=matrix_from_quat(
            self.robot_anchor_quat_w[batch]
          ).cpu().numpy(),
          scale=0.15,
          label=f"current_anchor_{batch}",
        )


@dataclass(kw_only=True)
class ReferenceTrajectoryCommandCfg(CommandTermCfg):
  """Configuration for reference trajectory sampling."""

  asset_name: str
  trajectory_file: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  reset_pose_noise: dict[str, tuple[float, float]] = field(default_factory=dict)
  reset_velocity_noise: dict[str, tuple[float, float]] = field(default_factory=dict)
  reset_joint_position_noise: tuple[float, float] = (-0.52, 0.52)
  failure_sampling_kernel_size: int = 1
  failure_sampling_decay: float = 0.8
  failure_sampling_uniform_ratio: float = 0.1
  failure_sampling_ema_alpha: float = 0.001
  start_from_zero: bool = False

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)
    ghost_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> ReferenceTrajectoryCommand:
    return ReferenceTrajectoryCommand(self, env)
