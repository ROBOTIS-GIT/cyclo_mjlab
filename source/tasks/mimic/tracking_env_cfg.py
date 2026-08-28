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

"""Robot-agnostic mimic tracking environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import source.tasks.mimic.mdp as mdp
from source.tasks.mimic.mdp import ReferenceTrajectoryCommandCfg


# Scene definition

VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def make_mimic_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base motion mimic task configuration."""


  # Observation terms

  actor_terms = {
    "motion_command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "reference_trajectory"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.reference_anchor_orientation_b,
      params={"command_name": "reference_trajectory"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos_rel": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel_rel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "last_action": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    "motion_command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "reference_trajectory"}
    ),
    "reference_anchor_position_b": ObservationTermCfg(
      func=mdp.reference_anchor_position_b,
      params={"command_name": "reference_trajectory"},
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.reference_anchor_orientation_b,
      params={"command_name": "reference_trajectory"},
    ),
    "measured_body_positions_in_anchor_frame": ObservationTermCfg(
      func=mdp.measured_body_positions_in_anchor_frame,
      params={"command_name": "reference_trajectory"},
    ),
    "measured_body_orientations_in_anchor_frame": ObservationTermCfg(
      func=mdp.measured_body_orientations_in_anchor_frame,
      params={"command_name": "reference_trajectory"},
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # Action terms

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    )
  }

  # Command terms

  commands: dict[str, CommandTermCfg] = {
    "reference_trajectory": ReferenceTrajectoryCommandCfg(
      asset_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      reset_pose_noise={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      reset_velocity_noise=VELOCITY_RANGE,
      reset_joint_position_noise=(-0.1, 0.1),
      # Override in robot cfg.
      trajectory_file="",
      anchor_body_name="",
      body_names=(),
    )
  }

  # Event terms

  events: dict[str, EventTermCfg] = {
    "physics_material": EventTermCfg(
      mode="startup",
      func=mdp.randomize_rigid_body_material,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=(".*_collision.*",)
        ),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    "joint_home_offset_noise": EventTermCfg(
      mode="startup",
      func=mdp.apply_home_joint_offset_noise,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "pos_distribution_params": (-0.01, 0.01),
        "operation": "add",
      },
    ),
    "torso_com_offset_noise": EventTermCfg(
      mode="startup",
      func=mdp.apply_link_com_offset_noise,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
        "com_range": {
          "x": (-0.025, 0.025),
          "y": (-0.05, 0.05),
          "z": (-0.05, 0.05),
        },
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE},
    ),
  }

  # Reward terms

  rewards: dict[str, RewardTermCfg] = {
    "reference_anchor_position": RewardTermCfg(
      func=mdp.reference_anchor_position_tracking,
      weight=0.5,
      params={"command_name": "reference_trajectory", "std": 0.3},
    ),
    "reference_anchor_orientation": RewardTermCfg(
      func=mdp.reference_anchor_orientation_tracking,
      weight=0.5,
      params={"command_name": "reference_trajectory", "std": 0.4},
    ),
    "reference_body_position": RewardTermCfg(
      func=mdp.reference_body_position_tracking,
      weight=1.0,
      params={"command_name": "reference_trajectory", "std": 0.3},
    ),
    "reference_body_orientation": RewardTermCfg(
      func=mdp.reference_body_orientation_tracking,
      weight=1.0,
      params={"command_name": "reference_trajectory", "std": 0.4},
    ),
    "reference_body_linear_velocity": RewardTermCfg(
      func=mdp.reference_body_linear_velocity_tracking,
      weight=1.0,
      params={"command_name": "reference_trajectory", "std": 1.0},
    ),
    "reference_body_angular_velocity": RewardTermCfg(
      func=mdp.reference_body_angular_velocity_tracking,
      weight=1.0,
      params={"command_name": "reference_trajectory", "std": 3.14},
    ),
    "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_torque": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-5),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }

  # Termination terms

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "reference_anchor_height": TerminationTermCfg(
      func=mdp.reference_anchor_height_deviation,
      params={"command_name": "reference_trajectory", "threshold": 0.25},
    ),
    "reference_anchor_gravity": TerminationTermCfg(
      func=mdp.reference_anchor_gravity_deviation,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "reference_trajectory",
        "threshold": 0.8,
      },
    ),
    "reference_body_height": TerminationTermCfg(
      func=mdp.reference_body_height_deviation,
      params={
        "command_name": "reference_trajectory",
        "threshold": 0.25,
        "body_names": (),  # Set per-robot.
      },
    ),
    "joint_velocity_limit": TerminationTermCfg(
      func=mdp.joint_velocity_limit_exceeded,
      params={
        "max_velocity": 100.0,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "mujoco_nan": TerminationTermCfg(func=mdp.nan_detection),
  }

  # Assemble and return configuration

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
