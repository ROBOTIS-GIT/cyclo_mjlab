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

"""Velocity task configuration.

The generic reward and command semantics follow official MJLab 1.2.0. Cyclo
K1-specific observation and penalty overrides live in the robot configuration.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import source.tasks.velocity.mdp as mdp


def make_locomotion_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration."""

  # Observation terms

  actor_observation_terms = {
    "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}, noise=Unoise(n_min=-0.2, n_max=0.2)),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
    "velocity_commands": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"}),
    "gait_phase": ObservationTermCfg(
      func=mdp.gait_phase,
      params={
        "cycle_time_s": 0.6,
        "command_name": "base_velocity",
        "motion_threshold": 0.1,
      },
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_observation_terms = {
    **actor_observation_terms,
    "base_lin_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}, noise=Unoise(n_min=-0.5, n_max=0.5)),
    "foot_height": ObservationTermCfg(func=mdp.foot_height, params={"asset_cfg": SceneEntityCfg("robot", site_names=())}),  # Set per-robot.
    "foot_air_time": ObservationTermCfg(func=mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"}),
    "foot_contact": ObservationTermCfg(func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}),
    "foot_contact_forces": ObservationTermCfg(func=mdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"}),
  }

  observation_groups = {
    "actor": ObservationGroupCfg(terms=actor_observation_terms, concatenate_terms=True, enable_corruption=True, history_length=1),
    "critic": ObservationGroupCfg(terms=critic_observation_terms, concatenate_terms=True, enable_corruption=False, history_length=1),
  }

  # Metric terms

  metric_terms = {
    "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
  }

  # Action terms

  action_terms: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=0.25, use_default_offset=True)
  }

  # Command terms

  command_terms: dict[str, CommandTermCfg] = {
    "base_velocity": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 1.0),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  # Event terms

  event_terms = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
  }

  # Reward terms

  reward_terms = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.2,
      params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.2,
      params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    ),
    "upright": RewardTermCfg(
      func=mdp.flat_orientation,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "default_stand_joint_deviation": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "base_velocity",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=0.0,  # Override per-robot.
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.05,  # Override per-robot.
      params={"sensor_name": "robot/root_angmom"},
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": 0.10,
        "command_name": "base_velocity",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "base_velocity",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "feet_lateral_separation": RewardTermCfg(
      func=mdp.feet_lateral_separation_l2,
      weight=-1.0,
      params={
        "min_width": 0.13,
        "command_name": "base_velocity",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-5e-4,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "base_velocity",
        "command_threshold": 0.05,
      },
    ),
  }

  # Termination terms

  termination_terms = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  # Curriculum terms

  curriculum_terms = {
    "velocity_commands": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "base_velocity",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
        ],
      },
    ),
  }

  # Assemble and return the environment configuration

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
    ),
    observations=observation_groups,
    actions=action_terms,
    commands=command_terms,
    events=event_terms,
    rewards=reward_terms,
    terminations=termination_terms,
    curriculum=curriculum_terms,
    metrics=metric_terms,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
