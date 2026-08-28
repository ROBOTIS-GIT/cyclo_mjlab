"""Cyclo K1 Rev.1 flat velocity environment configuration."""

from source.assets.robots import K1_REV1_ACTION_SCALE, get_k1_rev1_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

import source.tasks.velocity.mdp as mdp
from source.tasks.velocity.velocity_env_cfg import (
  make_locomotion_velocity_env_cfg,
)


K1_REV1_FOOT_SITE_NAMES = ("left_foot", "right_foot")
K1_REV1_FOOT_COLLISION_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


def cyclo_k1_rev1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Cyclo K1 Rev.1 flat-terrain velocity configuration."""
  env_cfg = make_locomotion_velocity_env_cfg()

  env_cfg.sim.njmax = 300
  env_cfg.sim.mujoco.ccd_iterations = 50
  env_cfg.sim.contact_sensor_maxmatch = 64
  env_cfg.sim.nconmax = None

  env_cfg.scene.entities = {"robot": get_k1_rev1_cfg()}

  feet_contact_sensor_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_contact_sensor_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + (
    feet_contact_sensor_cfg,
    self_contact_sensor_cfg,
  )

  joint_position_action_cfg = env_cfg.actions["joint_pos"]
  assert isinstance(joint_position_action_cfg, JointPositionActionCfg)
  joint_position_action_cfg.scale = K1_REV1_ACTION_SCALE

  env_cfg.viewer.body_name = "torso_link"

  velocity_command_cfg = env_cfg.commands["base_velocity"]
  assert isinstance(velocity_command_cfg, UniformVelocityCommandCfg)
  velocity_command_cfg.viz.z_offset = 1.1

  env_cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = K1_REV1_FOOT_SITE_NAMES

  env_cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = K1_REV1_FOOT_COLLISION_GEOM_NAMES
  env_cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Keep the official MJLab task/contact rewards and add Cyclo Lab's K1
  # non-contact penalties. Robot-specific pose shaping is expressed by the
  # joint-deviation terms below instead of a borrowed posture-tolerance table.
  for reward_name in (
    "body_ang_vel",
    "angular_momentum",
    "pose",
  ):
    env_cfg.rewards.pop(reward_name, None)

  leg_joints = (".*_hip_.*", ".*_knee_joint", ".*_ankle_.*")
  env_cfg.rewards.update(
    {
      "termination_penalty": RewardTermCfg(
        func=mdp.is_terminated, weight=-200.0
      ),
      "alternating_support": RewardTermCfg(
        func=mdp.alternating_support_tracking,
        weight=0.5,
        params={
          "cycle_time_s": 0.6,
          "phase_offsets": (0.0, 0.5),
          "stance_fraction": 0.56,
          "motion_threshold": 0.1,
          "command_name": "base_velocity",
          "contact_sensor_name": feet_contact_sensor_cfg.name,
        },
      ),
      "base_height": RewardTermCfg(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 0.80},
      ),
      "lin_vel_z_l2": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0),
      "ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.1),
      "flat_orientation_l2": RewardTermCfg(
        func=mdp.flat_orientation_l2, weight=-5.0
      ),
      "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
      "dof_acc_l2": RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-5.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=leg_joints)},
      ),
      "dof_torques_l2": RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-3.0e-6,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot", joint_names=leg_joints, actuator_names=leg_joints
          )
        },
      ),
      "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
      "joint_deviation_hip": RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot", joint_names=(".*_hip_yaw_joint", ".*_hip_roll_joint")
          )
        },
      ),
      "joint_deviation_arms": RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=(
              ".*_shoulder_roll_joint",
              ".*_shoulder_yaw_joint",
              ".*_wrist_roll_joint",
            ),
          )
        },
      ),
      "joint_deviation_shoulder_pitch": RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot", joint_names=".*_shoulder_pitch_joint"
          )
        },
      ),
      "joint_deviation_elbow": RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=".*_elbow_joint")
        },
      ),
      # Cyclo Lab intentionally applies both arm penalties to this joint set.
      "joint_deviation_arm_lateral": RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=(
              ".*_shoulder_roll_joint",
              ".*_shoulder_yaw_joint",
              ".*_wrist_roll_joint",
            ),
          )
        },
      ),
      "joint_deviation_torso": RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names="waist_yaw_joint")
        },
      ),
    }
  )

  env_cfg.rewards["upright"].params["asset_cfg"].body_names = (
    "torso_link",
  )
  env_cfg.rewards["foot_clearance"].params["target_height"] = 0.10
  env_cfg.rewards["foot_clearance"].params[
    "asset_cfg"
  ].site_names = K1_REV1_FOOT_SITE_NAMES
  env_cfg.rewards["foot_slip"].params[
    "asset_cfg"
  ].site_names = K1_REV1_FOOT_SITE_NAMES
  env_cfg.rewards["feet_lateral_separation"].params[
    "asset_cfg"
  ].site_names = K1_REV1_FOOT_SITE_NAMES
  env_cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_contact_sensor_cfg.name, "force_threshold": 10.0},
  )

  if play:
    env_cfg.episode_length_s = int(1e9)
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.events.pop("push_robot", None)
    env_cfg.curriculum = {}
    velocity_command_cfg = env_cfg.commands["base_velocity"]
    assert isinstance(velocity_command_cfg, UniformVelocityCommandCfg)
    velocity_command_cfg.ranges.lin_vel_x = (-0.75, 1.5)
    velocity_command_cfg.ranges.lin_vel_y = (-0.75, 0.75)
    velocity_command_cfg.ranges.ang_vel_z = (-1.0, 1.0)

  return env_cfg
