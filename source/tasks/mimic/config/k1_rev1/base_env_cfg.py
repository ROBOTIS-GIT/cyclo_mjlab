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

"""K1 Rev.1-specific configuration for Mimic tasks."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

import source.tasks.mimic.mdp as mdp
from source.assets.robots import K1_REV1_ACTION_SCALE, get_k1_rev1_cfg
from source.tasks.mimic.mdp import ReferenceTrajectoryCommandCfg
from source.tasks.mimic.tracking_env_cfg import make_mimic_env_cfg

TRACKED_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_roll_rubber_hand",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_roll_rubber_hand",
)

END_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_roll_rubber_hand",
  "right_wrist_roll_rubber_hand",
)


def k1_rev1_mimic_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the K1 Rev.1 base Mimic environment configuration."""
  cfg = make_mimic_env_cfg()

  cfg.scene.num_envs = 4096
  cfg.scene.entities = {"robot": get_k1_rev1_cfg()}

  cfg.sim.njmax = 300
  cfg.sim.nconmax = 128
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.mujoco.ccd_iterations = 50

  contact_forces_cfg = ContactSensorCfg(
    name="contact_forces",
    primary=ContactMatch(
      mode="body",
      pattern=".*",
      entity="robot",
      exclude=END_BODY_NAMES,
    ),
    fields=("force",),
    reduce="netforce",
    num_slots=1,
    history_length=3,
  )
  cfg.scene.sensors = (contact_forces_cfg,)

  cfg.rewards["undesired_contacts"] = RewardTermCfg(
    func=mdp.undesired_contacts,
    weight=-0.1,
    params={"sensor_name": "contact_forces", "threshold": 1.0},
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = K1_REV1_ACTION_SCALE

  reference_trajectory = cfg.commands["reference_trajectory"]
  assert isinstance(reference_trajectory, ReferenceTrajectoryCommandCfg)
  reference_trajectory.anchor_body_name = "torso_link"
  reference_trajectory.body_names = TRACKED_BODY_NAMES

  cfg.events["torso_com_offset_noise"].params[
    "asset_cfg"
  ].body_names = ("torso_link",)

  cfg.terminations["reference_body_height"].params[
    "body_names"
  ] = END_BODY_NAMES

  cfg.viewer.body_name = "torso_link"
  cfg.episode_length_s = 30.0

  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    reference_trajectory.start_from_zero = True
    cfg.events.pop("push_robot", None)
    cfg.observations["actor"].enable_corruption = False

  return cfg
