# Copyright 2025 ROBOTIS CO., LTD.
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

"""K1 Rev.1 humanoid robot asset configuration."""

import copy
import math
from functools import partial
from pathlib import Path

import mujoco

from source import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


K1_REV1_XML_PATH: Path = (SRC_PATH / "assets" / "robots" / "robotis_k1" / "xmls" / "k1.xml")
assert K1_REV1_XML_PATH.exists()

NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0

ARMATURE_QC060_200_R020_RE = 0.00564892
ARMATURE_QC080_240_R020_RE = 0.01936542
MAX_TORQUE_QC060_200_R020_RE = 47.277
MAX_TORQUE_QC080_240_R020_RE = 96.864
STIFFNESS_QC060_200_R020_RE = ARMATURE_QC060_200_R020_RE * NATURAL_FREQ**2
STIFFNESS_QC080_240_R020_RE = ARMATURE_QC080_240_R020_RE * NATURAL_FREQ**2

DAMPING_QC060_200_R020_RE = (2.0 * DAMPING_RATIO * ARMATURE_QC060_200_R020_RE * NATURAL_FREQ)
DAMPING_QC080_240_R020_RE = (2.0 * DAMPING_RATIO * ARMATURE_QC080_240_R020_RE * NATURAL_FREQ)

K1_REV1_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.78),
  joint_pos={
    ".*_hip_pitch_joint": -0.3,
    ".*_knee_joint": 0.63,
    ".*_ankle_pitch_joint": -0.33,
    ".*_elbow_joint": 0.95,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)


K1_REV1_LEGS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
  ),
  stiffness=STIFFNESS_QC080_240_R020_RE,
  damping=DAMPING_QC080_240_R020_RE,
  effort_limit=MAX_TORQUE_QC080_240_R020_RE,
  armature=ARMATURE_QC080_240_R020_RE,
)

K1_REV1_ANKLE_PITCH_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=STIFFNESS_QC080_240_R020_RE,
  damping=DAMPING_QC080_240_R020_RE,
  effort_limit=MAX_TORQUE_QC080_240_R020_RE,
  armature=ARMATURE_QC080_240_R020_RE,
)

K1_REV1_ANKLE_ROLL_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=STIFFNESS_QC060_200_R020_RE,
  damping=DAMPING_QC060_200_R020_RE,
  effort_limit=MAX_TORQUE_QC060_200_R020_RE,
  armature=ARMATURE_QC060_200_R020_RE,
)

K1_REV1_WAIST_YAW_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=STIFFNESS_QC080_240_R020_RE,
  damping=DAMPING_QC080_240_R020_RE,
  effort_limit=MAX_TORQUE_QC080_240_R020_RE,
  armature=ARMATURE_QC080_240_R020_RE,
)

K1_REV1_ARMS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_QC060_200_R020_RE,
  damping=DAMPING_QC060_200_R020_RE,
  effort_limit=MAX_TORQUE_QC060_200_R020_RE,
  armature=ARMATURE_QC060_200_R020_RE,
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  condim={
    r"^(left|right)_foot[1-7]_collision$": 3,
    ".*_collision.*": 1,
  },
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  contype=0,
  conaffinity=1,
  condim={
    r"^(left|right)_foot[1-7]_collision$": 3,
    ".*_collision.*": 1,
  },
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)


K1_REV1_ARTICULATION_CFG = EntityArticulationInfoCfg(
  actuators=(
    K1_REV1_LEGS_ACTUATOR_CFG,
    K1_REV1_ANKLE_PITCH_ACTUATOR_CFG,
    K1_REV1_ANKLE_ROLL_ACTUATOR_CFG,
    K1_REV1_WAIST_YAW_ACTUATOR_CFG,
    K1_REV1_ARMS_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


K1_REV1_CFG = EntityCfg(
  init_state=K1_REV1_INIT_STATE,
  collisions=(FULL_COLLISION,),
  spec_fn=partial(mujoco.MjSpec.from_file, str(K1_REV1_XML_PATH)),
  articulation=K1_REV1_ARTICULATION_CFG,
)


def get_k1_rev1_cfg() -> EntityCfg:
  """Return a fresh ROBOTIS K1 Rev.1 robot configuration."""
  return copy.deepcopy(K1_REV1_CFG)


def _action_scale_from_actuators(
  cfg: EntityArticulationInfoCfg,
) -> dict[str, float]:
  scale = {}
  for actuator in cfg.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    assert actuator.effort_limit is not None
    for name in actuator.target_names_expr:
      scale[name] = 0.25 * actuator.effort_limit / actuator.stiffness
  return scale


K1_REV1_ACTION_SCALE = _action_scale_from_actuators(K1_REV1_ARTICULATION_CFG)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_k1_rev1_cfg())
  viewer.launch(robot.spec.compile())
