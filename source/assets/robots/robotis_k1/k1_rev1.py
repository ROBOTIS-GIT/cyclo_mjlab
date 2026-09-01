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
from pathlib import Path

import mujoco

from source import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


K1_REV1_XML_PATH: Path = (
  SRC_PATH.parent
  / "third_party"
  / "ai_sapiens"
  / "ai_sapiens_description"
  / "mujoco"
  / "k1"
  / "k1.xml"
)
assert K1_REV1_XML_PATH.is_file()

K1_REV1_FOOT_SITE_NAMES = ("left_foot", "right_foot")
K1_REV1_FOOT_COLLISION_GEOM_NAMES = tuple(
  f"{side}_ankle_roll_link_collision_{index}"
  for side in ("left", "right")
  for index in range(9)
)
K1_REV1_FOOT_COLLISION_GEOM_PATTERN = (
  r"^(left|right)_ankle_roll_link_collision_[0-8]$"
)

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
    K1_REV1_FOOT_COLLISION_GEOM_PATTERN: 3,
    ".*_collision.*": 1,
  },
  priority={K1_REV1_FOOT_COLLISION_GEOM_PATTERN: 1},
  friction={K1_REV1_FOOT_COLLISION_GEOM_PATTERN: (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  contype=0,
  conaffinity=1,
  condim={
    K1_REV1_FOOT_COLLISION_GEOM_PATTERN: 3,
    ".*_collision.*": 1,
  },
  priority={K1_REV1_FOOT_COLLISION_GEOM_PATTERN: 1},
  friction={K1_REV1_FOOT_COLLISION_GEOM_PATTERN: (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(K1_REV1_FOOT_COLLISION_GEOM_PATTERN,),
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


def _load_k1_rev1_spec() -> mujoco.MjSpec:
  """Load the upstream K1 MJCF and add MJLab runtime-only elements."""
  spec = mujoco.MjSpec.from_file(str(K1_REV1_XML_PATH))

  # MJLab supplies position actuators below. Drop the upstream torque motors
  # so the robot still exposes exactly the 23 policy-controlled actuators.
  for actuator in tuple(spec.actuators):
    spec.delete(actuator)

  pelvis = spec.body("pelvis")
  pelvis.add_site(
    name="imu",
    pos=(0.0, 0.0, 0.0),
    size=(0.01, 0.01, 0.01),
    group=5,
  )
  for side in ("left", "right"):
    spec.body(f"{side}_ankle_roll_link").add_site(
      name=f"{side}_foot",
      pos=(0.0325, 0.0, -0.0635),
      size=(0.01, 0.01, 0.01),
      group=5,
    )

  for name, sensor_type in (
    ("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
    ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
    ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
  ):
    spec.add_sensor(
      name=name,
      type=sensor_type,
      objtype=mujoco.mjtObj.mjOBJ_SITE,
      objname="imu",
    )
  spec.add_sensor(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="pelvis",
  )

  return spec


K1_REV1_CFG = EntityCfg(
  init_state=K1_REV1_INIT_STATE,
  collisions=(FULL_COLLISION,),
  spec_fn=_load_k1_rev1_spec,
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
