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
# Authors: Woojin Wie, Kiwoong Park

"""K1 adaptation using the unchanged public Rev.1 Cyclo asset and gains."""

import re

TRACKED_BODIES = (
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


def robot_cfg():
    from source.assets.robots.ai_sapiens.k1_rev1 import get_k1_rev1_cfg

    return tracking_pose(get_k1_rev1_cfg())


def tracking_pose(cfg):
    # Task-local nominal elbow. Independent deepcopy; shared defaults stay intact.
    cfg.init_state.joint_pos[".*_elbow_joint"] = 0.6
    return cfg


def make_robot():
    from source.assets.robots.ai_sapiens.k1_rev1 import K1_REV1_XML_PATH

    return make_k1_spec("k1", K1_REV1_XML_PATH, robot_cfg)


def make_k1_spec(name, xml_path, cfg_factory):
    import mujoco
    from blinkmimic.robot import RobotSpec
    from source.assets.robots.ai_sapiens.k1_rev1 import K1_REV1_ACTION_SCALE

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    names = tuple(
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    )
    scales = {
        n: next(v for pat, v in K1_REV1_ACTION_SCALE.items() if re.fullmatch(pat, n))
        for n in names
    }
    return RobotSpec(
        name=name,
        mjcf_path=str(xml_path),
        joint_names=names,
        tracked_bodies=TRACKED_BODIES,
        default_joint_pos=cfg_factory().init_state.joint_pos,
        action_scale=scales,
        com_body="torso_link",
        com_range=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)),
        robot_cfg=cfg_factory,
        motion_dir="",
        motion_glob="*.npz",
        load_motion=None,
        njmax=300,
        nconmax=128,
        foot_contact_z=0.074,
    )
