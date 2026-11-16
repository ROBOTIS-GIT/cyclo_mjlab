"""K1 adaptation using the unchanged public Rev.1 Cyclo asset and gains."""

import re

import mujoco
from blinkmimic.robot import RobotSpec

from source.assets.robots.robotis_k1.k1_rev1 import (
    K1_REV1_ACTION_SCALE,
    K1_REV1_XML_PATH,
    get_k1_rev1_cfg,
)

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
    cfg = get_k1_rev1_cfg()
    # Recorded K-POP producer nominal pose. Independent deepcopy; shared defaults stay intact.
    cfg.init_state.joint_pos[".*_elbow_joint"] = 0.6
    return cfg


def make_robot():
    model = mujoco.MjModel.from_xml_path(str(K1_REV1_XML_PATH))
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
        name="k1",
        mjcf_path=str(K1_REV1_XML_PATH),
        joint_names=names,
        tracked_bodies=TRACKED_BODIES,
        default_joint_pos=robot_cfg().init_state.joint_pos,
        action_scale=scales,
        com_body="torso_link",
        com_range=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)),
        robot_cfg=robot_cfg,
        motion_dir="",
        motion_glob="*.npz",
        load_motion=None,
        njmax=300,
        nconmax=128,
    )
