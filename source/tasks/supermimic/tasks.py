"""K1 task adaptation; all dynamics and policy learning remain upstream."""

from dataclasses import fields

from mjlab.managers import TerminationTermCfg
from mjlab.tasks.registry import load_rl_cfg, register_mjlab_task
from supermimic.tasks.env_cfgs import gencoord_env_cfg, maxcoord_env_cfg

from .native_motion import NativeMotionCommandCfg, motion_end
from .robot import (
    K1_REV1_ACTION_SCALE,
    K1_REV1_FOOT_COLLISION_GEOM_PATTERN,
    get_k1_rev1_cfg,
)

# K1 has one-joint waist/knee chains as well as 2/3-joint chains. Keeping all
# 23 hinges covered exactly once is required by maximal-coordinate analytic IK.
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
TASKS = {
    f"Cyclo-SuperMimic-{rep}-K1-Rev1": rep.lower() for rep in ("GenCoord", "MaxCoord")
}


def k1_env_cfg(representation="gencoord", play=False):
    if representation not in ("gencoord", "maxcoord"):
        raise ValueError(f"Unknown representation: {representation}")
    cfg = (gencoord_env_cfg if representation == "gencoord" else maxcoord_env_cfg)(
        play=play
    )
    cfg.scene.entities = {"robot": get_k1_rev1_cfg()}
    previous_motion = cfg.commands["motion"]
    cfg.commands["motion"] = NativeMotionCommandCfg(
        **{f.name: getattr(previous_motion, f.name) for f in fields(previous_motion)}
    )
    cfg.scene.sensors = ()  # G1-only self-collision sensor is unused by SuperMimic.
    cfg.actions["joint_pos"].scale = K1_REV1_ACTION_SCALE
    cfg.commands["motion"].anchor_body_name = "pelvis"
    cfg.commands["motion"].body_names = TRACKED_BODIES
    cfg.terminations["motion_end"] = TerminationTermCfg(
        func=motion_end, time_out=True, params={"command_name": "motion"}
    )
    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = K1_REV1_FOOT_COLLISION_GEOM_PATTERN
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.viewer.body_name = "torso_link"
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 300
    cfg.scene.num_envs = 1 if play else 4096
    return cfg


def register_tasks():
    import mjlab.tasks  # noqa: F401

    for task_id, representation in TASKS.items():
        register_mjlab_task(
            task_id=task_id,
            env_cfg=k1_env_cfg(representation),
            play_env_cfg=k1_env_cfg(representation, play=True),
            rl_cfg=load_rl_cfg("Mjlab-Tracking-Flat-Unitree-G1"),
        )
