"""Translate a Cyclo RobotSpec to the portable X-Mimic simulated scene."""

import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import mujoco
from mjlab.scene import Scene

from .config import robot_contract


def export_scene(robot, contact_bodies, directory):
    from blinkmimic.env import make_scene_cfg, make_sim
    from blinkmimic.kinematics import Kinematics

    scene = Scene(make_scene_cfg(robot, 1), device="cpu")
    model = scene.compile()
    make_sim(robot).mujoco.apply(model)
    scene.write(directory)
    # Scene.write handles in-memory assets. Copy file-backed assets from the
    # robot's declared MJCF mesh/texture directories into the same bundle.
    source = Path(robot.mjcf_path)
    compiler = ET.parse(source).getroot().find("compiler")
    settings = compiler.attrib if compiler is not None else {}
    asset_dir = directory / "assets"
    for element in ET.parse(directory / "scene.xml").getroot().iter():
        filename = element.get("file")
        if not filename:
            continue
        path = asset_dir / filename
        if not path.resolve().is_relative_to(asset_dir.resolve()):
            raise ValueError(f"Nonportable scene asset: {filename}")
        if path.is_file():
            continue
        kind = "texturedir" if element.tag == "texture" else "meshdir"
        original = (
            source.parent / settings.get(kind, settings.get("assetdir", "")) / filename
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, path)
    # The trainer consumes this description under its own MuJoCo version.
    mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))

    kin = Kinematics(robot, "cpu")

    def names(kind, size):
        return {
            mujoco.mj_id2name(model, kind, index).removeprefix("robot/"): index
            for index in range(size)
            if mujoco.mj_id2name(model, kind, index)
        }

    joint_ids = names(mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    body_ids = names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    joints = [joint_ids[name] for name in robot.joint_names]
    (free,) = [
        i for i, kind in enumerate(model.jnt_type) if kind == mujoco.mjtJoint.mjJNT_FREE
    ]
    actuators = []
    for joint in joints:
        (actuator,) = [
            i
            for i in range(model.nu)
            if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
            and model.actuator_trnid[i, 0] == joint
        ]
        actuators.append(actuator)
    root = model.jnt_bodyid[free]
    options = (
        "timestep",
        "integrator",
        "cone",
        "solver",
        "iterations",
        "tolerance",
        "ls_iterations",
        "ls_tolerance",
        "ccd_iterations",
        "impratio",
        "jacobian",
        "disableflags",
        "enableflags",
    )
    exact = (
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "dof_armature",
        "body_mass",
        "body_inertia",
        "body_ipos",
        "dof_damping",
        "dof_frictionloss",
        "geom_friction",
    )
    meta = {
        "robot": robot_contract(robot),
        "joint_names": list(robot.joint_names),
        "tracked_bodies": list(robot.tracked_bodies),
        "contact_bodies": list(contact_bodies),
        "qpos_adr": model.jnt_qposadr[joints].tolist(),
        "dof_adr": model.jnt_dofadr[joints].tolist(),
        "free_qpos_adr": int(model.jnt_qposadr[free]),
        "free_dof_adr": int(model.jnt_dofadr[free]),
        "actuator_of_joint": actuators,
        "robot_geom_ids": [
            i
            for i, body in enumerate(model.geom_bodyid)
            if model.body_rootid[body] == root
        ],
        "com_body_id": body_ids[robot.com_body],
        "com_range": robot.com_range,
        "default_joint_pos": robot.default_joint_pos_list(),
        "action_scale": robot.action_scale_list(),
        "q_lo": kin.q_lo.tolist(),
        "q_hi": kin.q_hi.tolist(),
        "kin_parent": kin.parent,
        "kin_joint_idx": kin.joint_idx,
        "kin_axes": kin.axes.tolist(),
        "kin_off_p": kin.off_p.tolist(),
        "kin_off_R": kin.off_R.tolist(),
        "kin_tracked_idx": kin.tracked_idx,
        "dr": asdict(robot.dr),
        "sim": {
            **{key: getattr(model.opt, key) for key in options},
            "gravity": model.opt.gravity.tolist(),
            "njmax": robot.njmax,
            "nconmax": robot.nconmax,
        },
        "exact": {key: getattr(model, key).tolist() for key in exact},
    }
    (directory / "meta.json").write_text(json.dumps(meta, allow_nan=False) + "\n")
    return meta
