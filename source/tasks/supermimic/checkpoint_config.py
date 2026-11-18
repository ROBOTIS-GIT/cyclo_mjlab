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
# Author: Kiwoong Park

"""
Versioned local inference bundles; never deserialize Python configuration code.
"""

import copy
import dataclasses
import enum
import hashlib
import math
from functools import partial
from pathlib import Path

import mjlab
import numpy as np
import tyro
from mjlab.tasks.registry import load_env_cfg
from supermimic.runners.run_config import load_runner_cfg

from .checkpoint_archive import load_checkpoint
from .motion import model_joint_names
from source.assets.robots.robotis_k1.k1_rev1 import K1_REV1_XML_PATH

SCHEMA_VERSION = 1


def validate_control_rate(env_cfg):
    timestep = env_cfg.sim.mujoco.timestep
    decimation = env_cfg.decimation
    step_dt = timestep * decimation
    if (
        timestep <= 0
        or decimation <= 0
        or not math.isclose(step_dt, 0.02, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError(
            "50 Hz K1 references require env.sim.mujoco.timestep *"
            f" env.decimation == 0.02 seconds; got {timestep} * {decimation} ="
            f" {step_dt}. Use compensating overrides that keep the control"
            " rate at 50 Hz."
        )


def json_value(value):
    if dataclasses.is_dataclass(value):
        return {
            f.name: json_value(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, partial):
        return {
            "partial": json_value(value.func),
            "args": json_value(value.args),
            "kwargs": json_value(value.keywords),
        }
    if callable(value):
        return {"callable": value.__module__ + "." + value.__qualname__}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": str(value)}
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(
        f"Unsupported configuration value: {type(value).__name__}"
    )


def environment_options(arguments):
    """Retain only declarative tyro environment overrides and their values."""
    result, keep = [], False
    for token in arguments:
        if token.startswith("runner.") and ":" in token:
            keep = False
        if token.startswith("--"):
            keep = token.startswith("--env.")
        if keep:
            result.append(token)
    return result


def environment_contract(cfg):
    cfg = copy.deepcopy(cfg)
    # Playback chooses its local input and one environment; all physical,
    # action, observation and substep settings remain part of the contract.
    cfg.scene.num_envs = 1
    cfg.commands["motion"].motion_file = "<validated-local-reference>"
    return json_value(cfg)


def robot_contract(cfg):
    from mjlab.entity import Entity
    from mjlab.envs.mdp.actions import JointPositionActionCfg

    if (
        set(cfg.actions) != {"joint_pos"}
        or type(cfg.actions["joint_pos"]) is not JointPositionActionCfg
    ):
        raise ValueError("K1 bundles require the joint_pos action contract")
    action = cfg.actions["joint_pos"]
    if action.entity_name != "robot" or action.clip is not None:
        raise ValueError(
            "SuperMimic requires robot joint-position actions without target"
            " clipping"
        )
    entity = Entity(cfg.scene.entities["robot"])
    model = entity.spec.compile()
    _, action_names = entity.find_joints_by_actuator_names(
        action.actuator_names
    )
    names = list(model_joint_names(model))
    if action_names != names:
        raise ValueError(
            "SuperMimic action joint order must exactly match all native"
            " representation joints; reordering and subselection are"
            " unsupported"
        )
    physical = hashlib.sha256()
    for attribute in (
        "body_pos",
        "body_quat",
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "dof_armature",
        "dof_damping",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "mesh_vert",
        "mesh_face",
        "actuator_gear",
        "actuator_gainprm",
        "actuator_biasprm",
    ):
        physical.update(np.asarray(getattr(model, attribute)).tobytes())
    return {
        "name": "K1-Rev1",
        "xml_sha256": (
            hashlib.sha256(K1_REV1_XML_PATH.read_bytes()).hexdigest()
        ),
        "joint_order": names,
        "action_joint_order": action_names,
        "physical_model_sha256": physical.hexdigest(),
        "action_config": json_value(cfg.actions),
    }


def make_manifest(task, cfg, runner_cfg, options):
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task,
        "representation": runner_cfg.representation,
        "actor_obs": runner_cfg.actor_obs,
        "runner_cfg": json_value(runner_cfg),
        "environment_options": options,
        "environment": environment_contract(cfg),
        "robot": robot_contract(cfg),
    }


def validate_manifest(manifest, task, *, for_resume=False):
    from .config import TASKS, check_representation_matches

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported local checkpoint manifest version")
    source_task = manifest.get("task_id")
    if source_task not in TASKS or task not in TASKS or (
        source_task != task and not for_resume
    ):
        raise ValueError(
            "Checkpoint task identity does not match the selected K1 task"
        )
    rcfg = load_runner_cfg(manifest)
    check_representation_matches(source_task, rcfg.representation)
    check_representation_matches(task, rcfg.representation)
    if (
        manifest.get("representation") != rcfg.representation
        or manifest.get("actor_obs") != rcfg.actor_obs
        or rcfg.actor_obs not in ("rep_state", "proprio")
    ):
        raise ValueError(
            "Checkpoint representation/actor observation contract is"
            " inconsistent"
        )
    options = manifest.get("environment_options")
    if not isinstance(options, list) or not all(
        isinstance(v, str) for v in options
    ):
        raise ValueError("Invalid environment override list")
    if environment_options(options) != options:
        raise ValueError(
            "Checkpoint permits only declarative --env.* overrides"
        )
    defaults = load_env_cfg(task)
    cfg = tyro.cli(
        type(defaults),
        args=[
            (
                "--" + token[len("--env."):]
                if token.startswith("--env.")
                else token
            )
            for token in options
        ],
        default=defaults,
        config=mjlab.TYRO_FLAGS,
    )
    cfg.commands["motion"].sampling_mode = (
        "external" if rcfg.sampling_mode == "return" else rcfg.sampling_mode
    )
    validate_control_rate(cfg)
    if environment_contract(cfg) != manifest.get("environment"):
        raise ValueError(
            "Checkpoint environment differs from this checkout or saved"
            " overrides"
        )
    if robot_contract(cfg) != manifest.get("robot"):
        raise ValueError(
            "Checkpoint robot/action/joint-order contract differs from this"
            " checkout"
        )
    cfg.scene.num_envs = 1
    return cfg, rcfg


def read_bundle(directory, task):
    checkpoint = load_checkpoint(directory)
    return validate_manifest(checkpoint.manifest, task)
