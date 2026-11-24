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

import dataclasses
import enum
import math
from functools import partial
from pathlib import Path

import numpy as np
from mjlab.tasks.registry import load_env_cfg
from supermimic.runners.run_config import load_runner_cfg

from .archive import check_motion_contract, load_checkpoint
from ...reference.motion import model_joint_names

SCHEMA_VERSION = 2


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
        # Keep the callable identity used by existing checkpoint manifests.
        module = value.__module__
        prefix = "source.assets.robots.ai_sapiens."
        if module.startswith(prefix):
            module = "source.assets.robots.robotis_k1." + module.removeprefix(prefix)
        name = module + "." + value.__qualname__
        if name in (
            "source.tasks.supermimic.mdp.rewards.heading_velocity_tracking",
            "source.tasks.supermimic.mdp.terminations.motion_end",
        ):
            name = "source.tasks.supermimic.native_motion." + value.__qualname__
        return {"callable": name}
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


def effective_observation_group(group):
    """Ignore term noise disabled by MJLab, while retaining the recorded config."""
    if group is None or group.get("enable_corruption") is not False:
        return group
    terms = {
        name: {key: value for key, value in term.items() if key != "noise"}
        for name, term in group["terms"].items()
    }
    return {**group, "terms": terms}


def check_actor_observations(saved, current):
    saved = effective_observation_group(saved)
    current = effective_observation_group(current)
    if (
        saved is None or current is None or saved != current
        or list(saved["terms"]) != list(current["terms"])
    ):
        raise ValueError("Checkpoint actor observation configuration differs from the target")


def motion_contract(cfg):
    motion = cfg.commands["motion"]
    return {"body_names": list(motion.body_names), "anchor_body_name": motion.anchor_body_name}


def observation_contract(cfg):
    return {
        "commands": {"motion": motion_contract(cfg)},
        "observations": json_value(cfg.observations),
    }


def robot_contract(cfg):
    from source.assets.robots.ai_sapiens.k1_rev2 import load_k1_rev2_spec

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
    rev2 = cfg.scene.entities["robot"].spec_fn is load_k1_rev2_spec
    return {
        "name": "K1-Rev2" if rev2 else "K1-Rev1",
        "joint_order": names,
        "action_joint_order": action_names,
    }


def make_manifest(task, cfg, runner_cfg):
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task,
        "representation": runner_cfg.representation,
        "actor_obs": runner_cfg.actor_obs,
        "runner_cfg": json_value(runner_cfg),
        "environment": observation_contract(cfg),
        "robot": robot_contract(cfg),
    }


def validate_manifest(manifest, task, *, for_resume=False):
    from ...config import TASKS, check_representation_matches

    if manifest.get("schema_version") not in (1, SCHEMA_VERSION):
        raise ValueError("Unsupported local checkpoint manifest version")
    source_task = manifest.get("task_id")
    if task not in TASKS or (not for_resume and source_task != task):
        raise ValueError(
            "Checkpoint task identity does not match the selected K1 task"
        )
    rcfg = load_runner_cfg(manifest)
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
    cfg = load_env_cfg(task)
    cfg.commands["motion"].sampling_mode = (
        "external" if rcfg.sampling_mode == "return" else rcfg.sampling_mode
    )
    validate_control_rate(cfg)
    if not for_resume:
        robot = robot_contract(cfg)
        for key in ("joint_order", "action_joint_order"):
            if manifest.get("robot", {}).get(key) != robot[key]:
                raise ValueError(f"Checkpoint {key} differs from the export task")
        saved_env = manifest["environment"]
        if "commands" not in saved_env:
            raise ValueError("Missing checkpoint motion configuration")
        check_motion_contract(saved_env["commands"]["motion"], motion_contract(cfg))
        if rcfg.actor_obs == "proprio":
            check_actor_observations(
                saved_env["observations"].get("actor"), json_value(cfg.observations["actor"])
            )
    return cfg, rcfg


def read_bundle(directory, task):
    checkpoint = load_checkpoint(directory)
    return validate_manifest(checkpoint.manifest, task)
