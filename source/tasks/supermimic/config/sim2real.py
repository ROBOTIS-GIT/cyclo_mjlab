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

"""SuperMimic deployment settings from the resolved training configuration."""

import math
import re
from pathlib import Path

import torch
from mjlab.entity import Entity
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.utils.string import resolve_expr

from source.utils.sim2real_yaml import dump_sim2real_yaml


def resolve_action_settings(cfg, joint_names, *, model=None):
    """
    Resolve the same native pose and affine action values without a simulator.
    """
    if model is None:
        model = Entity(cfg.scene.entities["robot"]).spec.compile()
    qadr = [model.joint(name).qposadr[0] for name in joint_names]
    q0 = torch.tensor(model.key("init_state").qpos[qadr], dtype=torch.float32)
    qd0 = torch.tensor(
        resolve_expr(cfg.scene.entities["robot"].init_state.joint_vel, joint_names, 0.0),
        dtype=torch.float32,
    )

    def resolve(value, fallback):
        if isinstance(value, (int, float)):
            return torch.full_like(q0, float(value))
        values = torch.full_like(q0, fallback)
        ids, _, matched = resolve_matching_names_values(value, joint_names)
        values[ids] = torch.tensor(matched, dtype=torch.float32)
        return values

    action = cfg.actions["joint_pos"]
    scale = resolve(action.scale, 1.0)
    offset = (
        q0.clone()
        if action.use_default_offset
        else resolve(action.offset, 0.0)
    )
    return dict(
        joint_order=joint_names,
        q0=q0.tolist(),
        qdot0=qd0.tolist(),
        scale=scale.tolist(),
        offset=offset.tolist(),
    )


def joint_properties(env_cfg, action):
    """Nominal PD gains and pose, before per-environment randomization."""
    from mjlab.actuator import BuiltinPositionActuatorCfg, IdealPdActuatorCfg

    articulation = env_cfg.scene.entities["robot"].articulation
    if articulation is None:
        raise ValueError(
            "SuperMimic export requires articulated position actuators"
        )
    properties = {}
    for name, default in zip(action["joint_order"], action["q0"], strict=True):
        matches = [
            actuator
            for actuator in articulation.actuators
            if any(
                re.fullmatch(pattern, name)
                for pattern in actuator.target_names_expr
            )
        ]
        if len(matches) != 1 or not isinstance(
            matches[0], (BuiltinPositionActuatorCfg, IdealPdActuatorCfg)
        ):
            raise ValueError(f"Expected one position actuator for {name}")
        actuator = matches[0]
        values = [
            float(default),
            float(actuator.stiffness),
            float(actuator.damping),
        ]
        if (
            not all(math.isfinite(value) for value in values)
            or min(values[1:]) < 0
        ):
            raise ValueError(f"Invalid nominal joint properties: {name}")
        properties[name] = dict(
            default_position=values[0],
            stiffness=values[1],
            damping=values[2],
            position_limit=None,
        )
    return properties


def validate_proprio_observations(terms):
    """
    Preserve the native sensor order and calculations used by the world model.
    """
    from supermimic.tasks.env_cfgs import _sensor_actor_group

    native = _sensor_actor_group().terms
    if list(terms) != list(native):
        raise ValueError(
            "Actor observation order differs from the SuperMimic"
            " representation"
        )
    for name, term in terms.items():
        if (
            term.func is not native[name].func
            or term.params != native[name].params
        ):
            raise ValueError(
                f"Actor observation {name} calculation differs from native"
                " proprio"
            )


def validate_policy_config(cfg, rcfg, task):
    """Reject training observation overrides unsupported by the exporter."""
    from mjlab.tasks.registry import load_env_cfg
    from ..training.checkpoint.manifest import effective_observation_group, json_value

    if rcfg.actor_obs == "rep_state":
        return
    reference = load_env_cfg(task)
    actor = effective_observation_group(json_value(cfg.observations["actor"]))
    if actor != effective_observation_group(json_value(reference.observations["actor"])):
        raise ValueError("Custom actor observations are not supported by this exporter")


def observation_terms(env_cfg, runner_cfg, representation, num_joints):
    """
    Resolve ordered actor fields from training configuration and
    representation.
    """
    if type(runner_cfg.history_length) is not int or runner_cfg.history_length < 1:
        raise ValueError("Invalid history_length")
    if runner_cfg.actor_obs == "proprio":
        configured = env_cfg.observations["actor"].terms
        validate_proprio_observations(configured)
        return [
            dict(name=name, width=width, history=runner_cfg.history_length)
            for name, width in zip(
                configured,
                representation.proprio_term_widths(num_joints),
                strict=True,
            )
        ]
    if runner_cfg.actor_obs != "rep_state" or runner_cfg.history_length != 1:
        raise ValueError("rep_state export requires history_length=1")
    return [
        dict(name="rep_state", width=representation.rep_state_dim, history=1)
    ]


def observation_config(
    env_cfg, runner_cfg, representation, num_joints, encoder
):
    """Describe native history packing and external goal groups."""
    from supermimic.representations import GOAL_FRAME_STRIDE, NUM_GOAL_FRAMES

    if encoder not in ("motion", "velocity"):
        raise ValueError(f"Unsupported SuperMimic encoder: {encoder}")
    aliases = {
        "joint_pos": "joint_pos_rel",
        "joint_vel": "joint_vel_rel",
        "actions": "last_action",
    }
    input_terms = observation_terms(
        env_cfg, runner_cfg, representation, num_joints
    )
    is_proprio = runner_cfg.actor_obs == "proprio"

    terms = {}
    for term in input_terms:
        # rep_state is assembled by the representation, not the actor
        # observation group.
        source = env_cfg.observations["actor"].terms[term["name"]] if is_proprio else None
        scale = (
            source.scale
            if source is not None and source.scale is not None
            else 1.0
        )
        if isinstance(scale, torch.Tensor):
            scale = scale.detach().cpu().tolist()
        scale = (
            [float(scale)] * term["width"]
            if isinstance(scale, (int, float))
            else list(scale)
        )
        if len(scale) != term["width"]:
            raise ValueError(
                f"Observation {term['name']} scale does not match its width"
            )
        terms[aliases.get(term["name"], term["name"])] = dict(
            scale=scale,
            clip=source.clip if source is not None else None,
            history_length=term["history"],
        )

    cfg = dict(history=dict(layout="term_major", terms=terms))

    def group(*fields):
        return dict(
            layout="frame_major",
            terms={
                name: dict(scale=[1.0] * width, clip=None)
                for name, width in fields
            },
        )

    if encoder == "motion":
        cfg["motion_goal"] = dict(
            frames=NUM_GOAL_FRAMES,
            stride=GOAL_FRAME_STRIDE,
            start_offset=1,
            groups=[
                group(
                    ("future_joint_pos", num_joints),
                    ("future_joint_vel", num_joints),
                ),
                group(("future_root_orientation", 6)),
            ],
        )
    else:
        cfg["velocity_goal"] = group(("velocity_command", 3))
    return cfg


def export_sim2real_cfg(
    output_dir,
    env_cfg,
    runner_cfg,
    representation,
    joint_names,
    *,
    encoder="motion",
):
    """
    Write deployment settings directly from resolved training configuration.
    """
    action = resolve_action_settings(env_cfg, joint_names)
    observation = dict(
        schema="cyclo.supermimic.external_goal.v1",
        encoder=encoder,
        representation=runner_cfg.representation,
        actor_obs=runner_cfg.actor_obs,
    )
    if any(value != 0 for value in action["qdot0"]):
        observation["default_joint_velocity"] = action["qdot0"]
    observation.update(
        observation_config(
            env_cfg,
            runner_cfg,
            representation,
            len(joint_names),
            encoder,
        )
    )

    commands = {}
    if encoder == "velocity":
        velocity = runner_cfg.tasks.velocity
        commands["base_velocity"] = {
            "ranges": dict(
                lin_vel_x=list(velocity.lin_vel_x_range),
                lin_vel_y=list(velocity.lin_vel_y_range),
                ang_vel_z=list(velocity.ang_vel_z_range),
            )
        }
    cfg = dict(
        policy_joints=action["joint_order"],
        step_dt=env_cfg.sim.mujoco.timestep * env_cfg.decimation,
        joint_properties=joint_properties(env_cfg, action),
        commands=commands,
        actions={
            "joint_pos": dict(
                clip=None,
                scale=action["scale"],
                offset=action["offset"],
            )
        },
        observations={"supermimic": observation},
    )
    directory = Path(output_dir) / "params"
    if encoder != "motion":
        directory /= encoder
    path = directory / "sim2real.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_sim2real_yaml(cfg), encoding="utf-8")
    return path
