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

"""Deployment configuration export for BlinkMimic."""

import math
import re
from dataclasses import dataclass
from pathlib import Path

from source.utils.sim2real_yaml import dump_sim2real_yaml


@dataclass(frozen=True)
class ObservationField:
    name: str
    width: int
    scale: float = 1.0


def observation_fields(num_joints):
    return (
        ObservationField("joint_pos_rel", num_joints),
        ObservationField("joint_vel", num_joints),
        ObservationField("base_ang_vel", 3),
        ObservationField("projected_gravity", 3),
        ObservationField("last_action", num_joints),
        ObservationField("reference_orientation", 6),
    )


def goal_fields(num_joints):
    return (
        ObservationField("future_root_displacement", 3),
        ObservationField("future_root_orientation", 6),
        ObservationField("future_root_linear_velocity", 3),
        ObservationField("future_root_angular_velocity", 3),
        ObservationField("future_joint_pos", num_joints),
        ObservationField("future_joint_vel", num_joints, 0.1),
    )


def native_dimensions(num_joints):
    from blinkmimic.mdp import obs_dim
    from blinkmimic.motion import GOAL_Q_OFFSET

    return obs_dim(num_joints), GOAL_Q_OFFSET + 2 * num_joints


def validate_field_dimensions(num_joints):
    """Fail export when the upstream layout no longer fits the named fields."""
    expected = native_dimensions(num_joints)
    actual = tuple(sum(field.width for field in fields) for fields in (
        observation_fields(num_joints), goal_fields(num_joints),
    ))
    if actual != expected:
        raise ValueError(f"BlinkMimic field dimensions {actual} differ from upstream {expected}")


def joint_properties(robot):
    """Resolve nominal gains from the same actuator configuration used in training."""
    articulation = robot.robot_cfg().articulation
    if articulation is None:
        raise ValueError("BlinkMimic export requires articulated position actuators")
    properties = {}
    for name, default in zip(robot.joint_names, robot.default_joint_pos_list(), strict=True):
        matches = [
            actuator for actuator in articulation.actuators
            if any(re.fullmatch(pattern, name) for pattern in actuator.target_names_expr)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one actuator for {name}, found {len(matches)}")
        actuator = matches[0]
        values = [float(default), float(actuator.stiffness), float(actuator.damping)]
        if not all(math.isfinite(value) for value in values) or min(values[1:]) < 0:
            raise ValueError(f"Invalid nominal joint properties: {name}")
        properties[name] = dict(
            default_position=values[0], stiffness=values[1], damping=values[2],
            position_limit=None,
        )
    return properties


def observation_config(history, future, num_joints):
    """Describe the native actor input before ONNX normalization."""
    validate_field_dimensions(num_joints)

    def terms(fields, history_length=None):
        return {field.name: dict(scale=[field.scale] * field.width, clip=None,
                                **({"history_length": history_length} if history_length else {}))
                for field in fields}

    sensor = terms(observation_fields(num_joints), history)
    goal = terms(goal_fields(num_joints))
    return dict(
        history=dict(layout="frame_major", terms=sensor),
        motion_goal=dict(frames=future, stride=1, start_offset=1,
                         groups=[dict(layout="frame_major", terms=goal)]),
    )


def export_sim2real_cfg(output_dir, training_args, robot):
    """Write the deployment settings from the trainer's resolved arguments."""
    from blinkmimic.motion import CONTROL_DT

    history = training_args.hist
    future = training_args.goal_frames
    if any(type(value) is not int or value < 1 for value in (history, future)):
        raise ValueError("History length and future frames must be positive integers")
    if type(training_args.pi_ref_skip) is not bool:
        raise ValueError("Reference skip must be a boolean")
    cfg = dict(
        policy_joints=list(robot.joint_names),
        step_dt=CONTROL_DT,
        joint_properties=joint_properties(robot),
        commands={},
        actions={"joint_pos": dict(clip=None, scale=robot.action_scale_list(), offset="default_position")},
        observations={"blinkmimic": dict(
            schema="cyclo.blinkmimic.csv.v1",
            **observation_config(history, future, robot.num_joints),
            input_name="observations",
            normalization_in_onnx=True,
            reference_skip_in_onnx=training_args.pi_ref_skip,
            imu_frame="pelvis",
            motion_fps=1.0 / CONTROL_DT,
            reference_velocity="central_difference",
        )},
    )
    path = Path(output_dir) / "params" / "sim2real.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_sim2real_yaml(cfg), encoding="utf-8")
    return path
