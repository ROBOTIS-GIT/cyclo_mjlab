"""Cyclo task configuration for the independent X-Mimic learner."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blinkmimic.robot import RobotSpec

from source.tasks.blinkmimic.config.k1_rev1.robot_spec import make_robot


@dataclass(frozen=True)
class XMimicTaskCfg:
    robot_factory: Callable[[], "RobotSpec"]
    contact_bodies: tuple[str, str]


TASKS = {
    "Cyclo-XMimic-K1-Rev1-001": XMimicTaskCfg(
        robot_factory=make_robot,
        contact_bodies=("left_ankle_roll_link", "right_ankle_roll_link"),
    )
}


def robot_contract(robot):
    return {
        "name": robot.name,
        "joint_names": list(robot.joint_names),
        "default_joint_pos": robot.default_joint_pos_list(),
        "action_scale": robot.action_scale_list(),
    }
