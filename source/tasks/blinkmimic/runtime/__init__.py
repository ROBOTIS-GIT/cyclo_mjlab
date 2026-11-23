"""Load vendored BlinkMimic only on explicit selection, in a separate process."""

import sys
from source import SRC_PATH

ROOT = SRC_PATH.parent
VENDOR = ROOT / "third_party/blinkmimic"
UPSTREAM_REVISION = "d264658d6c9ce61fb52f9f0f8985a369f3ccc409"


def prepare():
    sys.path.insert(0, str(VENDOR / "src"))


def install_robot(robot):
    """Process-local registration; never imports or mutates SuperMimic learner."""
    import types

    import blinkmimic.motion
    import blinkmimic.robot

    from ..reference.motion import ExactMotionLib

    module = types.ModuleType(f"blinkmimic.{robot.name}.{robot.name}_constants")
    module.SPEC = robot
    package = types.ModuleType(f"blinkmimic.{robot.name}")
    package.__path__ = []
    sys.modules[package.__name__] = package
    sys.modules[module.__name__] = module
    blinkmimic.motion.MotionLib = ExactMotionLib
