"""Load vendored BlinkMimic only on explicit selection, in a separate process."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "third_party/blinkmimic"


def prepare():
    sys.path.insert(0, str(VENDOR / "src"))


def install_robot(robot):
    """Process-local registration; never imports or mutates SuperMimic learner."""
    import types

    import blinkmimic.motion
    import blinkmimic.robot

    from .motion import ExactMotionLib

    module = types.ModuleType(f"blinkmimic.{robot.name}.{robot.name}_constants")
    module.SPEC = robot
    package = types.ModuleType(f"blinkmimic.{robot.name}")
    package.__path__ = []
    sys.modules[package.__name__] = package
    sys.modules[module.__name__] = module
    blinkmimic.motion.MotionLib = ExactMotionLib
