"""Load vendored BlinkMimic only on explicit selection, in a separate process."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "third_party/blinkmimic"


def prepare():
    sys.path.insert(0, str(VENDOR / "src"))


def install_k1():
    """Process-local registration; never imports or mutates SuperMimic learner."""
    import types

    import blinkmimic.motion
    import blinkmimic.robot

    from .motion import ExactMotionLib
    from .robot import make_robot

    module = types.ModuleType("blinkmimic.k1.k1_constants")
    module.SPEC = make_robot()
    package = types.ModuleType("blinkmimic.k1")
    package.__path__ = []
    sys.modules[package.__name__] = package
    sys.modules[module.__name__] = module
    blinkmimic.motion.MotionLib = ExactMotionLib
