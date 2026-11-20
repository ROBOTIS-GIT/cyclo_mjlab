"""BlinkMimic K1 configuration with the Rev2 asset."""

from ..k1_rev1.robot_spec import TRACKED_BODIES as TRACKED_BODIES, make_k1_spec, tracking_pose


def robot_cfg():
    from source.assets.robots.ai_sapiens.k1_rev2 import get_k1_rev2_cfg

    return tracking_pose(get_k1_rev2_cfg())


def make_robot():
    from source.assets.robots.ai_sapiens.k1_rev2 import K1_REV2_XML_PATH

    return make_k1_spec("k1_rev2", K1_REV2_XML_PATH, robot_cfg)
