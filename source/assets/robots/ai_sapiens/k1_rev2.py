"""K1 Rev2 uses the Rev1 actuator layout and its own official collision model."""

from copy import deepcopy
from dataclasses import replace

from .k1_rev1 import K1_REV1_CFG, K1_REV1_XML_PATH, load_k1_spec

K1_REV2_XML_PATH = K1_REV1_XML_PATH.parents[1] / "k1_rev2" / "k1.xml"


def load_k1_rev2_spec():
    return load_k1_spec(K1_REV2_XML_PATH)


K1_REV2_CFG = replace(K1_REV1_CFG, spec_fn=load_k1_rev2_spec)
K1_REV2_FOOT_GEOMS = ("left_ankle_roll_link_collision_0", "right_ankle_roll_link_collision_0")


def get_k1_rev2_cfg():
    return deepcopy(K1_REV2_CFG)
