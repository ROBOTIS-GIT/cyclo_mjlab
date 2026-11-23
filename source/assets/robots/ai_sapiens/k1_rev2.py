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
# Author: Insu Park

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
