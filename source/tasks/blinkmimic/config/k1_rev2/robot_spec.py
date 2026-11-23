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
# Authors: Woojin Wie, Kiwoong Park

"""BlinkMimic K1 configuration with the Rev2 asset."""

from ..k1_rev1.robot_spec import TRACKED_BODIES as TRACKED_BODIES, make_k1_spec, tracking_pose


def robot_cfg():
    from source.assets.robots.ai_sapiens.k1_rev2 import get_k1_rev2_cfg

    return tracking_pose(get_k1_rev2_cfg())


def make_robot():
    from source.assets.robots.ai_sapiens.k1_rev2 import K1_REV2_XML_PATH

    return make_k1_spec("k1_rev2", K1_REV2_XML_PATH, robot_cfg)
