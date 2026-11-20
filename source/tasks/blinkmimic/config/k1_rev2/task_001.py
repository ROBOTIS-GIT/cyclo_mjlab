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

"""K1 Rev.2 BlinkMimic task configuration."""

from source import SRC_PATH

from .. import BlinkMimicTaskCfg
from .robot_spec import make_robot


MOTION_PATHS = (
    SRC_PATH / "assets/motions/K1_rev2/dance1/dance1.npz",
    SRC_PATH / "assets/motions/K1_rev2/dance2/dance2.npz",
)


def make_task_cfg() -> BlinkMimicTaskCfg:
    return BlinkMimicTaskCfg(robot_factory=make_robot, motion_paths=MOTION_PATHS)
