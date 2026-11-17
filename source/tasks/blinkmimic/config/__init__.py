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

"""Registered BlinkMimic task configurations."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blinkmimic.robot import RobotSpec


@dataclass(frozen=True)
class BlinkMimicTaskCfg:
    robot_factory: Callable[[], "RobotSpec"]
    # Each entry may be an NPZ file, a directory (recursive), or a glob pattern.
    motion_paths: tuple[Path, ...]


# Define the configuration type before importing tasks that instantiate it.
from .k1_rev1 import TASKS  # noqa: E402
