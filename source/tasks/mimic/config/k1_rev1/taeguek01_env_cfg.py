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

"""K1 Rev.1 Taeguek 01 Mimic environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from source.tasks.mimic.motion_curriculum import TAEGUEK_TRAJECTORIES

from .taeguek_env_cfg import k1_rev1_taeguek_env_cfg

TRAJECTORY_FILE = TAEGUEK_TRAJECTORIES[1]


def k1_rev1_taeguek01_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the K1 Rev.1 Taeguek 01 Mimic configuration."""
  return k1_rev1_taeguek_env_cfg(1, play=play)
