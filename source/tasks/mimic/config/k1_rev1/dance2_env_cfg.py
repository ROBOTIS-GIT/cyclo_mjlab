# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""K1 Rev.1 Dance 2 Mimic environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from source import SRC_PATH
from source.tasks.mimic.mdp import ReferenceTrajectoryCommandCfg

from .base_env_cfg import k1_rev1_mimic_env_cfg

TRAJECTORY_FILE = (
  SRC_PATH / "assets" / "motions" / "K1_rev1" / "dance2" / "dance2.npz"
)


def k1_rev1_dance2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the K1 Rev.1 Dance 2 Mimic configuration."""
  cfg = k1_rev1_mimic_env_cfg(play=play)
  reference_trajectory = cfg.commands["reference_trajectory"]
  assert isinstance(reference_trajectory, ReferenceTrajectoryCommandCfg)
  reference_trajectory.trajectory_file = str(TRAJECTORY_FILE)
  return cfg
