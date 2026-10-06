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

"""K1 Rev.1 UMR Stand-Up Mimic environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from source import SRC_PATH
from source.tasks.mimic.mdp import ReferenceTrajectoryCommandCfg

from .base_env_cfg import k1_rev1_mimic_env_cfg

TRAJECTORY_FILE = (
  SRC_PATH / "assets" / "motions" / "K1_rev1" / "umr_stand_up"
  / "k1_0008_stand_up_soft_landing_converted.npz"
)


def k1_rev1_umr_stand_up_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the K1 Rev.1 UMR Stand-Up Mimic configuration."""
  cfg = k1_rev1_mimic_env_cfg(play=play)
  # This reference uses deep hip/knee flexion within the model hard limits.
  # Match soft limits to hard limits for this task, including reset clipping.
  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  articulation.soft_joint_pos_limit_factor = 1.0
  reference_trajectory = cfg.commands["reference_trajectory"]
  assert isinstance(reference_trajectory, ReferenceTrajectoryCommandCfg)
  reference_trajectory.trajectory_file = str(TRAJECTORY_FILE)
  if not play:
    reference_trajectory.first_frame_env_fraction = 0.3
    reference_trajectory.first_frame_joint_position_noise = (-0.2, 0.2)
  # Prone transitions intentionally use knee, arm, and torso ground contacts.
  # The dance task penalizes these contacts, which conflicts with this reference.
  cfg.rewards.pop("undesired_contacts", None)
  return cfg
