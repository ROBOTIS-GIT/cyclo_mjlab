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

"""K1 Rev.1 Mimic task registration."""

from mjlab.tasks.registry import register_mjlab_task
from source.tasks.mimic.rl import MotionTrackingOnPolicyRunner

from .agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg
from .dance1_env_cfg import k1_rev1_dance1_env_cfg
from .dance2_env_cfg import k1_rev1_dance2_env_cfg


register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-Dance1",
  env_cfg=k1_rev1_dance1_env_cfg(),
  play_env_cfg=k1_rev1_dance1_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Cyclo-Mimic-K1-Rev1-Dance2",
  env_cfg=k1_rev1_dance2_env_cfg(),
  play_env_cfg=k1_rev1_dance2_env_cfg(play=True),
  rl_cfg=k1_rev1_mimic_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
