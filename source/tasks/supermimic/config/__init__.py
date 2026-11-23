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

"""Registered SuperMimic task configurations."""

from .k1_rev1 import TASKS as REV1_TASKS
from .k1_rev2 import TASKS as REV2_TASKS

TASKS = {**REV1_TASKS, **REV2_TASKS}


def register_tasks():
    from mjlab.rl import RslRlBaseRunnerCfg
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    for task_id, (representation, make_env_cfg) in TASKS.items():
        if task_id in list_tasks():
            continue
        register_mjlab_task(
            task_id=task_id,
            env_cfg=make_env_cfg(representation),
            play_env_cfg=make_env_cfg(representation, play=True),
            # Required by the registry; training uses SuperMimicRunnerCfg directly.
            rl_cfg=RslRlBaseRunnerCfg(),
        )


register_tasks()


def check_representation_matches(task_id, representation):
    expected, _ = TASKS[task_id]
    if representation != expected:
        raise ValueError(
            f"{task_id} is a {expected} task, but the selected representation is {representation}"
        )
