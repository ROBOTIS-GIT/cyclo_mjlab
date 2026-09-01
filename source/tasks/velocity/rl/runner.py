# Copyright 2026 ROBOTIS CO., LTD.
# Copyright 2025, The mjlab Developers
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
#
# This file includes modifications by ROBOTIS CO., LTD. to code derived
# from mujocolab/mjlab.

from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  """Velocity runner that uses the standard checkpoint save behavior."""
