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

"""SuperMimic compatibility imports for the shared K1 reference adapter."""

from source.utils.k1_motion_npz_adapter import (  # noqa: F401
    ADAPTER_VERSION,
    BODY_FIELDS,
    FIELDS,
    ROOT_CONTRACT,
    adapt_motion,
    convert_motion,
    model_joint_names,
    prepare_motion_files,
    validate_motion,
)
