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
# Author: Woojin Wie

"""Carry relevant software attribution with policy-only export bundles."""

import hashlib
import importlib.metadata
from pathlib import Path


def distribution_license(name):
    """Locate the installed distribution's own license, excluding dependency copies."""
    distribution = importlib.metadata.distribution(name)
    candidates = [
        path
        for path in distribution.files or []
        if path.name == "LICENSE"
        and path.parent.name == "licenses"
        and str(path.parent.parent).endswith(".dist-info")
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Cannot locate the primary LICENSE for {name}")
    return Path(distribution.locate_file(candidates[0]))


def write_software_notices(directory, attribution, licenses):
    """Copy original texts and return relative file paths with SHA256 hashes."""
    files = {"NOTICE.txt": attribution.encode("utf-8")}
    files.update(
        {f"licenses/{name}": Path(path).read_bytes() for name, path in licenses.items()}
    )
    for relative, content in files.items():
        destination = Path(directory) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }
