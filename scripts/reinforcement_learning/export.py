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

"""Export a motion-independent policy graph from a supported local checkpoint."""

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


EXPORTERS = {
    "supermimic": "source.tasks.supermimic.export",
    "blinkmimic": "source.tasks.blinkmimic.export",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--algorithm", choices=tuple(EXPORTERS), required=True)
    if len(sys.argv) == 1 or sys.argv[1:] in (["-h"], ["--help"]):
        parser.print_help()
        return
    args, remaining = parser.parse_known_args()
    importlib.import_module(EXPORTERS[args.algorithm]).main(remaining)


if __name__ == "__main__":
    main()
