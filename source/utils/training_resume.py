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

"""Checkpoint selection using the same resume flags as the RSL runner."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResumeOptions:
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "last.pt"


def _boolean(value):
    if value not in ("True", "False"):
        raise argparse.ArgumentTypeError("Use True or False")
    return value == "True"


def add_resume_arguments(parser):
    parser.add_argument(
        "--agent.resume",
        dest="resume",
        type=_boolean,
        default=False,
        metavar="{True,False}",
        help="Resume saved models and available optimizer state in a new run",
    )
    parser.add_argument(
        "--agent.load-run",
        dest="load_run",
        default=".*",
        help="Run directory or regex under the task log directory",
    )
    parser.add_argument(
        "--agent.load-checkpoint",
        dest="load_checkpoint",
        default="last.pt",
        help="Checkpoint path or filename regex (default: last.pt)",
    )


def parse_resume_options(arguments):
    parser = argparse.ArgumentParser(add_help=False)
    add_resume_arguments(parser)
    options, _ = parser.parse_known_args(arguments)
    return ResumeOptions(**vars(options))


def select_resume_checkpoint(log_root, options):
    if not options.resume:
        if options.load_run != ".*" or options.load_checkpoint != "last.pt":
            raise ValueError("Checkpoint selection requires --agent.resume")
        return None
    checkpoint = Path(options.load_checkpoint).expanduser()
    if checkpoint.is_absolute():
        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint.resolve()

    root = Path(log_root)
    explicit = Path(options.load_run).expanduser()
    if not explicit.is_absolute():
        explicit = root / explicit
    if explicit.is_dir():
        runs = [explicit]
    else:
        pattern = re.compile(options.load_run)
        runs = sorted(
            (
                p
                for p in root.glob("*")
                if p.is_dir() and pattern.match(p.name)
            ),
            reverse=True,
        )
    pattern = re.compile(options.load_checkpoint)
    for run in runs:
        exact = run / checkpoint
        if exact.is_file():
            return exact.resolve()
        candidates = [
            p
            for directory in (run, run / "checkpoints")
            for p in directory.glob("*.pt")
            if pattern.fullmatch(p.name)
        ]
        if candidates:
            # model_10000.pt must sort after model_9000.pt.
            return max(
                candidates, key=lambda p: (len(p.stem), p.name)
            ).resolve()
    raise ValueError(
        f"No resume checkpoint {options.load_checkpoint!r} in "
        f"{root} (run={options.load_run!r})"
    )
