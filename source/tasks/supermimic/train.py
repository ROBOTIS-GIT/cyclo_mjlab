# Copyright 2026 The SuperMimic Authors.
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
# Authors: The SuperMimic Authors, Kiwoong Park

"""
Native K1 launcher adapted from official supermimic/scripts/train.py.

Reuse the original CLI config, environment factory and learner unchanged;
enforce
K1's 50 Hz reference/control-rate contract after parsing environment overrides.
"""

import os
from dataclasses import dataclass, field

import mjlab
import tyro
from mjlab.tasks.registry import list_tasks
from supermimic.scripts.train import TrainConfig as NativeTrainConfig
from supermimic.scripts.train_utils import create_env, setup_device

from source.utils.training_resume import (
    ResumeOptions,
    parse_resume_options,
    select_resume_checkpoint,
)
from . import REPO_ROOT
from .checkpoint_archive import load_checkpoint
from .resume import restore_training_state, validate_resume_configuration
from .config import TASKS, check_representation_matches
from .checkpoint import LocalBundleRunner as SuperMimicRunner
from .checkpoint_config import (
    environment_options,
    make_manifest,
    validate_control_rate,
    validate_manifest,
)


@dataclass
class TrainConfig(NativeTrainConfig):
    agent: ResumeOptions = field(default_factory=ResumeOptions)


def training_defaults(task):
    """Cyclo training defaults; saved bundle settings remain authoritative."""
    native = NativeTrainConfig.from_task(task)
    cfg = TrainConfig(env=native.env, runner=native.runner)
    representation, _ = TASKS[task]
    cfg.runner.representation = representation
    cfg.runner.actor_obs = "proprio"
    cfg.runner.history_length = 10
    return cfg


def parse_training_configuration(argv):
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        args=argv,
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    default = training_defaults(chosen_task)
    options = parse_resume_options(remaining_args)
    resume_path = select_resume_checkpoint(
        REPO_ROOT / "logs" / chosen_task, options
    )
    checkpoint = load_checkpoint(resume_path) if resume_path else None
    saved_environment_options = []
    if checkpoint is not None:
        default.env, default.runner = validate_manifest(
            checkpoint.manifest, chosen_task
        )
        default.runner.wandb_run_path = None
        default.num_envs = checkpoint.manifest.get(
            "num_envs", default.num_envs
        )
        saved_environment_options = checkpoint.manifest["environment_options"]
    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=default,
        config=mjlab.TYRO_FLAGS,
    )
    return (
        chosen_task,
        args,
        checkpoint,
        saved_environment_options,
        remaining_args,
    )


def main(argv=None):
    chosen_task, args, checkpoint, saved_options, arguments = (
        parse_training_configuration(argv)
    )
    check_representation_matches(chosen_task, args.runner.representation)
    validate_control_rate(args.env)
    if not args.motion_file:
        raise ValueError(
            "K1 training requires a validated local --motion-file"
        )
    print(f"[INFO] Loading environment: {chosen_task}")
    args.env.commands["motion"].motion_file = (
        [os.path.expanduser(p.strip()) for p in args.motion_file.split(",")]
        if "," in args.motion_file
        else os.path.expanduser(args.motion_file)
    )
    args.registry_name = None
    print(f"[INFO] Using local motion file: {args.motion_file}")
    args.env.commands["motion"].sampling_mode = (
        "external"
        if args.runner.sampling_mode == "return"
        else args.runner.sampling_mode
    )
    manifest = make_manifest(
        chosen_task,
        args.env,
        args.runner,
        [*saved_options, *environment_options(arguments)],
    )
    manifest["num_envs"] = args.num_envs
    validate_manifest(manifest, chosen_task)
    if checkpoint is not None:
        validate_resume_configuration(checkpoint, manifest)
        if args.runner.wandb_run_path is not None:
            raise ValueError(
                "Do not combine --agent.resume with world-model-only loading"
            )
        if args.runner.max_iterations <= checkpoint.step + 1:
            raise ValueError(
                "--runner.max-iterations must exceed the saved completed"
                " iterations"
            )
        manifest["resumed_from"] = str(checkpoint.source)
    device = setup_device(args.device)
    env = create_env(args.env, args.num_envs, device)
    try:
        runner = SuperMimicRunner(
            manifest=manifest,
            env=env,
            cfg=args.runner,
            task_id=chosen_task,
            registry_name=args.registry_name,
            device=device,
        )
        if checkpoint is not None:
            restore_training_state(runner, checkpoint)
        runner.run()
    finally:
        env.close()
