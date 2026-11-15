# Copyright 2026 The SuperMimic Authors.
# SPDX-License-Identifier: Apache-2.0
"""Native K1 launcher adapted from official supermimic/scripts/train.py.

Reuse the original CLI config, environment factory and learner unchanged; enforce
K1's 50 Hz reference/control-rate contract after parsing environment overrides.
"""

import math
import os

import mjlab
import tyro
from mjlab.tasks.registry import list_tasks
from supermimic.runners.supermimic_runner import SuperMimicRunner
from supermimic.scripts.train import TrainConfig
from supermimic.scripts.train_utils import create_env, setup_device


def validate_control_rate(env_cfg):
    timestep = env_cfg.sim.mujoco.timestep
    decimation = env_cfg.decimation
    step_dt = timestep * decimation
    if (
        timestep <= 0
        or decimation <= 0
        or not math.isclose(step_dt, 0.02, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError(
            "50 Hz K1 references require env.sim.mujoco.timestep * env.decimation "
            f"== 0.02 seconds; got {timestep} * {decimation} = {step_dt}. "
            "Use compensating overrides that keep the control rate at 50 Hz."
        )


def main():
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(list_tasks()),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        config=mjlab.TYRO_FLAGS,
    )
    validate_control_rate(args.env)
    if not args.motion_file:
        raise ValueError("K1 training requires a validated local --motion-file")
    device = setup_device(args.device)
    print(f"[INFO] Loading environment: {chosen_task}")
    args.env.commands["motion"].motion_file = (
        [os.path.expanduser(p.strip()) for p in args.motion_file.split(",")]
        if "," in args.motion_file
        else os.path.expanduser(args.motion_file)
    )
    args.registry_name = None
    print(f"[INFO] Using local motion file: {args.motion_file}")
    if "MaxCoord" in chosen_task and args.runner.representation == "gencoord":
        args.runner.representation = "maxcoord"
        print("[INFO] Auto-detected maxcoord representation from task_id")
    args.env.commands["motion"].sampling_mode = (
        "external"
        if args.runner.sampling_mode == "return"
        else args.runner.sampling_mode
    )
    env = create_env(args.env, args.num_envs, device)
    runner = SuperMimicRunner(
        env=env,
        cfg=args.runner,
        task_id=chosen_task,
        registry_name=args.registry_name,
        device=device,
    )
    runner.run()
