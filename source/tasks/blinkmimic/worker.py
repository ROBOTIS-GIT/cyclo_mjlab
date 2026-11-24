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
# Authors: Kiwoong Park, Woojin Wie

"""Private subprocess boundary for upstream MJLab performance patches."""

import argparse
from dataclasses import asdict
from enum import Enum
import json
import runpy
import sys
import traceback
from pathlib import Path

from .runtime import install_robot, prepare
from .config import TASKS


def plain_env_config(value):
    """Represent configuration values as portable YAML data."""
    if isinstance(value, dict):
        return {key: plain_env_config(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_env_config(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def record_training_failure(error):
    # Read only Python bookkeeping from the failing frame; do not touch a
    # failed
    # CUDA context or restart. Preserve the exact loop index absent from sparse
    # logs.
    tb = error.__traceback__
    state = {}
    while tb:
        frame = tb.tb_frame
        if (
            frame.f_code.co_filename.endswith("/blinkmimic/train.py")
            and "args" in frame.f_globals
        ):
            g = frame.f_globals
            it = g.get("it")
            state = {
                "loop_index_zero_based": it,
                "last_completed_iterations": it,
                "wm_update_calls": getattr(
                    g.get("wm_update_g"), "calls", None
                ),
                "policy_update_calls": getattr(
                    g.get("policy_update_g"), "calls", None
                ),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            out = Path(g["args"].log_dir)
        tb = tb.tb_next
    if state:
        (out / "failure.json").write_text(json.dumps(state, indent=2))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("mode", choices=("train", "evaluate", "play"))
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--log_dir", type=Path)
    parser.add_argument("--resume")
    parser.add_argument("--world-model-checkpoint", type=Path)
    parser.add_argument("--motion_dir", type=Path)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.log_dir:
        sys.argv.extend(["--log_dir", str(args.log_dir)])
    if args.motion_dir:
        sys.argv.extend(["--motion_dir", str(args.motion_dir)])
    if args.resume:
        sys.argv.extend(["--resume", args.resume])
    prepare()
    robot = TASKS[args.task].robot_factory()
    install_robot(robot)
    try:
        if args.mode == "play":
            from .play import main as play

            play(robot=robot)
        elif args.mode == "train":
            from .training.checkpoint.snapshots import include_critic_in_snapshots

            if "--ckpt_light" in remaining:
                raise ValueError(
                    "--ckpt_light is incompatible with critic snapshots"
                )
            from .training.checkpoint.resume import allow_model_only_resume, initialize_world_model

            if args.log_dir is not None:
                from blinkmimic.env import DECIMATION, make_sim
                import yaml

                env_cfg = {
                    "decimation": DECIMATION,
                    "sim": asdict(make_sim(robot)),
                    "robot": {**asdict(robot), "robot_cfg": asdict(robot.robot_cfg())},
                }
                path = args.log_dir / "params/env.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as stream:
                    yaml.safe_dump(plain_env_config(env_cfg), stream, sort_keys=False)

            from .training.checkpoint.sampler import sampler_clips

            clips = sampler_clips(args.motion_dir) if args.motion_dir else None
            with (
                allow_model_only_resume(
                    args.resume,
                    clips,
                    target_robot=robot,
                ) as history,
                include_critic_in_snapshots(history, clips),
                initialize_world_model(args.world_model_checkpoint),
            ):
                runpy.run_module("blinkmimic.train", run_name="__main__")
        else:
            from .training.checkpoint.resume import evaluation_reference_buffers

            with evaluation_reference_buffers(robot):
                runpy.run_module("blinkmimic.evaluate", run_name="__main__")
    except BaseException as error:
        record_training_failure(error)
        raise


if __name__ == "__main__":
    main()
