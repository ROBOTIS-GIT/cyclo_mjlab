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

"""
Task-specific isolated entry point for the released recipe (never bare
defaults).
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from source.utils.export_git_state import export_git_state, optional_git_metadata

from .runtime import ROOT, UPSTREAM_REVISION, VENDOR
from .config import TASKS
from .prepare import resolve_motion_files
from .training.checkpoint.selection import (
    checkpoint_recipe,
    directory_checkpoint,
    link_evaluation_checkpoint,
    publish_completed_snapshots,
    select_checkpoint,
)


LOG_ROOT = Path("logs/blinkmimic")


def training_run_order(directory, checkpoint=None):
    try:
        started = datetime.datetime.strptime(directory.name[:19], "%Y-%m-%d_%H-%M-%S").timestamp()
    except ValueError:
        # For a custom --run-dir, play and resume use the same training snapshot's time.
        if checkpoint is None:
            checkpoint = directory_checkpoint(directory, include_symlinks=False)
        started = checkpoint.stat().st_mtime if checkpoint else float("-inf")
    return started, directory.name


def latest_training_checkpoint(log_root, task):
    candidates = []
    for directory in (log_root / task).glob("*"):
        if not directory.is_dir():
            continue
        # Headless evaluations link their input checkpoint into the output directory.
        checkpoint = directory_checkpoint(directory, include_symlinks=False)
        if checkpoint is None:
            continue
        candidates.append((training_run_order(directory, checkpoint), checkpoint.resolve()))
    if not candidates:
        raise ValueError(
            f"No saved training checkpoints for {task} in {log_root}; use"
            " --checkpoint-dir or --checkpoint for a run outside this"
            " directory"
        )
    return max(candidates)[1]


def export_training_config_when_ready(child, run_dir, robot):
    """
    Use the trainer's saved arguments without modifying its parser or runtime.
    """
    from .runtime import prepare

    while True:
        try:
            settings = json.loads((run_dir / "args.json").read_text())
            break
        except (FileNotFoundError, json.JSONDecodeError):
            # args.json is written in place; retry while the trainer starts or
            # writes.
            if child.poll() is not None:
                raise RuntimeError(
                    "Trainer exited before saving complete args.json"
                )
            try:
                child.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass

    prepare()
    from .config.sim2real import export_sim2real_cfg

    if settings["robot"] != robot.name:
        raise ValueError(
            "Training robot does not match the deployment configuration"
        )
    from mjlab.utils.os import dump_yaml

    dump_yaml(run_dir / "params/agent.yaml", settings)
    return export_sim2real_cfg(run_dir, SimpleNamespace(**settings), robot)


def recipe():
    text = (VENDOR / "scripts/presets.sh").read_text()
    return shlex.split(re.search(r'\[main\]="([^"]+)"', text)[1])


def execution_recipe(mode, flags=None):
    """
    An explicit systems-only alternative; model and learning settings are
    intact.
    """
    flags = recipe() if flags is None else flags
    removed = []
    if mode == "compiled_no_torch_graphs":
        switches = {
            "--cuda_graph",
            "--graph_early",
            "--collect_graph",
            "--reset_masked",
            "--eprobe_graph",
        }
        result = []
        i = 0
        while i < len(flags):
            flag = flags[i]
            if flag in switches:
                removed.append(flag)
            elif flag == "--eprobe_async":
                removed.extend(flags[i: i + 2])
                i += 1
            else:
                result.append(flag)
            i += 1
        flags = result
    return flags, removed


def prepare_task_motions(task, output):
    from .runtime import prepare

    files = resolve_motion_files(TASKS[task].motion_paths)
    prepare()
    from .prepare import prepare_library

    robot = TASKS[task].robot_factory()
    prepare_library(files, output, robot.joint_names, mjcf_path=robot.mjcf_path)
    return output


def worker_environment(cache=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT), env.get("PYTHONPATH")))
    )
    for name, default in (
        ("WANDB_MODE", "disabled"),
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("OMP_NUM_THREADS", "4"),
    ):
        env.setdefault(name, default)
    if cache is not None:
        env.update(
            MPLCONFIGDIR=str(cache / "matplotlib"),
            TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"),
            TRITON_CACHE_DIR=str(cache / "triton"),
            XDG_CACHE_HOME=str(cache),
        )
    return env


def run_gui(args):
    command = [
        sys.executable,
        "-u",
        "-m",
        "source.tasks.blinkmimic.worker",
        "play",
        "--task",
        args.task,
        "--checkpoint",
        str(args.checkpoint),
        "--motion-dir",
        str(args.motion_dir.resolve()),
        "--port",
        str(args.port),
    ]
    if args.num_steps is not None:
        command.extend(["--num-steps", str(args.num_steps)])
    cache = args.cache_dir.resolve() if args.cache_dir else None
    return subprocess.run(
        command, cwd=ROOT, env=worker_environment(cache)
    ).returncode


def wait_for_training(child, run_dir, upload_run=None):
    """
    Collect training outputs while the child runs, including its final writes.
    """
    from .runtime.tensorboard import BlinkMimicTensorBoard

    metrics = BlinkMimicTensorBoard(run_dir)
    try:
        from source.utils.wandb_models import upload_checkpoint

        while True:
            finished = child.poll() is not None
            snapshots = publish_completed_snapshots(run_dir)
            metrics.update()
            if upload_run is not None:
                for snapshot in snapshots:
                    upload_checkpoint(upload_run, snapshot)
            if finished:
                return child.returncode
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    finally:
        metrics.close()


def create_parser():
    parser = argparse.ArgumentParser(
        description="K1 Rev.1 motion-only official BlinkMimic"
    )
    parser.add_argument("task", choices=tuple(TASKS))
    parser.add_argument("--num-envs", "--num_envs", type=int)
    parser.add_argument(
        "--motion-dir",
        type=Path,
        help="Override the selected task's motions with a prepared motion library",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Training output (default: logs/blinkmimic/<task>/<timestamp>)",
    )
    parser.add_argument(
        "--world-model-checkpoint",
        type=Path,
        help="Initialize only the world model when starting a new run",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "Training directory; defaults to the latest saved run for this"
            " task"
        ),
    )
    parser.add_argument(
        "--viewer", choices=("viser", "headless"), default="viser"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Browser viewer port"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Stop GUI playback after this many policy steps",
    )
    parser.add_argument(
        "--execution",
        choices=("official", "compiled_no_torch_graphs"),
        default="official",
        help=(
            "Explicit systems-only memory alternative; preserves"
            " models/batches/horizons. Native MuJoCo physics graphs remain."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Task-local compiler cache shared across isolated runs",
    )
    from source.utils.training_resume import add_resume_arguments

    add_resume_arguments(parser)
    return parser


def resolve_playback_inputs(args):
    if args.checkpoint is None and args.checkpoint_dir is None:
        args.checkpoint = latest_training_checkpoint(
            ROOT / LOG_ROOT, args.task
        )
    else:
        args.checkpoint = select_checkpoint(
            args.checkpoint, args.checkpoint_dir
        )
    if args.num_steps is not None and args.num_steps < 1:
        raise ValueError("--num-steps must be positive")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    print(f"[INFO] Selected checkpoint: {args.checkpoint}", flush=True)
    if args.motion_dir is not None:
        print(f"[INFO] Motion directory: {args.motion_dir}", flush=True)


def parse_arguments(parser, action, argv):
    args, extra = parser.parse_known_args(argv)
    if args.num_envs is not None:
        extra.extend(["--num_envs", str(args.num_envs)])
    if action == "train" and (args.checkpoint is not None or args.checkpoint_dir is not None):
        parser.error(
            "--checkpoint and --checkpoint-dir are only for play. "
            "To resume training, use --agent.resume True with "
            "--agent.load-checkpoint /path/to/model.pt or --agent.load-run /path/to/run. "
            "To load only the world model, use --world-model-checkpoint."
        )
    if args.world_model_checkpoint is not None:
        if action != "train" or args.resume:
            parser.error("--world-model-checkpoint requires fresh training")
        path = args.world_model_checkpoint.expanduser().resolve()
        if not path.is_file():
            parser.error(f"Missing world model checkpoint: {path}")
        args.world_model_checkpoint = path
    if action not in ("train", "play"):
        parser.error("Supported actions are train and play")
    # Recipe overrides remain explicit and recorded, with no implicit resource
    # reduction.
    forbidden = {
        "--robot",
        "--motion_dir",
        "--log_dir",
        "--run_dir",
        "--ckpt",
        "--resume",
    }
    if any(x.split("=", 1)[0] in forbidden for x in extra):
        parser.error("Use task-specific motion/run/checkpoint options")
    return args, extra


def prepare_run_inputs(action, args, extra):
    if action == "play":
        if (
            args.resume
            or args.load_run != ".*"
            or args.load_checkpoint != "last.pt"
        ):
            raise ValueError("--agent.* resume options are only for training")
        resolve_playback_inputs(args)

    if (
        args.motion_dir is not None
        and not (args.motion_dir / "manifest.json").is_file()
    ):
        raise ValueError(
            "Motion directory requires its provenance manifest.json; use"
            " --motion-dir"
        )

    if action == "play" and args.viewer == "viser":
        if extra:
            raise ValueError(f"Unsupported GUI options: {' '.join(extra)}")
        if args.run_dir is not None:
            raise ValueError(
                "--run-dir is only used for training or headless evaluation"
            )


def resolve_training_resume(args, extra, resume_path):
    from .training.checkpoint.resume import read_resume_checkpoint, training_arguments

    saved = read_resume_checkpoint(resume_path)
    effective_recipe, removed_systems_flags = execution_recipe(
        args.execution, training_arguments(saved)
    )
    if "--wm_structured" in recipe():
        effective_recipe.append("--wm_structured")
    iterations_parser = argparse.ArgumentParser(add_help=False)
    iterations_parser.add_argument(
        "--iterations", type=int, default=saved["args"]["iterations"]
    )
    iterations, _ = iterations_parser.parse_known_args(extra)
    if iterations.iterations <= saved["iteration"]:
        raise ValueError(
            "--iterations must exceed the saved completed iterations"
        )
    extra = [*extra, "--resume", str(resume_path)]
    print(
        f"[INFO] Resuming {resume_path} at iteration {saved['iteration']}; "
        "simulation and replay buffer restart",
        flush=True,
    )
    args.resume_checkpoint = resume_path
    args.resume_iteration = saved["iteration"]
    return effective_recipe, removed_systems_flags, extra


def initialize_resume_log(new_run):
    """The pinned native trainer writes this header only at iteration zero."""
    (new_run / "train_log.csv").write_text(
        "iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,"
        "pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr,"
        "pi_lr,pi_mult,nonfinite\n"
    )


def run_training(args, extra):
    from .runtime import prepare

    if args.run_dir is None:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.run_dir = ROOT / LOG_ROOT / args.task / stamp
    prepare()
    robot = TASKS[args.task].robot_factory()
    out = args.run_dir.resolve()
    effective_recipe, removed_systems_flags = execution_recipe(args.execution)
    from source.utils.training_resume import select_resume_checkpoint

    args.motion_source = "task" if args.motion_dir is None else "explicit"
    resume_path = select_resume_checkpoint(
        ROOT / LOG_ROOT / args.task, args,
        skip_discovered_symlinks=True, run_sort_key=training_run_order,
    )
    if resume_path is not None:
        effective_recipe, removed_systems_flags, extra = (
            resolve_training_resume(args, extra, resume_path)
        )
    out.mkdir(parents=True, exist_ok=False)
    export_git_state(out, ROOT)
    if args.motion_dir is None:
        args.motion_dir = prepare_task_motions(args.task, out / "motions")
    if resume_path and args.resume_iteration > 0:
        initialize_resume_log(out)
    if args.world_model_checkpoint:
        extra = [
            *extra,
            "--world-model-checkpoint",
            str(args.world_model_checkpoint),
        ]
    flags = [
        *checkpoint_recipe(effective_recipe),
        *extra,
        "--robot",
        robot.name,
        "--motion_dir",
        str(args.motion_dir.resolve()),
        "--log_dir",
        str(out),
    ]
    return run_worker(
        args,
        action="train",
        flags=flags,
        extra=extra,
        robot=robot,
        removed_systems_flags=removed_systems_flags,
    )


def run_playback(args, extra):
    if args.viewer == "viser":
        if args.motion_dir is None:
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as tmp:
                args.motion_dir = prepare_task_motions(args.task, Path(tmp) / "motions")
                return run_gui(args)
        return run_gui(args)

    from .runtime import prepare

    if args.run_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.run_dir = args.checkpoint.parent / "play" / stamp
    prepare()
    robot = TASKS[args.task].robot_factory()
    out = args.run_dir.resolve()
    _, removed_systems_flags = execution_recipe(args.execution)
    out.mkdir(parents=True, exist_ok=False)
    if args.motion_dir is None:
        args.motion_dir = prepare_task_motions(args.task, out / "motions")
    flags = [
        *extra,
        "--robot",
        robot.name,
        "--motion_dir",
        str(args.motion_dir.resolve()),
        "--run_dir",
        str(out),
    ]
    if args.checkpoint.name == "last.pt":
        # The upstream single-file evaluator writes optional results beside its input.
        local_checkpoint = out / "last.pt"
        local_checkpoint.symlink_to(args.checkpoint)
        flags.extend(["--ckpt", str(local_checkpoint)])
    else:
        link_evaluation_checkpoint(args.checkpoint, out)
    print(f"[INFO] Playback log: {out / 'console.log'}", flush=True)
    return run_worker(
        args,
        action="play",
        flags=flags,
        extra=extra,
        robot=robot,
        removed_systems_flags=removed_systems_flags,
    )


def build_run_record(
    args, action, command, cache, removed_systems_flags, extra
):
    return dict(
        task=args.task,
        action=action,
        motion_source=getattr(args, "motion_source", None),
        resumed_from=(
            str(args.resume_checkpoint)
            if getattr(args, "resume_checkpoint", None)
            else None
        ),
        command=command,
        cwd=str(ROOT),
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        repository_commit=optional_git_metadata(ROOT, "rev-parse", "HEAD"),
        upstream=UPSTREAM_REVISION,
        manifest=str(args.motion_dir.resolve() / "manifest.json"),
        checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
        manifest_sha256=hashlib.sha256(
            (args.motion_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        runtime={
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "mjlab",
                "mujoco",
                "mujoco-warp",
                "warp-lang",
            )
        },
        python=sys.version,
        compiler_cache=str(cache),
        source_status=optional_git_metadata(ROOT, "status", "--short"),
        recipe=recipe(),
        execution=args.execution,
        removed_systems_flags=removed_systems_flags,
        explicit_overrides=extra,
        status="starting",
        stop=(
            "Verify child PID and command, then SIGINT that PID only. No"
            " automatic restart."
        ),
    )


def update_run_record(path, record, **fields):
    record.update(fields)
    path.write_text(json.dumps(record, indent=2))


def stop_worker(child):
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def run_worker(args, *, action, flags, extra, robot, removed_systems_flags):
    mode = "train" if action == "train" else "evaluate"
    command = [
        sys.executable,
        "-u",
        "-m",
        "source.tasks.blinkmimic.worker",
        mode,
        "--task",
        args.task,
        *flags,
    ]
    out = args.run_dir.resolve()
    cache = args.cache_dir.resolve() if args.cache_dir else out / "cache"
    env = worker_environment(cache)
    record = build_run_record(
        args, action, command, cache, removed_systems_flags, extra
    )
    record_path = out / "run-record.json"
    update_run_record(record_path, record)
    with (out / "console.log").open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            update_run_record(
                record_path, record, pid=child.pid, status="running"
            )
            if action == "train":
                config_path = export_training_config_when_ready(
                    child, out, robot
                )
                print(
                    f"[INFO] Exported sim2real config: {config_path}",
                    file=log,
                    flush=True,
                )
                from source.utils.wandb_models import model_upload_run

                with model_upload_run(out, args.task) as upload_run:
                    code = wait_for_training(child, out, upload_run)
            else:
                code = child.wait()
        except BaseException as error:
            try:
                stop_worker(child)
            except Exception as cleanup_error:
                error.add_note(f"Worker cleanup failed: {cleanup_error}")
            try:
                update_run_record(
                    record_path,
                    record,
                    status=(
                        "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed"
                    ),
                    returncode=child.returncode,
                    finished_at=datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                )
            except Exception as record_error:
                error.add_note(
                    f"Could not record worker failure: {record_error}"
                )
            raise
    update_run_record(
        record_path,
        record,
        status=(
            "completed"
            if code == 0
            else "interrupted" if code == -2 else "failed"
        ),
        returncode=code,
        finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    return code


def main(action="train", argv=None):
    parser = create_parser()
    args, extra = parse_arguments(parser, action, argv)
    try:
        prepare_run_inputs(action, args, extra)
    except ValueError as error:
        parser.error(str(error))

    if action == "train":
        code = run_training(args, extra)
    else:
        code = run_playback(args, extra)
    raise SystemExit(code)


if __name__ == "__main__":
    main(argv=sys.argv[1:])
