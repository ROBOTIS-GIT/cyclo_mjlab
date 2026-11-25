# Copyright 2026 The SuperMimic Authors.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Cyclo: local validated bundles and bounded headless playback.
"""
Local K1 bundle playback with the official actor and optional W&B playback.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab
import torch
import tyro
from mjlab.tasks.registry import load_env_cfg
from supermimic.representations import extract_actor_goal
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.train_utils import create_env, setup_device
from supermimic.tasks.task_spec import build_actor_obs_td

from . import LOG_ROOT
from .config import TASKS, check_representation_matches, select_trained_encoder
from .training.checkpoint.archive import initialize_world_model, load_checkpoint


@dataclass
class PlayConfig:
    motion_file: str
    checkpoint_dir: Path | None = None
    checkpoint: Path | None = None
    wandb_path: str | None = None
    device: str | None = None
    viewer: Literal["auto", "headless", "native", "viser"] = "viser"
    num_steps_rollout: int = 1000000
    initial_task: Literal["motion", "velocity"] | None = None
    export_policy: bool = True


def load_models(directory, rcfg, env, device):
    checkpoint = load_checkpoint(directory)
    wm = build_world_model(rcfg, env, device)
    initialize_world_model(wm, checkpoint.world_model)
    actor = build_actor(rcfg, wm, device)
    actor.load_state_dict(checkpoint.actor)
    actor.eval()
    wm.dynamics.eval()
    return wm, actor


def headless_play(env, wm, actor, steps, task="motion"):
    """
    Actual native simulator playback, including actor-history reset boundaries.
    """
    actor.set_slot_mix({task: 1.0})
    obs, _ = env.reset()
    if task == "velocity":
        spec = next(spec for spec in actor.task_specs if spec.name == task)
        command = obs["full_state"].new_tensor([goal.default for goal in spec.goal_specs])
        velocity_goal = spec.build_inference_goal_obs(command.expand(env.num_envs, -1))
    history = None
    with torch.no_grad():
        for _ in range(steps):
            state = obs["full_state"]
            rep = wm.representation.represent(state)
            aux = wm.representation.extract_aux_state(state)
            sensor = wm.representation.compute_actor_sensor(
                rep,
                wm.sim_context.read(env.unwrapped),
                aux_state=aux,
                sim_actor_obs=(
                    obs["actor"]
                    if wm.representation.actor_obs_mode == "proprio"
                    else None
                ),
            )
            history = (
                sensor.unsqueeze(1).repeat(1, actor.history_length, 1)
                if history is None
                else torch.cat([history[:, 1:], sensor.unsqueeze(1)], dim=1)
            )
            goal = extract_actor_goal(obs["goal"]) if task == "motion" else velocity_goal
            action = actor(build_actor_obs_td({task: goal}, history))
            if not torch.isfinite(action).all():
                raise ValueError("Checkpoint produced nonfinite actions")
            obs, _, dones, _ = env.step(action)
            if dones.any():
                # Playback is single-environment; reset padding matches
                # collection.
                history = None
    print(
        f"PASS local playback: {steps} native simulator steps, finite actions"
    )


def latest_playback_checkpoint(log_root, task):
    """Return the latest complete checkpoint for this task, already loaded."""
    for directory in sorted((log_root / task).glob("*"), reverse=True):
        if (directory / "last.pt").is_file():
            checkpoint = load_checkpoint(directory)
            if checkpoint.manifest.get("task_id") == task:
                return checkpoint
            continue
        if not all((directory / name).is_file() for name in ("actor.pt", "world_model.pt")):
            continue
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
        except (FileNotFoundError, ValueError):
            continue
        if isinstance(manifest, dict) and manifest.get("task_id") == task:
            return load_checkpoint(directory)
    raise ValueError(
        f"No saved checkpoint bundle for {task} in {log_root / task}; "
        "use --checkpoint-dir for a run outside this directory"
    )


def resolve_playback_checkpoint(task, cfg):
    if (
        sum(
            value is not None
            for value in (cfg.checkpoint, cfg.checkpoint_dir, cfg.wandb_path)
        )
        > 1
    ):
        raise ValueError(
            "Choose only one --checkpoint, --checkpoint-dir or --wandb-path"
        )
    if cfg.num_steps_rollout < 1:
        raise ValueError("--num-steps-rollout must be positive")
    if cfg.checkpoint is None and cfg.checkpoint_dir is None and cfg.wandb_path is None:
        checkpoint = latest_playback_checkpoint(LOG_ROOT, task)
        print(f"[INFO] Selected checkpoint: {checkpoint.source}")
        return checkpoint
    if cfg.wandb_path is not None:
        import wandb
        from supermimic.utils.wandb_artifacts import download_model_dir

        directory = download_model_dir(wandb.Api().run(cfg.wandb_path))
        return load_checkpoint(directory)
    return load_checkpoint(
        cfg.checkpoint if cfg.checkpoint is not None else cfg.checkpoint_dir,
    )


def run_interactive_playback(env, wm, actor, cfg, slot_mix, initial_task):
    from supermimic.scripts.viewers import (
        SuperMimicNativeViewer,
        create_wm_vis_model,
    )

    from .viewer import ClipPlaybackViewer, adapt_ghost_visualizer

    resolved = cfg.viewer
    if resolved == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved = "native" if has_display else "viser"
    print(f"[INFO] Starting viewer: {resolved}")
    viewer = (
        SuperMimicNativeViewer
        if resolved == "native"
        else ClipPlaybackViewer
    )
    import glob

    options = (
        {
            "motion_names": [
                Path(p).stem for p in sorted(glob.glob(cfg.motion_file))
            ]
        }
        if resolved == "viser"
        else {}
    )
    playback = viewer(
        env=env,
        actor=actor,
        dynamics=wm.dynamics,
        representation=wm.representation,
        wm_vis_model=create_wm_vis_model(env.unwrapped.sim.mj_model),
        num_steps_rollout=cfg.num_steps_rollout,
        sim_context=wm.sim_context,
        task_specs=[spec for spec in actor.task_specs if slot_mix.get(spec.name, 0) > 0],
        initial_task=initial_task,
        **options,
    )
    adapt_ghost_visualizer(playback)
    playback.run()


def play(task, cfg):
    checkpoint = resolve_playback_checkpoint(task, cfg)
    rcfg = load_runner_cfg(checkpoint.manifest)
    initial_task = select_trained_encoder(rcfg.slot_mix, cfg.initial_task)
    check_representation_matches(task, rcfg.representation)
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.commands["motion"].motion_file = cfg.motion_file
    if cfg.export_policy:
        from .export import export_play_policies

        source_task = checkpoint.manifest["task_id"]
        if source_task in TASKS:
            export_play_policies(checkpoint, source_task)
        else:
            print(f"[INFO] Skipping automatic ONNX export: training task {source_task!r} is unregistered.")
    device = setup_device(cfg.device)
    env = create_env(env_cfg, 1, device)
    try:
        wm, actor = load_models(checkpoint, rcfg, env, device)
        if cfg.viewer == "headless":
            headless_play(
                env, wm, actor, cfg.num_steps_rollout, initial_task
            )
        else:
            run_interactive_playback(env, wm, actor, cfg, rcfg.slot_mix, initial_task)
    finally:
        env.close()


def main(argv=None):
    task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(TASKS),
        args=argv,
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    play(task, tyro.cli(PlayConfig, args=remaining, config=mjlab.TYRO_FLAGS))
