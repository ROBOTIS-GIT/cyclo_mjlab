"""Opt-in bounded CPU acceptance of the actual learners and local playback.

Run from the checkout: python tests/supermimic/validate_native.py GenCoord dance1 /tmp/output
Set WANDB_MODE=offline and TORCHDYNAMO_DISABLE=1 for eager CPU validation.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg
from supermimic.scripts.train_utils import create_env

from source.tasks.supermimic import REPO_ROOT
from source.tasks.supermimic.checkpoint import (
    LocalBundleRunner,
    make_manifest,
    read_bundle,
)
from source.tasks.supermimic.motion import prepare_motion_files
from source.tasks.supermimic.play import PlayConfig, load_models, play
from source.tasks.supermimic.tasks import k1_env_cfg, register_tasks


def parameters(module):
    return [p.detach().clone() for p in module.parameters()]


def changed(before, module):
    return any(
        not torch.equal(a, b) for a, b in zip(before, module.parameters(), strict=True)
    )


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    else:
        assert a == b


def main():
    representation, dance, output = sys.argv[1:]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    torch.set_num_threads(2)
    torch.manual_seed(7)
    register_tasks()
    task = f"Cyclo-SuperMimic-{representation}-K1-Rev1"
    cfg = k1_env_cfg(representation.lower())
    # Exercise the nondefault substep contract in actual training and playback.
    cfg.sim.mujoco.timestep, cfg.decimation = 0.0025, 8
    source = REPO_ROOT / f"source/assets/motions/K1_rev1/{dance}/{dance}.npz"
    original = source.read_bytes()
    motion = prepare_motion_files(
        str(source), cfg.scene.entities["robot"].spec_fn().compile(), output / "adapter"
    )
    cfg.commands["motion"].motion_file = motion
    cfg.commands["motion"].sampling_mode = "external"
    rcfg = SuperMimicRunnerCfg(
        representation=representation.lower(),
        max_iterations=6,
        save_interval=1000,
        log_interval=100,
        buffer_size=16,
        replay_ratio=2,
        dynamics_hidden_dims=(32, 32),
        actor_hidden_dims=(32, 32),
        critic_hidden_dims=(32, 32),
    )
    rcfg.wm.horizon, rcfg.wm.batch_size, rcfg.wm.num_batches = 2, 2, 1
    rcfg.policy.horizon, rcfg.policy.batch_size, rcfg.policy.num_batches = 2, 2, 1
    manifest = make_manifest(
        task,
        cfg,
        rcfg,
        ["--env.sim.mujoco.timestep", "0.0025", "--env.decimation", "8"],
    )
    env = create_env(cfg, 2, "cpu")
    updates = {"wm": 0, "actor": 0}
    try:
        runner = LocalBundleRunner(
            manifest=manifest, env=env, cfg=rcfg, task_id=task, device="cpu"
        )
        wm_before, actor_before = (
            parameters(runner._dynamics),
            parameters(runner.policy_learner.actor),
        )
        wm_update, policy_update = (
            runner.world_model_learner.update,
            runner.policy_learner.update,
        )

        def tracked_wm(data):
            result = wm_update(data)
            updates["wm"] += 1
            return result

        def tracked_policy():
            before = parameters(runner._dynamics)
            result = policy_update()
            assert not changed(before, runner._dynamics), (
                "WM changed during actor update"
            )
            assert all(not p.requires_grad for p in runner._dynamics.parameters())
            updates["actor"] += 1
            return result

        runner.world_model_learner.update = tracked_wm
        runner.policy_learner.update = tracked_policy
        runner.run()
        assert updates["wm"] > 0 and updates["actor"] > 0
        assert changed(wm_before, runner._dynamics) and changed(
            actor_before, runner.policy_learner.actor
        )
        bundle = max((output / "logs" / task).iterdir(), key=lambda p: p.name)
        equal_tree(
            torch.load(bundle / "world_model.pt", weights_only=True),
            runner._wm.bundle(),
        )
        equal_tree(
            torch.load(bundle / "actor.pt", weights_only=True),
            runner.policy_learner.actor.state_dict(),
        )
        equal_tree(
            torch.load(bundle / "critic.pt", weights_only=True),
            runner.policy_learner.critic.state_dict(),
        )
        restored, restored_runner = read_bundle(bundle, task)
        assert restored.decimation == 8 and restored.sim.mujoco.timestep == 0.0025
        with torch.inference_mode():
            wm, actor = load_models(bundle, restored_runner, env, "cpu")
        equal_tree(wm.bundle(), runner._wm.bundle())
        equal_tree(actor.state_dict(), runner.policy_learner.actor.state_dict())
    finally:
        env.close()
    play(
        task,
        PlayConfig(
            motion_file=motion,
            checkpoint_dir=bundle,
            device="cpu",
            viewer="headless",
            num_steps_rollout=4,
        ),
    )
    assert source.read_bytes() == original
    result = {
        "task": task,
        "source": str(source),
        "bundle": str(bundle),
        "updates": updates,
        "final_weights_exact": True,
        "wm_frozen_during_actor": True,
        "play_steps": 4,
        "timestep": 0.0025,
        "decimation": 8,
        "source_unchanged": True,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS", json.dumps(result))


if __name__ == "__main__":
    main()
