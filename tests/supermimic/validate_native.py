"""
Opt-in bounded native acceptance of the actual learners and local playback.

Run from the checkout: python tests/supermimic/validate_native.py GenCoord
dance1 /tmp/output
W&B disabled mode is enforced for this process. CPU/eager is the default;
CUDA and compiled/captured updates require explicit options. No global logging
preferences are changed.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Direct script execution needs the checkout on sys.path before local imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from supermimic.runners.supermimic_runner import (  # noqa: E402
    SuperMimicRunnerCfg,
)
from supermimic.scripts.train_utils import (  # noqa: E402
    create_env,
    setup_device,
)

from source.tasks.supermimic import REPO_ROOT  # noqa: E402
from source.tasks.supermimic.training.checkpoint.archive import load_checkpoint  # noqa: E402
from source.tasks.supermimic.training.runner import LocalBundleRunner  # noqa: E402
from source.tasks.supermimic.training.checkpoint.manifest import (  # noqa: E402
    make_manifest,
    read_bundle,
)
from source.tasks.supermimic.reference.motion import prepare_motion_files  # noqa: E402
from source.tasks.supermimic.play import (  # noqa: E402
    PlayConfig,
    load_models,
    play,
)
from source.tasks.supermimic.config import register_tasks  # noqa: E402
from source.tasks.supermimic.config.k1_rev1.base_env_cfg import (  # noqa: E402
    make_env_cfg,
)


def parameters(module):
    return [p.detach().clone() for p in module.parameters()]


def changed(before, module):
    return any(
        not torch.equal(a, b)
        for a, b in zip(before, module.parameters(), strict=True)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("representation", choices=("GenCoord", "MaxCoord"))
    parser.add_argument("dance", choices=("dance1", "dance2"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument(
        "--learner-execution", choices=("eager", "compiled"), default="eager"
    )
    parser.add_argument(
        "--actor-obs", choices=("rep_state", "proprio"), default="rep_state"
    )
    parser.add_argument(
        "--history-length", type=int, choices=(1, 2), default=1
    )
    args = parser.parse_args()
    if args.history_length > 1 and args.actor_obs != "proprio":
        parser.error(
            "Multi-frame history requires explicit proprio observations"
        )
    if args.learner_execution == "compiled" and args.device == "cpu":
        parser.error("This bounded compiled/captured acceptance requires CUDA")
    representation, dance = args.representation, args.dance
    output = args.output.resolve()
    os.environ["TORCHDYNAMO_DISABLE"] = (
        "1" if args.learner_execution == "eager" else "0"
    )
    import torch._dynamo

    torch._dynamo.config.disable = args.learner_execution == "eager"
    output.mkdir(parents=True, exist_ok=True)
    # This validation entry point must never depend on ambient user logging
    # mode.
    os.environ["WANDB_MODE"] = "disabled"
    for variable in (
        "WANDB_DIR",
        "WANDB_DATA_DIR",
        "WANDB_CACHE_DIR",
        "WANDB_CONFIG_DIR",
    ):
        directory = output / variable.lower()
        directory.mkdir(exist_ok=True)
        os.environ[variable] = str(directory)
    assert os.environ["WANDB_MODE"] == "disabled"
    print(f"[INFO] Validation logging: disabled; evidence directory: {output}")
    os.chdir(output)
    torch.set_num_threads(2)
    torch.manual_seed(7)
    setup_device(args.device)
    register_tasks()
    task = f"Cyclo-SuperMimic-{representation}-K1-Rev1-001"
    cfg = make_env_cfg(representation.lower())
    # Exercise the nondefault substep contract in actual training and playback.
    cfg.sim.mujoco.timestep, cfg.decimation = 0.0025, 8
    source = REPO_ROOT / f"source/assets/motions/K1_rev1/{dance}/{dance}.npz"
    original = source.read_bytes()
    motion = prepare_motion_files(
        str(source),
        cfg.scene.entities["robot"].spec_fn().compile(),
        output / "adapter",
    )
    cfg.commands["motion"].motion_file = motion
    cfg.commands["motion"].sampling_mode = "external"
    rcfg = SuperMimicRunnerCfg(
        representation=representation.lower(),
        actor_obs=args.actor_obs,
        history_length=args.history_length,
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
    rcfg.policy.horizon, rcfg.policy.batch_size, rcfg.policy.num_batches = (
        2,
        2,
        1,
    )
    manifest = make_manifest(
        task,
        cfg,
        rcfg,
    )
    env = create_env(cfg, 2, args.device)
    updates = {"wm": 0, "actor": 0}
    update_seconds = {"wm": [], "actor": []}
    try:
        runner = LocalBundleRunner(
            manifest=manifest,
            env=env,
            cfg=rcfg,
            task_id=task,
            device=args.device,
        )
        if args.learner_execution == "eager":
            # Test-only warmup extension keeps official updates eager on CUDA.
            # The compiled mode leaves the production capture schedule
            # unchanged.
            runner.world_model_learner._captured_update._warmup = (
                rcfg.max_iterations + 1
            )
            runner.policy_learner._captured_update._warmup = (
                rcfg.max_iterations + 1
            )
        wm_before, actor_before = (
            parameters(runner._dynamics),
            parameters(runner.policy_learner.actor),
        )
        wm_muon = [
            (p, p.detach().clone())
            for optimizer in runner.world_model_learner.optimizers
            if isinstance(optimizer, torch.optim.Muon)
            for group in optimizer.param_groups
            for p in group["params"]
        ]
        actor_muon = [
            (p, p.detach().clone())
            for optimizer in runner.policy_learner.actor_optimizers
            if isinstance(optimizer, torch.optim.Muon)
            for group in optimizer.param_groups
            for p in group["params"]
        ]
        assert wm_muon and actor_muon
        wm_update, policy_update = (
            runner.world_model_learner.update,
            runner.policy_learner.update,
        )

        def tracked_wm(data):
            start = time.perf_counter()
            print(f"[INFO] WM update {updates['wm'] + 1} starting", flush=True)
            result = wm_update(data)
            update_seconds["wm"].append(time.perf_counter() - start)
            print(
                "[INFO] WM update completed in"
                f" {update_seconds['wm'][-1]:.3f}s",
                flush=True,
            )
            updates["wm"] += 1
            return result

        def tracked_policy(starts):
            before = parameters(runner._dynamics)
            start = time.perf_counter()
            print(
                f"[INFO] Actor update {updates['actor'] + 1} starting",
                flush=True,
            )
            result = policy_update(starts)
            update_seconds["actor"].append(time.perf_counter() - start)
            print(
                "[INFO] Actor update completed in"
                f" {update_seconds['actor'][-1]:.3f}s",
                flush=True,
            )
            assert not changed(
                before, runner._dynamics
            ), "WM changed during actor update"
            assert all(
                not p.requires_grad for p in runner._dynamics.parameters()
            )
            updates["actor"] += 1
            return result

        runner.world_model_learner.update = tracked_wm
        runner.policy_learner.update = tracked_policy
        import wandb

        initialize_wandb = runner._init_wandb
        logging_modes = []

        def disabled_wandb(log_dir):
            initialize_wandb(log_dir)
            assert (
                wandb.run is not None and wandb.run.settings.mode == "disabled"
            )
            logging_modes.append(wandb.run.settings.mode)

        runner._init_wandb = disabled_wandb
        runner.run()
        assert logging_modes == ["disabled"]
        captured = {
            "wm": (
                runner.world_model_learner._captured_update._graph is not None
            ),
            "actor": runner.policy_learner._captured_update._graph is not None,
        }
        if args.learner_execution == "compiled":
            assert all(
                captured.values()
            ), "Production learner CUDA graphs were not captured"
            assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0
        else:
            assert not any(captured.values())
        assert updates["wm"] > 0 and updates["actor"] > 0
        assert changed(wm_before, runner._dynamics) and changed(
            actor_before, runner.policy_learner.actor
        )
        assert any(not torch.equal(p, old) for p, old in wm_muon)
        assert any(not torch.equal(p, old) for p, old in actor_muon)
        bundle = max((output / "logs" / "supermimic" / task).iterdir(), key=lambda p: p.name)
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
        restored, restored_runner, _, _ = read_bundle(bundle, task)
        assert (
            restored.decimation * restored.sim.mujoco.timestep == 0.02
        )
        # Simulate a checkpoint trained with different action and context transforms.
        checkpoint = load_checkpoint(bundle)
        for group, keys in (("dynamics", ("action_scale", "action_offset")),
                            ("sim_context", ("mean", "scale"))):
            for key in keys:
                checkpoint.world_model[group][key].add_(1)
        with torch.inference_mode():
            wm, actor = load_models(checkpoint, restored_runner, env, args.device)
        equal_tree(wm.bundle(), runner._wm.bundle())
        equal_tree(
            actor.state_dict(), runner.policy_learner.actor.state_dict()
        )
    finally:
        env.close()
    play(
        task,
        PlayConfig(
            motion_file=motion,
            checkpoint_dir=bundle,
            device=args.device,
            viewer="headless",
            num_steps_rollout=4,
        ),
    )
    assert source.read_bytes() == original
    result = {
        "task": task,
        "source": str(source),
        "logging_mode": "disabled",
        "device": args.device,
        "learner_execution": args.learner_execution,
        "learner_graphs_captured": captured,
        "compiled_unique_graphs": torch._dynamo.utils.counters["stats"][
            "unique_graphs"
        ],
        "actor_obs": args.actor_obs,
        "history_length": args.history_length,
        "evidence_directory": str(output),
        "bundle": str(bundle),
        "updates": updates,
        "update_seconds": update_seconds,
        "final_weights_exact": True,
        "wm_frozen_during_actor": True,
        "wm_and_actor_muon_matrices_updated": True,
        "play_steps": 4,
        "timestep": 0.0025,
        "decimation": 8,
        "source_unchanged": True,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS", json.dumps(result))


if __name__ == "__main__":
    main()
