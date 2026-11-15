import copy
import sys
from types import SimpleNamespace

import pytest
import torch
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg

from source.tasks.supermimic.checkpoint import make_manifest, validate_manifest
from source.tasks.supermimic.tasks import TASKS, k1_env_cfg, register_tasks


@pytest.fixture
def manifest():
    from mjlab.tasks.registry import list_tasks

    if not set(TASKS).issubset(list_tasks()):
        register_tasks()
    cfg = k1_env_cfg()
    cfg.sim.mujoco.timestep = 0.0025
    cfg.decimation = 8
    cfg.commands["motion"].sampling_mode = "external"
    return make_manifest(
        "Cyclo-SuperMimic-GenCoord-K1-Rev1",
        cfg,
        SuperMimicRunnerCfg(),
        ["--env.sim.mujoco.timestep", "0.0025", "--env.decimation", "8"],
    )


def test_manifest_restores_exact_substeps(manifest):
    cfg, rcfg = validate_manifest(manifest, manifest["task_id"])
    assert cfg.sim.mujoco.timestep == 0.0025 and cfg.decimation == 8
    assert rcfg.actor_obs == "rep_state"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 99),
        ("task_id", "wrong"),
        ("actor_obs", "proprio"),
        ("representation", "maxcoord"),
        ("environment_options", []),
        ("robot", {}),
        ("environment", {}),
    ],
)
def test_manifest_rejects_incompatible_contract(manifest, field, value):
    task = manifest["task_id"]
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_manifest(manifest, task)


@pytest.mark.parametrize(
    "rep,override", [("GenCoord", "maxcoord"), ("MaxCoord", "gencoord")]
)
def test_train_mismatch_fails_before_device(monkeypatch, manifest, rep, override):
    from source.tasks.supermimic import train

    monkeypatch.setattr(
        train, "setup_device", lambda _: pytest.fail("device setup reached")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            f"Cyclo-SuperMimic-{rep}-K1-Rev1",
            "--motion-file",
            "unused.npz",
            "--runner.representation",
            override,
        ],
    )
    with pytest.raises(ValueError, match="task"):
        train.main()


@pytest.mark.parametrize("mode", ["offline", "disabled", "dryrun"])
def test_offline_artifacts_never_call_remote_api(monkeypatch, mode):
    from supermimic.utils import wandb_artifacts as artifacts

    logged = []
    monkeypatch.setattr(
        artifacts.wandb,
        "run",
        SimpleNamespace(
            id="local", settings=SimpleNamespace(mode=mode), log_artifact=logged.append
        ),
    )
    monkeypatch.setattr(
        artifacts.wandb,
        "Artifact",
        lambda *a, **kw: SimpleNamespace(add_file=lambda _: None),
    )
    monkeypatch.setattr(
        artifacts.wandb, "Api", lambda: pytest.fail("remote API called")
    )
    artifacts.log_model_artifact(["actor.pt"])
    assert len(logged) == 1


@pytest.mark.parametrize(
    "iterations,interval,upload_failure",
    [(16, 1000, False), (3, 2, False), (16, 1000, True)],
)
def test_real_runner_loop_saves_final_parameters(
    monkeypatch, tmp_path, iterations, interval, upload_failure
):
    from supermimic.runners import supermimic_runner as module

    actor = torch.nn.Linear(1, 1, bias=False)
    critic = torch.nn.Linear(1, 1, bias=False)
    dynamics = torch.nn.Linear(1, 1, bias=False)
    for model in (actor, critic, dynamics):
        model.weight.data.zero_()
    count = {"wm": 0, "policy": 0}

    def update_wm(_):
        dynamics.weight.data.add_(1)
        count["wm"] += 1
        return {}

    def update_policy():
        actor.weight.data.add_(1)
        critic.weight.data.add_(1)
        count["policy"] += 1
        return {}

    buffer = SimpleNamespace(num_transitions=0, sample=lambda *a: None)

    def step():
        buffer.num_transitions += 1
        return 1

    cfg = SimpleNamespace(
        max_iterations=iterations,
        save_interval=interval,
        log_interval=100,
        wm=SimpleNamespace(num_batches=1, batch_size=1, horizon=8),
        replay_ratio=8,
    )
    cleanup = []
    runner = SimpleNamespace(
        cfg=cfg,
        env=SimpleNamespace(num_envs=1, reset=lambda: None),
        _create_log_dir=lambda: tmp_path,
        _init_wandb=lambda _: None,
        _seed_sim_history=lambda: None,
        _cleanup=lambda: cleanup.append(True),
        step=step,
        replay_buffer=buffer,
        world_model_learner=SimpleNamespace(update=update_wm),
        policy_learner=SimpleNamespace(
            update=update_policy, actor=actor, critic=critic
        ),
        _wm=SimpleNamespace(bundle=lambda: copy.deepcopy(dynamics.state_dict())),
        _representation=SimpleNamespace(actor_obs_mode="rep_state"),
        logger=SimpleNamespace(log=lambda **kw: None),
    )
    logged = []
    monkeypatch.setattr(module.wandb, "run", SimpleNamespace(name="local"))

    def log(files):
        logged.append(torch.load(files[0], weights_only=True)["weight"].item())
        if upload_failure and len(logged) == 2:
            raise RuntimeError("final artifact failed")

    monkeypatch.setattr(module, "log_model_artifact", log)
    if upload_failure:
        with pytest.raises(RuntimeError, match="final artifact failed"):
            module.SuperMimicRunner.run(runner)
    else:
        module.SuperMimicRunner.run(runner)
    assert cleanup == [True]
    for name, model in [
        ("world_model", dynamics),
        ("actor", actor),
        ("critic", critic),
    ]:
        saved = torch.load(tmp_path / (name + ".pt"), weights_only=True)
        assert torch.equal(saved["weight"], model.weight)
    assert logged[-1] == count["wm"]
    if iterations == 16:
        assert count == {"wm": 9, "policy": 9}
        assert logged == [0, 9]
    else:
        assert len(logged) == 2  # final iteration also lies on an interval
