import copy
import sys
from types import SimpleNamespace

import pytest
import torch
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg

from source.tasks.supermimic.checkpoint_config import (
    make_manifest,
    validate_manifest,
)
from source.tasks.supermimic.config import TASKS, register_tasks
from source.tasks.supermimic.config.k1_rev1.base_env_cfg import make_env_cfg


@pytest.fixture
def manifest():
    from mjlab.tasks.registry import list_tasks

    if not set(TASKS).issubset(list_tasks()):
        register_tasks()
    cfg = make_env_cfg()
    cfg.sim.mujoco.timestep = 0.0025
    cfg.decimation = 8
    cfg.commands["motion"].sampling_mode = "external"
    return make_manifest(
        "Cyclo-SuperMimic-GenCoord-K1-Rev1-001",
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
def test_train_mismatch_fails_before_device(
    monkeypatch, manifest, rep, override
):
    from source.tasks.supermimic import train

    monkeypatch.setattr(
        train, "setup_device", lambda _: pytest.fail("device setup reached")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            f"Cyclo-SuperMimic-{rep}-K1-Rev1-001",
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
            id="local",
            settings=SimpleNamespace(mode=mode),
            log_artifact=logged.append,
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
@pytest.mark.parametrize("local_archive", [False, True])
@pytest.mark.parametrize(
    "local_loop,request_upload,mode",
    [(False, True, "online"), (True, True, "online"),
     (True, False, "online"), (True, True, "offline"),
     (True, True, "disabled")],
)
def test_real_runner_loop_saves_final_parameters(
    monkeypatch,
    tmp_path,
    iterations,
    interval,
    upload_failure,
    local_archive,
    local_loop,
    request_upload,
    mode,
):
    from supermimic.runners import supermimic_runner as module

    actor = torch.nn.Linear(1, 1, bias=False)
    critic = torch.nn.Linear(1, 1, bias=False)
    dynamics = torch.nn.Linear(1, 1, bias=False)
    for model in (actor, critic, dynamics):
        model.weight.data.zero_()
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=0.0)
        for name, model in (
            ("actor", actor),
            ("critic", critic),
            ("world_model", dynamics),
        )
    }

    def update_optimizer(name, model):
        model.weight.grad = torch.ones_like(model.weight)
        optimizers[name].step()

    count = {"wm": 0, "policy": 0}

    def update_wm(_):
        update_optimizer("world_model", dynamics)
        dynamics.weight.data.add_(1)
        count["wm"] += 1
        return {}

    def update_policy(starts):
        assert len(list(starts)) == cfg.policy.num_batches
        update_optimizer("actor", actor)
        update_optimizer("critic", critic)
        actor.weight.data.add_(1)
        critic.weight.data.add_(1)
        count["policy"] += 1
        return {}

    buffer = SimpleNamespace(
        num_transitions=0, sample=lambda horizon, batch: torch.zeros(batch, horizon)
    )

    def step():
        buffer.num_transitions += 1
        return 1

    cfg = SimpleNamespace(
        max_iterations=iterations,
        save_interval=interval,
        log_interval=100,
        wm=SimpleNamespace(num_batches=1, batch_size=1, horizon=8),
        policy=SimpleNamespace(num_batches=2, batch_size=1),
        history_length=1,
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
        checkpoint_extra_files=lambda _: [],
        step=step,
        replay_buffer=buffer,
        world_model_learner=SimpleNamespace(
            update=update_wm, optimizers=[optimizers["world_model"]]
        ),
        policy_learner=SimpleNamespace(
            update=update_policy,
            actor=actor,
            critic=critic,
            actor_optimizers=[optimizers["actor"]],
            critic_optimizers=[optimizers["critic"]],
        ),
        _wm=SimpleNamespace(
            bundle=lambda: copy.deepcopy(dynamics.state_dict())
        ),
        _representation=SimpleNamespace(actor_obs_mode="rep_state"),
        logger=SimpleNamespace(log=lambda **kw: None),
    )
    if local_archive:
        from source.tasks.supermimic.checkpoint import LocalBundleRunner

        local = LocalBundleRunner.__new__(LocalBundleRunner)
        local.__dict__.update(vars(runner))
        del local.step
        del local.checkpoint_extra_files
        local._checkpoint_step = -1
        local.manifest = {"task_id": "test"}
        local._deployment_files = []
        monkeypatch.setattr(
            module.SuperMimicRunner, "step", lambda self: step()
        )
        runner = local
    logged = []
    if request_upload:
        monkeypatch.setenv("CYCLO_WANDB_UPLOAD_MODELS", "1")
    else:
        monkeypatch.delenv("CYCLO_WANDB_UPLOAD_MODELS", raising=False)
    monkeypatch.setattr(
        module.wandb, "run",
        SimpleNamespace(name="local", settings=SimpleNamespace(mode=mode)),
    )

    def log(files):
        logged.append(torch.load(files[0], weights_only=True)["weight"].item())
        if upload_failure and len(logged) == 2:
            raise RuntimeError("final artifact failed")

    monkeypatch.setattr(module, "log_model_artifact", log)
    from source.tasks.supermimic import training_loop

    monkeypatch.setattr(training_loop, "log_model_artifact", log)
    run = (
        training_loop.run_training_loop
        if local_loop
        else module.SuperMimicRunner.run
    )
    should_upload = request_upload and mode == "online"
    if upload_failure and should_upload:
        with pytest.raises(RuntimeError, match="final artifact failed"):
            run(runner)
    else:
        run(runner)
    assert cleanup == [True]
    for name, model in [
        ("world_model", dynamics),
        ("actor", actor),
        ("critic", critic),
    ]:
        saved = torch.load(tmp_path / (name + ".pt"), weights_only=True)
        assert torch.equal(saved["weight"], model.weight)
    if local_archive:
        from source.tasks.supermimic.checkpoint_archive import load_checkpoint

        snapshot = load_checkpoint(
            tmp_path / "checkpoints" / f"model_{iterations - 1}.pt"
        )
        assert snapshot.optimizers is None
        last = load_checkpoint(tmp_path)
        assert last.source == tmp_path / "last.pt"
        assert last.step == iterations - 1
        for state, model in [
            (last.actor, actor),
            (last.critic, critic),
            (last.world_model, dynamics),
        ]:
            assert torch.equal(state["weight"], model.weight)
        first = load_checkpoint(tmp_path / "checkpoints/model_0.pt")
        assert first.actor["weight"].item() == 0
        assert first.critic["weight"].item() == 0
        assert first.optimizers is None
        assert set(last.optimizers) == {"actor", "critic", "world_model"}
        for name, optimizer in optimizers.items():
            saved = last.optimizers[name][0]
            expected = optimizer.state_dict()
            assert saved["param_groups"] == expected["param_groups"]
            for parameter, state in expected["state"].items():
                for key, value in state.items():
                    assert torch.equal(saved["state"][parameter][key], value)
    if not should_upload:
        assert logged == []
        return
    assert logged[-1] == count["wm"]
    if iterations == 16:
        assert count == {"wm": 9, "policy": 9}
        assert logged == [0, 9]
    else:
        assert len(logged) == 2  # final iteration also lies on an interval


@pytest.mark.parametrize(
    "optimizer_type", [torch.optim.Adam, torch.optim.Muon]
)
def test_optimizer_snapshot_restores_next_update(tmp_path, optimizer_type):
    from source.tasks.supermimic.checkpoint_archive import (
        load_checkpoint,
        save_checkpoint,
    )

    model = torch.nn.Linear(3, 3, bias=False)
    optimizer = optimizer_type(model.parameters(), lr=0.001)
    model.weight.grad = torch.full_like(model.weight, 0.5)
    optimizer.step()
    path = save_checkpoint(
        tmp_path,
        step=1000,
        manifest={},
        actor=model.state_dict(),
        world_model={},
        optimizers={"actor": [optimizer.state_dict()]},
    )
    assert load_checkpoint(path).optimizers is None
    checkpoint = load_checkpoint(tmp_path / "last.pt")
    restored = torch.nn.Linear(3, 3, bias=False)
    restored.load_state_dict(checkpoint.actor)
    restored_optimizer = optimizer_type(restored.parameters(), lr=0.1)
    restored_optimizer.load_state_dict(checkpoint.optimizers["actor"][0])

    gradient = torch.arange(9, dtype=model.weight.dtype).reshape(3, 3) / 10
    model.weight.grad = gradient.clone()
    restored.weight.grad = gradient.clone()
    optimizer.step()
    restored_optimizer.step()
    torch.testing.assert_close(model.weight, restored.weight, rtol=0, atol=0)


@pytest.mark.parametrize(
    "variant", ["clipping", "subset", "reordered", "entity"]
)
def test_unsupported_native_action_contract_rejected(variant, monkeypatch):
    from mjlab.entity import Entity

    from source.tasks.supermimic.checkpoint_config import robot_contract

    cfg = make_env_cfg()
    action = cfg.actions["joint_pos"]
    if variant == "clipping":
        action.clip = {".*": (-1.0, 1.0)}
    elif variant == "subset":
        action.actuator_names = ("left_hip_pitch_joint",)
    elif variant == "entity":
        action.entity_name = "different_robot"
    else:
        resolve = Entity.find_joints_by_actuator_names

        def reverse(entity, patterns):
            ids, names = resolve(entity, patterns)
            return ids[::-1], names[::-1]

        monkeypatch.setattr(Entity, "find_joints_by_actuator_names", reverse)
    with pytest.raises(ValueError, match="clipping|joint order"):
        robot_contract(cfg)


def test_latest_saved_bundle_skips_incomplete_and_other_tasks(tmp_path):
    import json
    from source.tasks.supermimic.play import latest_checkpoint_directory

    task = "Cyclo-SuperMimic-GenCoord-K1-Rev1-001"

    def run(task_dir, timestamp, saved_task, *, complete=True):
        directory = tmp_path / task_dir / timestamp
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps({"task_id": saved_task})
        )
        (directory / "actor.pt").touch()
        if complete:
            (directory / "world_model.pt").touch()
        return directory

    expected = run(task, "2026-11-17_13-00-00", task)
    run(task, "2026-11-17_14-00-00", task, complete=False)
    run(task, "2026-11-17_15-00-00", "another-task")
    assert latest_checkpoint_directory(tmp_path, task) == expected
    with pytest.raises(ValueError, match="No saved checkpoint bundle"):
        latest_checkpoint_directory(tmp_path, "missing-task")


def test_play_resolves_latest_only_without_explicit_checkpoint(
    tmp_path, monkeypatch
):
    from source.tasks.supermimic import play

    task = "Cyclo-SuperMimic-GenCoord-K1-Rev1-001"
    directory = tmp_path / "saved"
    selected = []
    monkeypatch.setattr(
        play, "latest_checkpoint_directory", lambda root, name: directory
    )

    class BeforeLoading(Exception):
        pass

    def load_checkpoint(path):
        selected.append((path, task))
        raise BeforeLoading

    monkeypatch.setattr(play, "load_checkpoint", load_checkpoint)
    with pytest.raises(BeforeLoading):
        play.play(task, play.PlayConfig(motion_file="reference.npz"))
    assert selected == [(directory, task)]
    explicit = tmp_path / "explicit"
    monkeypatch.setattr(
        play,
        "latest_checkpoint_directory",
        lambda *args: pytest.fail("unexpected search"),
    )
    with pytest.raises(BeforeLoading):
        play.play(
            task,
            play.PlayConfig(
                motion_file="reference.npz", checkpoint_dir=explicit
            ),
        )
    assert selected[-1] == (explicit, task)
    with pytest.raises(ValueError, match="Choose only one"):
        play.play(
            task,
            play.PlayConfig(
                motion_file="reference.npz",
                checkpoint_dir=explicit,
                wandb_path="run",
            ),
        )


def test_archive_retention_and_preserves_existing_file_on_failure(
    tmp_path, monkeypatch
):
    from source.tasks.supermimic.checkpoint_archive import save_checkpoint

    for step in range(0, 16000, 1000):
        save_checkpoint(
            tmp_path, step=step, manifest={}, actor={}, world_model={}
        )
    directory = tmp_path / "checkpoints"
    assert {p.name for p in directory.glob("*.pt")} == {
        f"model_{step}.pt"
        for step in (0, 10000, 11000, 12000, 13000, 14000, 15000)
    }
    original = (directory / "model_15000.pt").read_bytes()
    latest = (tmp_path / "last.pt").read_bytes()

    def fail(saved, stream):
        stream.write(b"partial write")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(
            tmp_path, step=15000, manifest={}, actor={}, world_model={}
        )
    assert (directory / "model_15000.pt").read_bytes() == original
    assert (tmp_path / "last.pt").read_bytes() == latest
    assert not list(directory.glob("*.tmp"))
    assert not list(tmp_path.glob("*.tmp"))


def test_play_accepts_archive_and_rejects_conflicting_sources(
    tmp_path, monkeypatch
):
    from source.tasks.supermimic import play

    selected = []

    class Selected(Exception):
        pass

    def load(source):
        selected.append(source)
        raise Selected

    monkeypatch.setattr(play, "load_checkpoint", load)
    monkeypatch.setattr(
        play,
        "latest_checkpoint_directory",
        lambda *a: pytest.fail("searched latest"),
    )
    path = tmp_path / "model_1000.pt"
    with pytest.raises(Selected):
        play.play(
            "task", play.PlayConfig(motion_file="unused", checkpoint=path)
        )
    assert selected == [path]
    for other in ({"checkpoint_dir": tmp_path}, {"wandb_path": "run"}):
        with pytest.raises(ValueError, match="Choose only one"):
            play.play(
                "task",
                play.PlayConfig(
                    motion_file="unused", checkpoint=path, **other
                ),
            )


def test_resume_cli_uses_saved_settings_and_restores_before_run(
    monkeypatch, tmp_path, manifest
):
    from source.tasks.supermimic import train
    from source.tasks.supermimic.checkpoint_archive import save_checkpoint

    save_checkpoint(
        tmp_path,
        step=1000,
        manifest=manifest,
        actor={},
        critic={},
        world_model={},
    )
    task = manifest["task_id"]
    calls = []
    monkeypatch.setattr(train, "setup_device", lambda _: "cpu")
    env = SimpleNamespace(close=lambda: calls.append("close"))
    monkeypatch.setattr(train, "create_env", lambda cfg, count, device: env)

    def build(**kwargs):
        assert (
            kwargs["cfg"].actor_obs == "rep_state"
        )  # current default is proprio
        assert (
            kwargs["manifest"]["environment_options"]
            == manifest["environment_options"]
        )
        assert kwargs["manifest"]["num_envs"] == 8
        assert kwargs["cfg"].max_iterations == 2000
        return SimpleNamespace(run=lambda: calls.append("run"))

    monkeypatch.setattr(train, "SuperMimicRunner", build)
    monkeypatch.setattr(
        train, "restore_training_state", lambda *a: calls.append("restore")
    )
    train.main(
        [
            task,
            "--agent.resume",
            "True",
            "--agent.load-checkpoint",
            str(tmp_path / "last.pt"),
            "--motion-file",
            "unused.npz",
            "--runner.max-iterations",
            "2000",
            "--num-envs",
            "8",
        ]
    )
    assert calls == ["restore", "run", "close"]


def test_resume_configuration_failure_precedes_gpu(
    monkeypatch, tmp_path, manifest
):
    from source.tasks.supermimic import train
    from source.tasks.supermimic.checkpoint_archive import save_checkpoint

    save_checkpoint(
        tmp_path,
        step=1000,
        manifest=manifest,
        actor={},
        critic={},
        world_model={},
    )
    monkeypatch.setattr(
        train, "setup_device", lambda _: pytest.fail("GPU setup reached")
    )
    with pytest.raises(ValueError, match="configuration differs"):
        train.main(
            [
                manifest["task_id"],
                "--agent.resume",
                "True",
                "--agent.load-checkpoint",
                str(tmp_path / "last.pt"),
                "--motion-file",
                "unused.npz",
                "--runner.history-length",
                "6",
            ]
        )


def test_cross_task_resume_keeps_target_clips_and_checks_contract(
    manifest, monkeypatch
):
    from source.tasks.supermimic import checkpoint_config

    source = manifest["task_id"]
    target = source + "-new-clips"
    monkeypatch.setitem(TASKS, target, TASKS[source])
    load = checkpoint_config.load_env_cfg

    def target_config(task):
        assert task == target
        cfg = load(source)
        cfg.commands["motion"].motion_file = "new-clips.npz"
        return cfg

    monkeypatch.setattr(checkpoint_config, "load_env_cfg", target_config)
    cfg, _ = validate_manifest(manifest, target, for_resume=True)
    assert cfg.commands["motion"].motion_file == "new-clips.npz"
    with pytest.raises(ValueError, match="identity"):
        validate_manifest(manifest, target)
    with pytest.raises(ValueError, match="representation"):
        validate_manifest(
            manifest, source.replace("GenCoord", "MaxCoord"), for_resume=True
        )
    manifest["robot"]["action_joint_order"].reverse()
    with pytest.raises(ValueError, match="joint-order"):
        validate_manifest(manifest, target, for_resume=True)


@pytest.mark.parametrize(
    "file", ["last.pt", "checkpoints/model_1000.pt", "world_model.pt"]
)
def test_local_world_model_accepts_target_policy_changes(
    manifest, tmp_path, file
):
    import json
    from source.tasks.supermimic.checkpoint_archive import (
        load_world_model,
        save_checkpoint,
    )

    weights = {"dynamics": {"w": torch.arange(3)}, "sim_context": {}}
    save_checkpoint(
        tmp_path, step=1000, manifest=manifest, actor={}, world_model=weights
    )
    torch.save(weights, tmp_path / "world_model.pt")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    target = copy.deepcopy(manifest)
    target["task_id"] += "-other"
    target["runner_cfg"]["slot_mix"] = {"velocity": 1.0, "motion": 0.0}
    assert torch.equal(
        load_world_model(tmp_path / file, target)["dynamics"]["w"],
        weights["dynamics"]["w"],
    )
    target["environment"]["decimation"] += 1
    with pytest.raises(ValueError, match="dynamics configuration"):
        load_world_model(tmp_path / file, target)


@pytest.mark.parametrize("retired_task", [False, True])
@pytest.mark.parametrize("source,target", [(1, 2), (2, 1)])
@pytest.mark.parametrize("representation", ["GenCoord", "MaxCoord"])
@pytest.mark.filterwarnings("error::UserWarning")
def test_cross_revision_play_loads_once_and_uses_target_robot(
    tmp_path, monkeypatch, source, target, representation, retired_task
):
    from unittest.mock import patch
    from mjlab.tasks.registry import load_env_cfg
    from source.tasks.supermimic import play

    task = f"Cyclo-SuperMimic-{representation}-K1-Rev{source}-001"
    target_task = task.replace(f"Rev{source}", f"Rev{target}")
    cfg = load_env_cfg(task)
    cfg.commands["motion"].sampling_mode = "external"
    cfg.sim.mujoco.timestep = 0.0025
    cfg.decimation = 8
    manifest = make_manifest(task, cfg, SuperMimicRunnerCfg(representation=representation.lower()),
                             ["--env.sim.mujoco.timestep", "0.0025", "--env.decimation", "8"])
    if retired_task:
        manifest["task_id"] = task.replace("-001", "-retired")
    snapshot = tmp_path / "model_1.pt"
    torch.save(dict(format="cyclo.supermimic.checkpoint.v1", manifest=manifest, step=1,
                    actor={}, world_model={}), snapshot)

    class EnvironmentSelected(Exception):
        pass

    def check_environment(env_cfg, count, device):
        expected = load_env_cfg(target_task, play=True)
        assert env_cfg.scene.entities["robot"].spec_fn is expected.scene.entities["robot"].spec_fn
        assert env_cfg.events["foot_friction"] == expected.events["foot_friction"]
        assert env_cfg.sim.mujoco.timestep == expected.sim.mujoco.timestep
        assert env_cfg.decimation == expected.decimation
        assert env_cfg.episode_length_s == expected.episode_length_s
        assert env_cfg.commands["motion"].motion_file == "reference.npz"
        assert env_cfg.commands["motion"].sampling_mode == "start"
        assert env_cfg.commands["motion"].pose_range == env_cfg.commands["motion"].velocity_range == {}
        raise EnvironmentSelected

    monkeypatch.setattr(play, "setup_device", lambda _: "cpu")
    monkeypatch.setattr(play, "create_env", check_environment)
    with patch.object(torch, "load", wraps=torch.load) as loads:
        with pytest.raises(EnvironmentSelected):
            play.play(target_task, play.PlayConfig("reference.npz", checkpoint=snapshot, export_policy=retired_task))
        loads.assert_called_once()
