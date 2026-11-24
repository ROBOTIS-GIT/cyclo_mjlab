import copy
import sys
from types import SimpleNamespace

import pytest
import torch
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg

from source.tasks.supermimic.training.checkpoint.manifest import (
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
    )


def test_legacy_manifest_uses_current_substeps(manifest):
    manifest.update(schema_version=1, environment_options=["--env.decimation", "1"])
    cfg, rcfg = validate_manifest(manifest, manifest["task_id"])
    assert cfg.decimation == make_env_cfg().decimation
    assert rcfg.actor_obs == "rep_state"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 99),
        ("task_id", "wrong"),
        ("actor_obs", "proprio"),
        ("representation", "maxcoord"),
        ("robot", {}),
        ("environment", {}),
    ],
)
def test_manifest_rejects_incompatible_contract(manifest, field, value):
    task = manifest["task_id"]
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_manifest(manifest, task)


@pytest.mark.filterwarnings("error::UserWarning")
def test_manifest_accepts_changed_physics_for_export(manifest):
    manifest["robot"]["physical_model_sha256"] = "previous-model"
    manifest["environment_options"] = []
    validate_manifest(manifest, manifest["task_id"])


def test_full_state_is_recorded_and_excluded_from_compatibility_checks(manifest, tmp_path):
    from source.tasks.supermimic.training.checkpoint.archive import load_world_model, save_checkpoint
    from source.tasks.supermimic.training.checkpoint.resume import validate_resume_configuration

    saved_obs = manifest["environment"]["observations"]
    assert "terms" in saved_obs["full_state"]
    target = copy.deepcopy(manifest)
    saved_obs["full_state"]["terms"].pop("joint_vel")
    saved_obs.pop("critic")
    checkpoint = SimpleNamespace(manifest=manifest, step=1)
    weights = {"dynamics": {}, "sim_context": {}}
    save_checkpoint(tmp_path, step=1, manifest=manifest, actor={}, world_model=weights)

    validate_resume_configuration(checkpoint, target)
    assert load_world_model(tmp_path, target) == weights
    validate_manifest(manifest, manifest["task_id"])
    saved_obs.pop("actor")
    validate_resume_configuration(checkpoint, target)
    validate_manifest(manifest, manifest["task_id"])  # rep_state bypasses sensor observations.
    manifest["actor_obs"] = manifest["runner_cfg"]["actor_obs"] = "proprio"
    with pytest.raises(ValueError, match="observation configuration"):
        validate_manifest(manifest, manifest["task_id"])
    with pytest.raises(ValueError, match="actor observation"):
        validate_resume_configuration(checkpoint, target)


def test_resume_ignores_only_inactive_term_noise(manifest):
    from source.tasks.supermimic.training.checkpoint.resume import validate_resume_configuration

    manifest["actor_obs"] = manifest["runner_cfg"]["actor_obs"] = "proprio"
    target = copy.deepcopy(manifest)
    saved_actor = manifest["environment"]["observations"]["actor"]
    target_actor = target["environment"]["observations"]["actor"]
    term = saved_actor["terms"]["joint_vel"]
    saved_actor["terms"] = dict(reversed(saved_actor["terms"].items()))
    with pytest.raises(ValueError, match="actor observation"):
        validate_resume_configuration(SimpleNamespace(manifest=manifest, step=1), target)
    saved_actor["terms"] = dict(reversed(saved_actor["terms"].items()))
    term["noise"] = {"n_min": -0.1, "n_max": 0.1}
    checkpoint = SimpleNamespace(manifest=manifest, step=1)
    validate_resume_configuration(checkpoint, target)
    assert term["noise"]["n_max"] == 0.1
    saved_actor["enable_corruption"] = target_actor["enable_corruption"] = True
    with pytest.raises(ValueError, match="actor observation"):
        validate_resume_configuration(checkpoint, target)
    saved_actor["enable_corruption"] = target_actor["enable_corruption"] = False
    term["scale"] = 0.5
    with pytest.raises(ValueError, match="actor observation"):
        validate_resume_configuration(checkpoint, target)


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
        from source.tasks.supermimic.training.runner import LocalBundleRunner

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

    def log(files, run_dir=None):
        logged.append(torch.load(files[0], weights_only=True)["weight"].item())
        if upload_failure and len(logged) == 2:
            raise RuntimeError("final artifact failed")

    monkeypatch.setattr(module, "log_model_artifact", log)
    from source.tasks.supermimic.training import loop as training_loop

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
        from source.tasks.supermimic.training.checkpoint.archive import load_checkpoint

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
    from source.tasks.supermimic.training.checkpoint.archive import (
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

    from source.tasks.supermimic.training.checkpoint.manifest import robot_contract

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
    from source.tasks.supermimic.play import latest_playback_checkpoint

    task = "Cyclo-SuperMimic-GenCoord-K1-Rev1-001"

    def run(task_dir, timestamp, saved_task, *, complete=True):
        directory = tmp_path / task_dir / timestamp
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps({"task_id": saved_task})
        )
        torch.save({}, directory / "actor.pt")
        if complete:
            torch.save({}, directory / "world_model.pt")
        return directory

    expected = run(task, "2026-11-17_13-00-00", task)
    run(task, "2026-11-17_14-00-00", task, complete=False)
    run(task, "2026-11-17_15-00-00", "another-task")
    assert latest_playback_checkpoint(tmp_path, task).source == expected
    from source.tasks.supermimic.training.checkpoint.archive import save_checkpoint
    newer = tmp_path / task / "2026-11-18"
    save_checkpoint(newer, step=15, manifest={"task_id": task}, actor={}, world_model={})
    assert latest_playback_checkpoint(tmp_path, task).step == 15
    (newer / "manifest.json").write_text('{"task_id":"stale"}')
    assert latest_playback_checkpoint(tmp_path, task).source == newer / "last.pt"
    with pytest.raises(ValueError, match="No saved checkpoint bundle"):
        latest_playback_checkpoint(tmp_path, "missing-task")


def test_play_resolves_latest_only_without_explicit_checkpoint(
    tmp_path, monkeypatch
):
    from source.tasks.supermimic import play

    task = "Cyclo-SuperMimic-GenCoord-K1-Rev1-001"
    directory = tmp_path / "saved"
    selected = []
    auto = SimpleNamespace(source=directory)
    monkeypatch.setattr(play, "latest_playback_checkpoint", lambda root, name: auto)
    assert play.resolve_playback_checkpoint(task, play.PlayConfig("reference.npz")) is auto

    class BeforeLoading(Exception):
        pass

    def load_checkpoint(path):
        selected.append((path, task))
        raise BeforeLoading

    monkeypatch.setattr(play, "load_checkpoint", load_checkpoint)
    explicit = tmp_path / "explicit"
    monkeypatch.setattr(
        play,
        "latest_playback_checkpoint",
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
    from source.tasks.supermimic.training.checkpoint.archive import save_checkpoint

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
        "latest_playback_checkpoint",
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
    from source.tasks.supermimic.training.checkpoint.archive import save_checkpoint

    manifest.update(schema_version=1, num_envs=999, environment_options=["--env.decimation", "999"])
    manifest["environment"]["sim"] = {"mujoco": {"timestep": 0.1}}
    manifest["robot"]["xml_sha256"] = "old-xml"
    task = manifest["task_id"]
    manifest["task_id"] = "retired-training-task"
    save_checkpoint(
        tmp_path,
        step=1000,
        manifest=manifest,
        actor={},
        critic={},
        world_model={},
    )
    calls = []
    monkeypatch.setattr(train, "setup_device", lambda _: "cpu")
    env = SimpleNamespace(close=lambda: calls.append("close"))
    def create_current_env(cfg, count, device):
        assert cfg.decimation == make_env_cfg().decimation
        assert count == 8
        return env

    monkeypatch.setattr(train, "create_env", create_current_env)

    def build(**kwargs):
        assert (
            kwargs["cfg"].actor_obs == "rep_state"
        )  # current default is proprio
        assert "environment_options" not in kwargs["manifest"]
        assert "sim" not in kwargs["manifest"]["environment"]
        assert set(kwargs["manifest"]["robot"]) == {"name", "joint_order", "action_joint_order"}
        assert "num_envs" not in kwargs["manifest"]
        assert kwargs["cfg"].wandb_project == "new-project"
        assert kwargs["manifest"]["task_id"] == task
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
            "--runner.wandb-project",
            "new-project",
            "--num-envs",
            "8",
        ]
    )
    assert calls == ["restore", "run", "close"]


@pytest.mark.parametrize("separator", ["-", "_"])
def test_resume_cli_loads_requested_snapshot_not_latest_run(tmp_path, monkeypatch, manifest, separator):
    from source.tasks.supermimic import train
    from source.tasks.supermimic.training.checkpoint.archive import save_checkpoint

    task = manifest["task_id"]
    monkeypatch.setattr(train, "LOG_ROOT", tmp_path)
    for name, step in (("older", 9), ("recent", 19)):
        save_checkpoint(tmp_path / task / name, step=step, manifest=manifest, actor={}, world_model={})
    _, cfg, checkpoint = train.parse_training_configuration([
        task, "--agent.resume", "True", f"--agent.load{separator}run", "older",
        f"--agent.load{separator}checkpoint", "model_9.pt",
    ])
    assert cfg.agent.load_run == "older"
    assert checkpoint.step == 9 and checkpoint.source.name == "model_9.pt"


def test_resume_configuration_failure_precedes_gpu(
    monkeypatch, tmp_path, manifest
):
    from source.tasks.supermimic import train
    from source.tasks.supermimic.training.checkpoint.archive import save_checkpoint

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
    from source.tasks.supermimic.training.checkpoint import manifest as checkpoint_config

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
    from source.tasks.supermimic.training.checkpoint.resume import validate_resume_configuration
    manifest["task_id"] = "retired-training-task"
    cfg, runner_cfg = validate_manifest(manifest, target, for_resume=True)
    destination = make_manifest(target, cfg, runner_cfg)
    validate_resume_configuration(SimpleNamespace(manifest=manifest, step=0), destination)


@pytest.mark.parametrize(
    "file", ["last.pt", "checkpoints/model_1000.pt", "world_model.pt", "wm_100k.pt"]
)
@pytest.mark.filterwarnings("error::UserWarning")
def test_local_world_model_accepts_target_policy_changes(
    manifest, tmp_path, file
):
    import json
    from source.tasks.supermimic.training.checkpoint.archive import (
        load_world_model,
        save_checkpoint,
    )

    weights = {"dynamics": {"w": torch.arange(3)}, "sim_context": {}}
    save_checkpoint(
        tmp_path, step=1000, manifest=manifest, actor={}, world_model=weights
    )
    torch.save(weights, tmp_path / "world_model.pt")
    (tmp_path / "wm_100k.pt").hardlink_to(tmp_path / "world_model.pt")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    target = copy.deepcopy(manifest)
    target["task_id"] += "-other"
    target["runner_cfg"]["slot_mix"] = {"velocity": 1.0, "motion": 0.0}
    assert torch.equal(
        load_world_model(tmp_path / file, target)["dynamics"]["w"],
        weights["dynamics"]["w"],
    )
    target["environment"]["decimation"] = 8
    load_world_model(tmp_path / file, target)
    target["robot"]["joint_order"].reverse()
    load_world_model(tmp_path / file, target)
    bodies = target["environment"]["commands"]["motion"]["body_names"]
    bodies[1], bodies[2] = bodies[2], bodies[1]
    target["environment"]["commands"]["motion"]["anchor_body_name"] = "torso_link"
    load_world_model(tmp_path / file, target)


def test_maxcoord_world_model_rejects_reordered_bodies(manifest, tmp_path):
    from source.tasks.supermimic.training.checkpoint.archive import load_world_model, save_checkpoint

    manifest["representation"] = manifest["runner_cfg"]["representation"] = "maxcoord"
    weights = {"dynamics": {}, "sim_context": {}}
    save_checkpoint(tmp_path, step=1, manifest=manifest, actor={}, world_model=weights)
    target = copy.deepcopy(manifest)
    bodies = target["environment"]["commands"]["motion"]["body_names"]
    bodies[1], bodies[2] = bodies[2], bodies[1]
    with pytest.raises(ValueError, match="body_names"):
        load_world_model(tmp_path, target)


@pytest.mark.parametrize("source_rev,target_rev", [(1, 2), (2, 1)])
@pytest.mark.filterwarnings("error::UserWarning")
def test_training_transfer_across_revisions(tmp_path, source_rev, target_rev):
    from mjlab.tasks.registry import load_env_cfg
    from source.tasks.supermimic.training.checkpoint.archive import load_world_model, save_checkpoint
    from source.tasks.supermimic.training.checkpoint.resume import validate_resume_configuration

    def revision(rev):
        task = f"Cyclo-SuperMimic-GenCoord-K1-Rev{rev}-001"
        cfg = load_env_cfg(task)
        cfg.commands["motion"].sampling_mode = "external"
        return make_manifest(task, cfg, SuperMimicRunnerCfg())

    weights = {"dynamics": {"w": torch.ones(1)}, "sim_context": {}}
    source, target = revision(source_rev), revision(target_rev)
    save_checkpoint(tmp_path, step=1, manifest=source, actor={}, world_model=weights)
    assert load_world_model(tmp_path, target) == weights
    task = target["task_id"]
    cfg, runner = validate_manifest(source, task, for_resume=True)
    resumed = make_manifest(task, cfg, runner)
    assert resumed["robot"] == target["robot"]
    validate_resume_configuration(
        SimpleNamespace(manifest=source, step=1), resumed
    )


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
    manifest = make_manifest(task, cfg, SuperMimicRunnerCfg(representation=representation.lower()))
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


@pytest.mark.parametrize("via_wandb", [False, True])
@pytest.mark.parametrize("cyclo_manifest", [False, True])
def test_transfer_keeps_target_transforms(tmp_path, monkeypatch, manifest, cyclo_manifest, via_wandb):
    import json
    import wandb
    from supermimic.utils import wandb_artifacts
    from source.tasks.supermimic import train

    target = SimpleNamespace(dynamics=torch.nn.Linear(1, 1), sim_context=torch.nn.Module())
    for module, names in ((target.dynamics, ("action_scale", "action_offset")),
                          (target.sim_context, ("mean", "scale"))):
        for name in names:
            module.register_buffer(name, torch.tensor([4.0]))
    weights = {
        group: {k: torch.zeros_like(v) for k, v in getattr(target, group).state_dict().items()}
        for group in ("dynamics", "sim_context")
    }
    torch.save(weights, tmp_path / "world_model.pt")
    if cyclo_manifest:
        del manifest["environment"]["observations"]
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    def load_bundle(bundle):
        for group, state in bundle.items():
            getattr(target, group).load_state_dict(state)
    target.load_bundle = load_bundle
    run_path = "team/project/run"
    monkeypatch.setattr(wandb, "Api", lambda: SimpleNamespace(run=lambda path: path))
    def download(run):
        assert run == run_path
        return tmp_path
    monkeypatch.setattr(wandb_artifacts, "download_model_dir", download)
    monkeypatch.setattr(train, "setup_device", lambda _: "cpu")
    monkeypatch.setattr(train, "create_env", lambda *_: SimpleNamespace(close=lambda: None))
    def build(**kwargs):
        assert kwargs["cfg"].wandb_run_path is None
        assert kwargs["manifest"]["world_model_from"] == (run_path if via_wandb else str(tmp_path / "world_model.pt"))
        return SimpleNamespace(_wm=target, run=lambda: None)
    monkeypatch.setattr(train, "SuperMimicRunner", build)
    source = ["--runner.wandb-run-path", run_path] if via_wandb else ["--world-model-checkpoint", str(tmp_path / "world_model.pt")]
    train.main([manifest["task_id"], "--motion-file", "test.npz", *source])
    assert target.dynamics.weight.item() == 0
    for module in (target.dynamics, target.sim_context):
        assert all(value.item() == 4 for value in module.buffers())


def test_mixed_task_artifact_preserves_paths(tmp_path, monkeypatch):
    from source.tasks.supermimic.training import loop

    files = ["actor.pt", "world_model.pt", "params/sim2real.yaml", "params/velocity/sim2real.yaml"]
    for name in files:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(name)
    logged = []
    run = SimpleNamespace(id="local", entity="team", project="test", log_artifact=logged.append)
    monkeypatch.setattr(loop.wandb, "run", run)
    monkeypatch.setattr(loop.wandb, "Api", lambda: SimpleNamespace(artifacts=lambda *_: []))
    loop.log_model_artifact([str(tmp_path / name) for name in files], tmp_path)
    assert len(logged) == 1
    entries = logged[0].manifest.entries
    assert set(entries) == set(files)
    assert entries[files[-1]].digest != entries[files[-2]].digest
