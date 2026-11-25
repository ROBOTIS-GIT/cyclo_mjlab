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

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from source.tasks.supermimic.training.checkpoint.archive import (
    ModelCheckpoint,
    save_checkpoint,
    load_checkpoint,
)
from source.tasks.supermimic.training.checkpoint.resume import (
    restore_training_state,
    validate_resume_configuration,
)


def make_runner(optimizer_type=torch.optim.Adam, lr=0.001):
    models = {
        name: torch.nn.Linear(3, 3, bias=False)
        for name in ("actor", "critic", "world_model")
    }
    optimizers = {
        name: optimizer_type(model.parameters(), lr=lr)
        for name, model in models.items()
    }
    dynamics = models["world_model"]
    dynamics.register_buffer("action_scale", torch.ones(3))
    dynamics.register_buffer("action_offset", torch.zeros(3))
    ctx = torch.nn.Module()
    ctx.register_buffer("mean", torch.zeros(1))
    ctx.register_buffer("scale", torch.ones(1))

    def load_bundle(state):
        dynamics.load_state_dict(state["dynamics"])
        ctx.load_state_dict(state["sim_context"])

    runner = SimpleNamespace(
        policy_learner=SimpleNamespace(
            actor=models["actor"],
            critic=models["critic"],
            actor_optimizers=[optimizers["actor"]],
            critic_optimizers=[optimizers["critic"]],
        ),
        world_model_learner=SimpleNamespace(
            optimizers=[optimizers["world_model"]]
        ),
        _wm=SimpleNamespace(
            dynamics=dynamics, sim_context=ctx,
            load_bundle=load_bundle,
        ),
        _checkpoint_step=-1,
    )
    return runner, models, optimizers


def update(models, optimizers):
    for name, model in models.items():
        model.weight.grad = (
            torch.arange(9, dtype=torch.float32).reshape(3, 3) / 10
        )
        optimizers[name].step()


@pytest.mark.parametrize(
    "optimizer_type", [torch.optim.Adam, torch.optim.Muon]
)
@pytest.mark.parametrize("full", [True, False])
def test_resume_restores_all_three_models_and_optimizer_updates(
    tmp_path, optimizer_type, full
):
    original, models, optimizers = make_runner(optimizer_type)
    update(models, optimizers)
    save_checkpoint(
        tmp_path,
        step=1000,
        manifest={},
        actor=models["actor"].state_dict(),
        critic=models["critic"].state_dict(),
        world_model={"dynamics": models["world_model"].state_dict(),
                     "sim_context": original._wm.sim_context.state_dict()},
        optimizers={
            name: [opt.state_dict()] for name, opt in optimizers.items()
        },
    )
    path = tmp_path / ("last.pt" if full else "model_1000.pt")
    checkpoint = load_checkpoint(path)
    restored, restored_models, restored_opts = make_runner(
        optimizer_type, lr=0.004
    )
    restored._wm.dynamics.action_scale.fill_(2)
    restored._wm.dynamics.action_offset.fill_(0.3)
    restored._wm.sim_context.mean.fill_(0.75)
    restored._wm.sim_context.scale.fill_(0.25)
    restore_training_state(restored, checkpoint)
    assert torch.all(restored._wm.dynamics.action_scale == 2)
    assert torch.all(restored._wm.dynamics.action_offset == 0.3)
    torch.testing.assert_close(restored._wm.sim_context.mean, original._wm.sim_context.mean)
    torch.testing.assert_close(restored._wm.sim_context.scale, original._wm.sim_context.scale)
    assert restored._checkpoint_step == 1000
    for name in models:
        torch.testing.assert_close(
            models[name].weight, restored_models[name].weight, rtol=0, atol=0
        )
    if full:
        update(models, optimizers)
        update(restored_models, restored_opts)
        for name in models:
            torch.testing.assert_close(
                models[name].weight,
                restored_models[name].weight,
                rtol=0,
                atol=0,
            )
    else:
        assert all(not opt.state for opt in restored_opts.values())


def test_restore_preserves_capturable_learning_rate_tensor():
    from source.tasks.supermimic.training.checkpoint.resume import restore_optimizer

    model = torch.nn.Linear(3, 3)
    old = torch.optim.Adam(model.parameters(), lr=torch.tensor(0.001))
    learning_rate = torch.tensor(0.01)
    current = torch.optim.Adam(model.parameters(), lr=learning_rate)
    restore_optimizer(current, old.state_dict())
    assert current.param_groups[0]["lr"] is learning_rate
    assert learning_rate.item() == pytest.approx(0.001)


def test_missing_critic_or_partial_optimizers_fail_before_loading():
    runner, models, opts = make_runner()
    checkpoint = ModelCheckpoint(Path("last.pt"), {}, {}, {}, step=1000)
    with pytest.raises(ValueError, match="critic"):
        restore_training_state(runner, checkpoint)
    checkpoint.critic = {}
    checkpoint.optimizers = {"actor": []}
    with pytest.raises(ValueError, match="optimizer groups"):
        restore_training_state(runner, checkpoint)


@pytest.fixture
def resume_manifest():
    return dict(
        task_id="task",
        schema_version=1, representation="gencoord",
        environment=dict(commands={"motion": {"body_names": [], "anchor_body_name": "root"}},
                         observations={"full_state": {}}, sim={}, decimation=4, events={}, actions={}, scene={"entities": {}}),
        robot={"joint_order": [], "action_joint_order": []},
        runner_cfg={
            "dynamics_hidden_dims": [32],
            "max_iterations": 100,
            "history_length": 10,
            "log_interval": 10,
            "buffer_size": 1000,
            "replay_ratio": 16,
        },
    )


@pytest.mark.parametrize("changes", [
    {"buffer_size": 500},
    {"replay_ratio": 8},
    {"buffer_size": 2000, "replay_ratio": 32},
    {"latent_bc_weight": 0.02, "termination_softness": 0.1},
    {"wm": {"horizon": 16, "batch_size": 256, "num_batches": 2,
            "nominal_error_joint_pos": 0.2}},
    {"policy": {"lam": 0.8, "gamma": 0.97, "horizon": 16,
                "batch_size": 256, "num_batches": 2, "grad_clip_value": 0.2}},
    {"tasks": {"velocity": {"lin_vel_x_range": [-2., 2.],
                            "window_frames": 10, "command_noise": [0.1, 0.1, 0.1]}}},
    {"slot_mix": {"motion": 0.5, "velocity": 0.5}, "sampling_mode": "uniform"},
])
def test_resume_contract_allows_training_changes(resume_manifest, changes):
    manifest = resume_manifest
    checkpoint = ModelCheckpoint(Path("last.pt"), manifest, {}, {}, step=99, optimizers={})
    resumed = copy.deepcopy(manifest)
    resumed["runner_cfg"].update(max_iterations=200, log_interval=20, wandb_project="new-project", headless=False)
    resumed["runner_cfg"].update(changes)
    manifest["runner_cfg"]["retired_logging_flag"] = True
    resumed["runner_cfg"]["latent_dim"] = 32  # Older runs omitted the current default.
    validate_resume_configuration(checkpoint, resumed)


@pytest.mark.parametrize("field,value", [
    ("history_length", 5), ("actor_obs", "proprio"),
    ("latent_dim", 64), ("dynamics_hidden_dims", [64]),
    ("actor_hidden_dims", [64]), ("critic_hidden_dims", [64]),
    ("policy", {"gamma": 1.0}),  # BPTT has no critic.
])
def test_resume_contract_rejects_model_changes(resume_manifest, field, value):
    checkpoint = ModelCheckpoint(Path("last.pt"), resume_manifest, {}, {}, step=99)
    resumed = copy.deepcopy(resume_manifest)
    resumed["runner_cfg"][field] = value
    with pytest.raises(ValueError, match="learner configuration.*" + field):
        validate_resume_configuration(checkpoint, resumed)


@pytest.mark.parametrize("group,field,value", [
    ("wm", "lr", 0.004), ("policy", "lr", 0.004), ("policy", "beta2", 0.9),
])
def test_resume_checks_optimizer_settings_only_when_restoring_them(resume_manifest, group, field, value):
    from dataclasses import asdict
    from supermimic.runners.run_config import load_runner_cfg

    resumed = copy.deepcopy(resume_manifest)
    resumed["runner_cfg"] = asdict(load_runner_cfg(resumed))
    resumed["runner_cfg"][group][field] = value
    checkpoint = ModelCheckpoint(Path("last.pt"), resume_manifest, {}, {}, step=99)
    validate_resume_configuration(checkpoint, resumed)
    checkpoint.optimizers = {}
    with pytest.raises(ValueError, match=f"optimizer setting {group}.{field}"):
        validate_resume_configuration(checkpoint, resumed)


def test_resumed_loop_continues_global_iteration_and_saves(
    tmp_path, monkeypatch
):
    from source.tasks.supermimic.training import loop as training_loop
    from source.tasks.supermimic.training.runner import LocalBundleRunner

    runner, models, optimizers = make_runner()
    original_models = {
        key: copy.deepcopy(model.state_dict()) for key, model in models.items()
    }
    checkpoint = ModelCheckpoint(
        Path("last.pt"),
        {},
        original_models["actor"],
        {"dynamics": original_models["world_model"], "sim_context": runner._wm.sim_context.state_dict()},
        original_models["critic"],
        999,
        {name: [opt.state_dict()] for name, opt in optimizers.items()},
    )
    restore_training_state(runner, checkpoint)
    collected, logged, cleaned = [], [], []

    def step():
        runner._checkpoint_step += 1
        collected.append(runner._checkpoint_step)
        return 1

    runner.step = step
    runner.cfg = SimpleNamespace(
        max_iterations=1003,
        save_interval=1000,
        log_interval=1,
        wm=SimpleNamespace(num_batches=1, batch_size=1, horizon=8),
        replay_ratio=1,
    )
    runner.env = SimpleNamespace(num_envs=1, reset=lambda: None)
    runner._create_log_dir = lambda: tmp_path
    runner._init_wandb = lambda _: None
    runner._seed_sim_history = lambda: None
    runner._cleanup = lambda: cleaned.append(True)
    runner._wm.bundle = lambda: models["world_model"].state_dict()
    runner._representation = SimpleNamespace(actor_obs_mode="rep_state")
    runner.replay_buffer = SimpleNamespace(num_transitions=0)
    runner.logger = SimpleNamespace(log=lambda **kw: logged.append(kw["it"]))
    runner.manifest = {}
    runner._deployment_files = []
    runner.checkpoint_extra_files = (
        lambda directory: LocalBundleRunner.checkpoint_extra_files(
            runner, directory
        )
    )
    monkeypatch.setattr(
        training_loop.wandb, "run", SimpleNamespace(name="test")
    )
    monkeypatch.setattr(training_loop, "log_model_artifact", lambda _: None)
    training_loop.run_training_loop(runner, runner._checkpoint_step + 1)
    assert collected == logged == [1000, 1001, 1002]
    assert cleaned == [True]
    assert load_checkpoint(tmp_path).step == 1002
    assert (tmp_path / "model_1000.pt").is_file()
    assert (tmp_path / "model_1002.pt").is_file()
    assert not (tmp_path / "model_0.pt").exists()


def test_sampler_remaps_concatenated_clip_bins_and_keeps_new_bins_fresh():
    from source.tasks.supermimic.training.checkpoint.sampler import remap_bins

    old = dict(bins=2, clips=[("a", 5), ("b", 5)], tensors=dict(
        count=torch.tensor([10., 20.]), value=torch.tensor([2., 4.]),
        effort=torch.tensor([5., 10.], dtype=torch.float64),
        probs=torch.tensor([0.25, 0.75]),
    ))
    # New bins span four frames each. The middle bin contains b and new frames.
    new = dict(bins=3, clips=[("b", 5), ("c", 7)], tensors=dict(
        count=torch.zeros(3), value=torch.full((3,), 100.),
        effort=torch.zeros(3, dtype=torch.float64), probs=torch.ones(3) / 3,
    ))
    assert remap_bins(old, old) is old["tensors"]
    state = remap_bins(old, new)
    assert state["count"].tolist() == [16, 4, 0]
    assert state["value"].tolist() == [4, 4, 100]
    assert state["effort"].tolist() == [8, 2, 0]
    assert state["probs"].sum().item() == pytest.approx(1)
    assert state["probs"][2] > 0
    new["clips"] = [("changed-content", 12)]
    assert remap_bins(old, new) is new["tensors"]


def test_sampler_snapshot_round_trip(tmp_path):
    from source.tasks.supermimic.training.checkpoint.sampler import (
        restore_sampler,
        sampler_snapshot,
    )

    command = SimpleNamespace(
        cfg=SimpleNamespace(sampling_mode="adaptive"),
        bin_count=2,
        motion=SimpleNamespace(clip_layout=[("a", 10)]),
        bin_failed_count=torch.tensor([2.0, 3.0]),
        _current_bin_failed=torch.tensor([4.0, 5.0]),
    )
    runner = SimpleNamespace(motion_cmd=command)
    save_checkpoint(
        tmp_path,
        step=1000,
        manifest={},
        actor={},
        world_model={},
        sampler=sampler_snapshot(runner),
    )
    command.bin_failed_count.zero_()
    command._current_bin_failed.zero_()
    restore_sampler(runner, load_checkpoint(tmp_path).sampler)
    assert command.bin_failed_count.tolist() == [2, 3]
    assert command._current_bin_failed.tolist() == [4, 5]


@pytest.mark.parametrize("saved_mode,current_mode", [
    ("external", "uniform"), ("external", "start"),
    ("external", "adaptive"), ("adaptive", "external"),
])
def test_sampler_mode_change_keeps_fresh_statistics_and_restores_training(saved_mode, current_mode, capsys):
    original, models, optimizers = make_runner()
    update(models, optimizers)
    checkpoint = ModelCheckpoint(
        Path("last.pt"), {}, models["actor"].state_dict(),
        {"dynamics": models["world_model"].state_dict(),
         "sim_context": original._wm.sim_context.state_dict()},
        models["critic"].state_dict(), 1000,
        {name: [opt.state_dict()] for name, opt in optimizers.items()},
        sampler={"mode": saved_mode},  # Incompatible statistics must not be read.
    )
    checkpoint = copy.deepcopy(checkpoint)
    restored, restored_models, restored_optimizers = make_runner()
    restored.motion_cmd = SimpleNamespace(
        cfg=SimpleNamespace(sampling_mode=current_mode),
        motion=SimpleNamespace(clip_layout=[("a", 10)]), bin_count=2,
        bin_failed_count=torch.zeros(2), _current_bin_failed=torch.zeros(2),
        _external_probs=torch.full((2,), 0.5),
    )
    restored._sampler = SimpleNamespace(
        count=torch.zeros(2), value=torch.full((2,), 100.), effort=torch.zeros(2),
    )
    restore_training_state(restored, checkpoint)
    assert "Reset sampler mode changed" in capsys.readouterr().out
    assert restored._checkpoint_step == 1000
    assert not restored.motion_cmd.bin_failed_count.any()
    assert not restored.motion_cmd._current_bin_failed.any()
    assert not restored._sampler.count.any()
    assert not restored._sampler.effort.any()
    assert restored._sampler.value.tolist() == [100., 100.]
    assert restored.motion_cmd._external_probs.tolist() == [0.5, 0.5]
    for name in models:
        torch.testing.assert_close(models[name].weight, restored_models[name].weight)
    update(models, optimizers)
    update(restored_models, restored_optimizers)
    for name in models:
        torch.testing.assert_close(models[name].weight, restored_models[name].weight)
