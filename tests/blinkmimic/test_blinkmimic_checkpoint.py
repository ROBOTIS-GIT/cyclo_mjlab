"""Verify checkpoint contents against the upstream trainer's actual save block."""

import os
import textwrap
from types import SimpleNamespace

import pytest
import torch

from source.tasks.blinkmimic.checkpoint_adapter import include_critic_in_snapshots
from source.tasks.blinkmimic.checkpoint import publish_completed_snapshots
from source.tasks.blinkmimic.runtime import VENDOR


def test_upstream_save_includes_critic_only_in_model_and_optimizers_only_in_last(tmp_path):
    source = (VENDOR / "src/blinkmimic/train.py").read_text()
    start = source.index('        last = os.path.join(args.log_dir, "last.pt")')
    end = source.index('    if out_of_time:', start)
    save_block = compile(textwrap.dedent(source[start:end]), "upstream_save_block", "exec")
    original_save = torch.save
    with include_critic_in_snapshots():
        for iteration in (1000, 2000):
            ck = {key: {"weight": torch.tensor([iteration + offset])}
                  for offset, key in enumerate(("pi", "pi_live", "wm", "critic"))}
            ck.update(iteration=iteration, wall_time=1, sim_steps=10, args={}, saved_at="now",
                      opt_pi={"state": 1}, opt_v={"state": 2}, opt_wm={"state": 3})
            exec(save_block, dict(torch=torch, os=os, args=SimpleNamespace(log_dir=str(tmp_path)),
                                  ck=ck, it=iteration - 1))
            publish_completed_snapshots(tmp_path)
            model = torch.load(tmp_path / f"model_{iteration}.pt", weights_only=True)
            last = torch.load(tmp_path / "last.pt", weights_only=True)
            for key in ("pi", "pi_live", "wm", "critic"):
                assert torch.equal(model[key]["weight"], ck[key]["weight"])
            for key in ("opt_pi", "opt_v", "opt_wm"):
                assert key not in model
                assert last[key] == ck[key]
    assert torch.save is original_save
    first = torch.load(tmp_path / "model_1000.pt", weights_only=True)
    assert first["critic"]["weight"].item() == 1003


def test_missing_or_mismatched_critic_is_not_saved(tmp_path):
    original_save = torch.save
    with pytest.raises(RuntimeError, match="matching latest critic"):
        with include_critic_in_snapshots():
            torch.save({"iteration": 1000, "critic": {}}, tmp_path / "last.pt.tmp")
            torch.save({"iteration": 2000}, tmp_path / "ckpt_002000.pt.tmp")
    assert torch.save is original_save
    assert not (tmp_path / "ckpt_002000.pt.tmp").exists()


@pytest.mark.parametrize("with_optimizers", [False, True])
@pytest.mark.parametrize("optimizer_kind", ["adam", "muon"])
def test_native_resume_restores_models_and_next_update(tmp_path, with_optimizers, optimizer_kind):
    import copy
    from source.tasks.blinkmimic.runtime import prepare
    prepare()
    from blinkmimic.optim import MuonAdam
    from source.tasks.blinkmimic.resume import allow_model_only_resume, read_resume_checkpoint

    def create():
        models = [torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 3),
                                      torch.nn.Linear(3, 3)) for _ in range(3)]
        optimizers = [MuonAdam(m, [m], lr=0.001, fused=False) if optimizer_kind == "muon"
                      else torch.optim.Adam(m.parameters(), lr=0.001) for m in models]
        return models, optimizers

    def update(models, optimizers):
        for model, optimizer in zip(models, optimizers):
            for parameter in model.parameters():
                parameter.grad = torch.full_like(parameter, 0.5)
            optimizer.step()

    models, optimizers = create()
    update(models, optimizers)
    saved = dict(zip(("wm", "pi", "critic"), [m.state_dict() for m in models]))
    saved.update(iteration=1000, wall_time=123, sim_steps=456, args={})
    if with_optimizers:
        saved.update(zip(("opt_wm", "opt_pi", "opt_v"), [o.state_dict() for o in optimizers]))
    path = tmp_path / ("last.pt" if with_optimizers else "model_1000.pt")
    torch.save(saved, path)
    assert read_resume_checkpoint(path)["iteration"] == 1000
    restored_models, restored_optimizers = create()
    # Run the actual upstream restoration block, including its assertion and EMA/control branches.
    source = (VENDOR / "src/blinkmimic/train.py").read_text()
    start = source.index("def load_opt_states(")
    end = source.index("# per-component floors", start)
    scope = dict(torch=torch, args=SimpleNamespace(resume=str(path)), dev="cpu", pi_avg=None,
                 check_pruned=lambda *a, **kw: None)
    scope.update(zip(("wm", "pi", "critic"), restored_models))
    scope.update(zip(("opt_wm", "opt_pi", "opt_v"), restored_optimizers))
    original_load, original_adam = torch.load, torch.optim.Adam.load_state_dict
    with allow_model_only_resume(path):
        exec(compile(source[start:end], "native_resume", "exec"), scope)
    assert torch.load is original_load and torch.optim.Adam.load_state_dict is original_adam
    assert (scope["start_it"], scope["wall_offset"], scope["sim_steps"]) == (1000, 123, 456)
    for model, restored in zip(models, restored_models):
        for expected, actual in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if with_optimizers:
        update(models, optimizers)
        update(restored_models, restored_optimizers)
        for model, restored in zip(models, restored_models):
            for expected, actual in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert all(not (o.adam.state if optimizer_kind == "muon" else o.state) for o in restored_optimizers)


def test_resume_adapter_restores_hooks_on_failure(tmp_path):
    from source.tasks.blinkmimic.resume import allow_model_only_resume
    original_load, original_adam = torch.load, torch.optim.Adam.load_state_dict
    path = tmp_path / "broken.pt"
    torch.save({"opt_pi": {}}, path)
    with pytest.raises(ValueError, match="Incomplete optimizer"):
        with allow_model_only_resume(path):
            torch.load(path, weights_only=True)
    assert torch.load is original_load and torch.optim.Adam.load_state_dict is original_adam


def test_saved_training_arguments_are_authoritative():
    from source.tasks.blinkmimic.resume import training_arguments
    saved = {"args": dict(robot="k1", hist=7, lr=0.004, cuda_graph=False, no_sync=True,
                          resume="old.pt", log_dir="old", ckpt_light=True)}
    assert training_arguments(saved, "k1") == ["--hist", "7", "--lr", "0.004", "--no_sync"]
    with pytest.raises(ValueError, match="robot"):
        training_arguments(saved, "other")


def test_training_launcher_forwards_resume_and_saved_arguments(tmp_path, monkeypatch):
    import json
    from source.tasks.blinkmimic import cli, runtime
    task = "Cyclo-BlinkMimic-K1-Rev1-001"
    run = tmp_path / "logs" / task / "old"
    motions = run / "motions"
    motions.mkdir(parents=True)
    (motions / "manifest.json").write_text("{}")
    (run / "run-record.json").write_text(json.dumps({"task": task, "manifest": str(motions / "manifest.json"),
                                                    "cwd": str(tmp_path)}))
    torch.save(dict(args=dict(robot="k1", hist=7, iterations=2000), pi={}, wm={}, critic={},
                    iteration=1000, wall_time=5, sim_steps=123), run / "last.pt")
    original = (run / "last.pt").read_bytes()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "prepare", lambda: None)
    monkeypatch.setattr(cli, "TASKS", {task: SimpleNamespace(robot_factory=lambda: SimpleNamespace(name="k1"))})
    received = []
    monkeypatch.setattr(cli, "run_worker", lambda args, **kw: received.append((args, kw)))
    parser = cli.create_parser()
    args, extra = cli.parse_arguments(parser, "train", [task, "--agent.resume", "True", "--iterations", "3000"])
    cli.run_training(args, extra)
    forwarded, kw = received[0]
    flags = kw["flags"]
    assert flags[flags.index("--resume") + 1] == str(run / "last.pt")
    assert flags[flags.index("--hist") + 1] == "7"
    assert forwarded.motion_dir == motions
    assert forwarded.run_dir != run
    assert (run / "last.pt").read_bytes() == original
