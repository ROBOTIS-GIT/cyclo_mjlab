"""
Verify checkpoint contents against the upstream trainer's actual save block.
"""

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
import textwrap
from types import SimpleNamespace

import pytest
import torch

from source.tasks.blinkmimic.training.checkpoint.snapshots import (
    include_critic_in_snapshots,
)
from source.tasks.blinkmimic.training.checkpoint.selection import publish_completed_snapshots
from source.tasks.blinkmimic.runtime import VENDOR


def test_upstream_saves_critic_in_models_and_optimizers_in_last(
    tmp_path,
):
    source = (VENDOR / "src/blinkmimic/train.py").read_text()
    start = source.index(
        '        last = os.path.join(args.log_dir, "last.pt")'
    )
    end = source.index("    if out_of_time:", start)
    save_block = compile(
        textwrap.dedent(source[start:end]), "upstream_save_block", "exec"
    )
    original_save = torch.save
    clips = [("dance2", "digest2", 50), ("dance1", "digest1", 80)]
    with include_critic_in_snapshots(clips=clips):
        for iteration in (1000, 2000):
            ck = {
                key: {"weight": torch.tensor([iteration + offset])}
                for offset, key in enumerate(("pi", "pi_live", "wm", "critic"))
            }
            ck.update(
                iteration=iteration,
                ctrl={"decay_at": 10},
                wall_time=1,
                sim_steps=10,
                args={},
                saved_at="now",
                opt_pi={"state": 1},
                opt_v={"state": 2},
                opt_wm={"state": 3},
            )
            exec(
                save_block,
                dict(
                    torch=torch,
                    os=os,
                    args=SimpleNamespace(log_dir=str(tmp_path)),
                    ck=ck,
                    it=iteration - 1,
                ),
            )
            publish_completed_snapshots(tmp_path)
            model = torch.load(
                tmp_path / f"model_{iteration}.pt", weights_only=True
            )
            last = torch.load(tmp_path / "last.pt", weights_only=True)
            assert last["ctrl"] == model["ctrl"] == {"decay_at": 10, "clips": clips}
            assert "joint_order" not in last and "joint_order" not in model
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
            torch.save(
                {"iteration": 1000, "critic": {}}, tmp_path / "last.pt.tmp"
            )
            torch.save({"iteration": 2000}, tmp_path / "ckpt_002000.pt.tmp")
    assert torch.save is original_save
    assert not (tmp_path / "ckpt_002000.pt.tmp").exists()


@pytest.mark.parametrize("time_limit", [False, True])
def test_native_light_saves_and_final_full_resume(tmp_path, time_limit):
    from source.tasks.blinkmimic.training.checkpoint.resume import read_resume_checkpoint

    source = (VENDOR / "src/blinkmimic/train.py").read_text()
    tree = ast.parse(source)
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in ("_save_atomic", "ckpt_wait")]
    start = source.index("    if args.ckpt_light and not final_ck")
    end = source.index("    if out_of_time:", start)
    save_block = compile(textwrap.dedent(source[start:end]), "native_light_save", "exec")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    clips = [("dance1", "digest", 50)]
    original_save = torch.save

    for adapted in (False, True):
        directory = tmp_path / str(adapted)
        directory.mkdir()
        args = SimpleNamespace(log_dir=str(directory), ckpt_light=True,
                               ckpt_every=1, iterations=3)
        env = SimpleNamespace(focus_q=None, focus_static=False, focus_share=0.,
                              bin_G=torch.ones(1), bin_fail=torch.zeros(1),
                              bin_w_avg=torch.ones(1))
        scope = dict(torch=torch, os=os, args=args, pi=model,
                     pi_avg=SimpleNamespace(module=model, state_dict=lambda: {}),
                     pi_live=model, wm=model, critic=model,
                     opt_wm=optimizer, opt_pi=optimizer, opt_v=optimizer,
                     env=env, DECAY_AT=[None], EPROBE=False,
                     time=SimpleNamespace(time=lambda: 1., strftime=lambda _: "now"),
                     t_start=0., sim_steps=100, CKPT_PENDING=[], timed=False)
        exec(compile(ast.Module(body=helpers, type_ignores=[]), "native_save_helpers", "exec"), scope)
        context = include_critic_in_snapshots(history={"old": {}}, clips=clips) if adapted else nullcontext()
        with context, ThreadPoolExecutor(max_workers=1) as pool:
            scope.update(CKPT_POOL=pool, it=0, final_ck=False, out_of_time=False)
            exec(save_block, scope)
            scope["ckpt_wait"](drain=True)
            assert not (directory / "last.pt").exists()
            light = torch.load(directory / "ckpt_000001.pt", weights_only=True)
            assert set(light) == {"pi", "iteration", "wall_time", "sim_steps", "args", "saved_at"}
            torch.testing.assert_close(light["pi"], model.state_dict())

            # Both a completed run and --max_hours produce a full final save.
            iteration = 2 if time_limit else 3
            scope.update(it=iteration - 1, final_ck=True, out_of_time=time_limit)
            exec(save_block, scope)
            scope["ckpt_wait"](drain=True)
        last = read_resume_checkpoint(directory / "last.pt")
        assert last["iteration"] == iteration
        assert last["args"]["ckpt_light"] is True
        for key in ("wm", "critic", "opt_wm", "opt_pi", "opt_v"):
            assert key in last
        if adapted:
            assert last["ctrl"]["clips"] == clips
            assert last["inactive_sampler"] == {"old": {}}
        # Final full saving must not add training state to the earlier light file.
        assert set(torch.load(directory / "ckpt_000001.pt", weights_only=True)) == set(light)
    assert torch.save is original_save
    original = torch.load(tmp_path / "False/ckpt_000001.pt", weights_only=True)
    adapted = torch.load(tmp_path / "True/ckpt_000001.pt", weights_only=True)
    original["args"].pop("log_dir")
    adapted["args"].pop("log_dir")
    torch.testing.assert_close(adapted.pop("pi"), original.pop("pi"), rtol=0, atol=0)
    assert adapted == original


@pytest.mark.parametrize("through_worker", [False, True])
def test_light_checkpoint_reports_resume_error(tmp_path, through_worker):
    from source.tasks.blinkmimic.training.checkpoint.resume import (
        native_training_resume, read_resume_checkpoint,
    )

    path = tmp_path / "model_100.pt"
    torch.save(dict(pi={}, args={"ckpt_light": True}, iteration=100,
                    wall_time=1., sim_steps=10), path)
    original_load = torch.load
    with pytest.raises(ValueError, match=r"policy-only.*last\.pt"):
        if through_worker:
            with native_training_resume(path):
                torch.load(path, weights_only=True)
        else:
            read_resume_checkpoint(path)
    assert torch.load is original_load


@pytest.mark.parametrize("with_optimizers", [False, True])
@pytest.mark.parametrize("optimizer_kind", ["adam", "muon"])
def test_native_resume_restores_models_and_next_update(
    tmp_path, with_optimizers, optimizer_kind
):
    from source.tasks.blinkmimic.runtime import prepare

    prepare()
    from blinkmimic.optim import MuonAdam
    from source.tasks.blinkmimic.training.checkpoint.resume import (
        native_training_resume,
        read_resume_checkpoint,
    )

    def create():
        models = [
            torch.nn.Sequential(
                torch.nn.Linear(3, 3),
                torch.nn.Linear(3, 3),
                torch.nn.Linear(3, 3),
            )
            for _ in range(3)
        ]
        optimizers = [
            (
                MuonAdam(m, [m], lr=0.001, fused=False)
                if optimizer_kind == "muon"
                else torch.optim.Adam(m.parameters(), lr=0.001)
            )
            for m in models
        ]
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
        saved.update(
            zip(
                ("opt_wm", "opt_pi", "opt_v"),
                [o.state_dict() for o in optimizers],
            )
        )
    path = tmp_path / ("last.pt" if with_optimizers else "model_1000.pt")
    torch.save(saved, path)
    assert read_resume_checkpoint(path)["iteration"] == 1000
    restored_models, restored_optimizers = create()
    # Run the actual upstream restoration block, including its assertion and
    # EMA/control branches.
    source = (VENDOR / "src/blinkmimic/train.py").read_text()
    statements = ast.parse(source).body
    load_states = next(
        node for node in statements
        if isinstance(node, ast.FunctionDef) and node.name == "load_opt_states"
    )
    resume = next(
        node for node in statements
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args.resume"
    )
    restore_block = ast.Module(body=[load_states, resume], type_ignores=[])
    scope = dict(
        torch=torch,
        args=SimpleNamespace(resume=str(path)),
        dev="cpu",
        pi_avg=None,
        check_pruned=lambda *a, **kw: None,
    )
    scope.update(zip(("wm", "pi", "critic"), restored_models))
    scope.update(zip(("opt_wm", "opt_pi", "opt_v"), restored_optimizers))
    original_load, original_adam = torch.load, torch.optim.Adam.load_state_dict
    with native_training_resume(path):
        exec(compile(restore_block, "native_resume", "exec"), scope)
    assert (
        torch.load is original_load
        and torch.optim.Adam.load_state_dict is original_adam
    )
    assert (scope["start_it"], scope["wall_offset"], scope["sim_steps"]) == (
        1000,
        123,
        456,
    )
    for model, restored in zip(models, restored_models):
        for expected, actual in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if with_optimizers:
        update(models, optimizers)
        update(restored_models, restored_optimizers)
        for model, restored in zip(models, restored_models):
            for expected, actual in zip(
                model.parameters(), restored.parameters()
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert all(
            not (o.adam.state if optimizer_kind == "muon" else o.state)
            for o in restored_optimizers
        )


def test_resume_adapter_restores_hooks_on_failure(tmp_path):
    from source.tasks.blinkmimic.training.checkpoint.resume import native_training_resume

    original_load, original_adam = torch.load, torch.optim.Adam.load_state_dict
    path = tmp_path / "broken.pt"
    torch.save({"opt_pi": {}}, path)
    with pytest.raises(ValueError, match="Incomplete optimizer"):
        with native_training_resume(path):
            torch.load(path, weights_only=True)
    assert (
        torch.load is original_load
        and torch.optim.Adam.load_state_dict is original_adam
    )


def test_saved_training_arguments_are_authoritative():
    from source.tasks.blinkmimic.training.checkpoint.resume import training_arguments

    saved = {
        "args": dict(
            robot="k1",
            hist=7,
            lr=0.004,
            cuda_graph=False,
            no_sync=True,
            resume="old.pt",
            log_dir="old",
            ckpt_light=True,
            wm_structured=True,
        )
    }
    assert training_arguments(saved) == [
        "--hist",
        "7",
        "--lr",
        "0.004",
        "--no_sync",
    ]


@pytest.mark.parametrize("current", [True, False])
def test_resume_uses_current_structured_dynamics_recipe(monkeypatch, current):
    from source.tasks.blinkmimic import cli
    from source.tasks.blinkmimic.training.checkpoint import resume

    saved = dict(args=dict(wm_structured=not current, iterations=100), iteration=10)
    monkeypatch.setattr(resume, "read_resume_checkpoint", lambda path: saved)
    monkeypatch.setattr(cli, "recipe", lambda: ["--wm_structured"] if current else [])
    args = SimpleNamespace(execution="official")
    flags, _, _ = cli.resolve_training_resume(args, [], "last.pt")
    assert ("--wm_structured" in flags) is current


def test_training_launcher_forwards_resume_and_saved_arguments(
    tmp_path, monkeypatch
):
    import json
    from source.tasks.blinkmimic import cli, runtime

    task = "Cyclo-BlinkMimic-K1-Rev1-001"
    run = tmp_path / "logs" / "blinkmimic" / task / "old"
    run.mkdir(parents=True)
    motions = tmp_path / "external-motions"
    motions.mkdir()
    (motions / "manifest.json").write_text("{}")
    (run / "run-record.json").write_text(
        json.dumps(
            {
                "task": task,
                "manifest": str(motions / "manifest.json"),
                "cwd": str(tmp_path),
            }
        )
    )
    torch.save(
        dict(
            args=dict(robot="k1", hist=7, iterations=2000),
            pi={},
            wm={},
            critic={},
            iteration=1000,
            wall_time=5,
            sim_steps=123,
        ),
        run / "last.pt",
    )
    original = (run / "last.pt").read_bytes()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(runtime, "prepare", lambda: None)
    monkeypatch.setattr(
        cli,
        "TASKS",
        {
            task: SimpleNamespace(
                robot_factory=lambda: SimpleNamespace(name="k1", joint_names=["a"])
            )
        },
    )
    monkeypatch.setattr(cli, "prepare_task_motions", lambda task, directory: directory)
    received = []
    monkeypatch.setattr(
        cli, "run_worker", lambda args, **kw: received.append((args, kw))
    )
    parser = cli.create_parser()
    args, extra = cli.parse_arguments(
        parser,
        "train",
        [task, "--agent.resume", "True", "--iterations", "3000"],
    )
    cli.run_training(args, extra)
    forwarded, kw = received[0]
    flags = kw["flags"]
    assert flags[flags.index("--resume") + 1] == str(run / "last.pt")
    assert flags[flags.index("--hist") + 1] == "7"
    assert forwarded.motion_dir == forwarded.run_dir / "motions"
    assert forwarded.run_dir != run
    assert (run / "last.pt").read_bytes() == original
