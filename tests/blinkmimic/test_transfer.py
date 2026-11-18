"""Exercise both initialization modes at the native BlinkMimic boundary."""

import json
from types import SimpleNamespace

import pytest
import torch

from source.tasks.blinkmimic import cli
from source.tasks.blinkmimic.runtime import prepare
from source.tasks.blinkmimic.resume import (
    allow_model_only_resume,
    initialize_world_model,
)


def library(path, clips):
    path.mkdir(parents=True)
    rows = [
        dict(song=name, derived_sha256=digest, frames=frames, fps=50)
        for name, digest, frames in clips
    ]
    (path / "manifest.json").write_text(json.dumps(dict(clips=rows)))
    return path


def test_sampler_follows_clip_identity_on_native_load(tmp_path):
    prepare()
    source = library(tmp_path / "source", [("a", "a1", 80), ("b", "b1", 50)])
    target = library(tmp_path / "target", [("b", "b1", 50), ("c", "c1", 101)])
    control = dict(
        decay_at=125,
        ep_fail=torch.tensor([1.0, 2.0, 3.0]),
        focus_q=torch.tensor([0.5, 0.5, 0.0]),
        focus_share=0.2,
        env_bins={
            "bin_G": torch.tensor([4.0, 5.0, 6.0]),
            "bin_fail": torch.tensor([7.0, 8.0, 9.0]),
            "bin_w_avg": torch.tensor([0.2, 0.4, 0.4]),
        },
    )
    path = tmp_path / "last.pt"
    torch.save(dict(ctrl=control, opt_pi={}, opt_v={}, opt_wm={}), path)
    with allow_model_only_resume(path, source, target, allow_removed=True):
        state = torch.load(path, weights_only=True)["ctrl"]
    assert state["decay_at"] == 125
    assert state["focus_q"] is None and state["focus_share"] == 0
    for field, expected in (("bin_G", 6), ("bin_fail", 9)):
        assert state["env_bins"][field].tolist() == [expected, 0, 0, 0]
    assert state["ep_fail"].tolist() == [3, 0, 0, 0]
    torch.testing.assert_close(
        state["env_bins"]["bin_w_avg"].sum(), torch.tensor(1.0)
    )
    with pytest.raises(ValueError, match="lost motion"):
        with allow_model_only_resume(path, source, target):
            torch.load(path, weights_only=True)
    extended = library(
        tmp_path / "extended",
        [
            ("c", "c1", 20),
            ("b", "b1", 50),
            ("a", "a1", 80),
        ],
    )
    with allow_model_only_resume(path, source, extended):
        state = torch.load(path, weights_only=True)["ctrl"]
    assert state["env_bins"]["bin_G"].tolist() == [0, 6, 4, 5]
    assert state["focus_q"].tolist() == [0, 0, 0.5, 0.5]
    changed = library(tmp_path / "changed", [("a", "new-hash", 80)])
    with pytest.raises(ValueError, match="same name"):
        with allow_model_only_resume(
            path, source, changed, allow_removed=True
        ):
            torch.load(path, weights_only=True)


def test_world_model_only_initialization(tmp_path):
    prepare()
    from blinkmimic.networks import WorldModel

    torch.manual_seed(87)
    original = WorldModel(2, 3, hidden=(8,), structured=True)
    original.in_norm.mean.fill_(4)
    path = tmp_path / "model_1000.pt"
    torch.save(
        dict(
            args=dict(robot="k1", wm_structured=True), wm=original.state_dict()
        ),
        path,
    )

    def fresh_models():
        torch.manual_seed(13)
        return WorldModel(2, 3, hidden=(8,), structured=True), torch.nn.Linear(
            2, 2
        )

    _, expected_actor = fresh_models()
    with initialize_world_model(path, "k1"):
        model, actor = fresh_models()
        assert callable(WorldModel.increments)
        assert not torch.optim.Adam(model.parameters()).state
    for key, value in original.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    assert torch.equal(actor.weight, expected_actor.weight)
    fresh, _ = fresh_models()
    assert not torch.equal(fresh.net[0].weight, model.net[0].weight)
    with pytest.raises(ValueError, match="representation"):
        with initialize_world_model(path, "k1"):
            WorldModel(2, 3, hidden=(8,), structured=False)
    with pytest.raises(ValueError, match="robot"):
        with initialize_world_model(path, "other"):
            pass


@pytest.mark.parametrize("cross_task", [False, True])
def test_resume_prepares_target_registered_motions(
    tmp_path, monkeypatch, cross_task
):
    src = "Cyclo-BlinkMimic-K1-Rev1-002"
    dst = "Cyclo-BlinkMimic-K1-Rev1-003" if cross_task else src
    run = tmp_path / "old"
    motions = library(run / "motions", [("dance", "hash", 50)])
    (run / "run-record.json").write_text(
        json.dumps(
            dict(
                task=src,
                manifest=str(motions / "manifest.json"),
            )
        )
    )
    path = run / "last.pt"
    torch.save(
        dict(
            args=dict(robot="k1", iterations=100),
            iteration=50,
            pi={},
            wm={},
            critic={},
            wall_time=0,
            sim_steps=0,
        ),
        path,
    )
    cfg = SimpleNamespace(robot_factory=lambda: SimpleNamespace(name="k1"))
    monkeypatch.setattr(cli, "TASKS", {src: cfg, dst: cfg})
    prepared = []

    def prepare_task(task, target):
        prepared.append(task)
        return library(target, [("dance", "hash", 50), ("new", "new", 20)])

    monkeypatch.setattr(cli, "prepare_task_motions", prepare_task)
    monkeypatch.setattr(cli, "run_worker", lambda args, **kw: kw["flags"])
    args, extra = cli.parse_arguments(
        cli.create_parser(),
        "train",
        [
            dst,
            "--run-dir",
            str(tmp_path / "out"),
            "--agent.resume",
            "True",
            "--agent.load-checkpoint",
            str(path),
        ],
    )
    flags = cli.run_training(args, extra)
    assert prepared == [dst]
    assert args.motion_dir == tmp_path / "out" / "motions"
    assert flags[flags.index("--resume-source-library") + 1] == str(motions)
    assert ("--allow-removed-clips" in flags) == cross_task
    with pytest.raises(SystemExit):
        cli.parse_arguments(
            cli.create_parser(),
            "train",
            [
                dst,
                "--agent.resume",
                "True",
                "--world-model-checkpoint",
                str(path),
            ],
        )
