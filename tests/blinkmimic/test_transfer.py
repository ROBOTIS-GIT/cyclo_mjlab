"""Exercise both initialization modes at the native BlinkMimic boundary."""

import json
from dataclasses import replace

import pytest
import torch

from source.tasks.blinkmimic import cli
from source.tasks.blinkmimic.runtime import prepare
from source.tasks.blinkmimic.training.checkpoint.sampler import sampler_clips
from source.tasks.blinkmimic.training.checkpoint.resume import (
    native_training_resume,
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


def test_sampler_follows_clip_identity_on_native_load(tmp_path, capsys):
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
    saved = dict(args={"robot": "k1"}, ctrl=control, opt_pi={"state": 1}, opt_v={}, opt_wm={},
                 pi={"weight": torch.ones(1)}, wm={"weight": torch.ones(1)}, critic={"weight": torch.ones(1)})
    torch.save(saved, path)
    # Even a valid original manifest is ignored when the checkpoint has no clip identities.
    record = tmp_path / "run-record.json"
    record.write_text(json.dumps({"manifest": str(source / "manifest.json")}))
    with native_training_resume(path, target_clips=sampler_clips(target)):
        reset = torch.load(path, weights_only=True)
    assert "Checkpoint has no sampler clip identities" in capsys.readouterr().out
    assert reset["ctrl"]["decay_at"] == 125 and reset["opt_pi"] == saved["opt_pi"]
    for key in ("pi", "wm", "critic"):
        torch.testing.assert_close(reset[key]["weight"], saved[key]["weight"])
    assert reset["ctrl"]["env_bins"]["bin_G"].tolist() == [0] * 4
    assert reset["ctrl"]["env_bins"]["bin_fail"].tolist() == [0] * 4
    assert reset["ctrl"]["env_bins"]["bin_w_avg"].tolist() == [0.25] * 4
    assert reset["ctrl"]["ep_fail"].tolist() == [0] * 4
    assert reset["ctrl"]["focus_q"] is None
    record.unlink()
    control["clips"] = sampler_clips(source)
    torch.save(saved, path)
    with native_training_resume(path, sampler_clips(target)):
        state = torch.load(path, weights_only=True)["ctrl"]
    assert state["decay_at"] == 125
    assert state["focus_q"] is None and state["focus_share"] == 0
    for field, expected in (("bin_G", 6), ("bin_fail", 9)):
        assert state["env_bins"][field].tolist() == [expected, 0, 0, 0]
    assert state["ep_fail"].tolist() == [3, 0, 0, 0]
    torch.testing.assert_close(
        state["env_bins"]["bin_w_avg"].sum(), torch.tensor(1.0)
    )
    from source.tasks.blinkmimic.training.checkpoint.snapshots import include_critic_in_snapshots

    reduced = library(tmp_path / "reduced", [("b", "b1", 50)])
    reduced_clips = sampler_clips(reduced)
    with (
        native_training_resume(path, reduced_clips) as history,
        include_critic_in_snapshots(history, reduced_clips),
    ):
        saved = torch.load(path, weights_only=True)
        assert saved["ctrl"]["env_bins"]["bin_G"].tolist() == [6]
        saved["ctrl"]["env_bins"]["bin_G"].fill_(9)  # The remaining clip keeps learning.
        saved.update(iteration=1, critic={})
        torch.save(saved, tmp_path / "last.pt.tmp")
        torch.save({"iteration": 1, "args": saved["args"]}, tmp_path / "ckpt_000001.pt.tmp")
    (reduced / "manifest.json").unlink()  # Move only the checkpoint into the next run.
    for snapshot in ("last.pt.tmp", "ckpt_000001.pt.tmp"):
        next_path = tmp_path / snapshot
        with native_training_resume(next_path, target_clips=sampler_clips(source)) as history:
            state = torch.load(next_path, weights_only=True)["ctrl"]
        assert not history
        assert state["env_bins"]["bin_G"].tolist() == [4, 5, 9]
        torch.testing.assert_close(state["env_bins"]["bin_fail"], control["env_bins"]["bin_fail"])
        torch.testing.assert_close(state["ep_fail"], control["ep_fail"])
        assert state["focus_q"] is None
        torch.testing.assert_close(state["env_bins"]["bin_w_avg"].sum(), torch.tensor(1.0))
    changed_return = library(tmp_path / "changed_return", [("a", "a2", 80), ("b", "b1", 50)])
    with native_training_resume(next_path, target_clips=sampler_clips(changed_return)) as history:
        state = torch.load(next_path, weights_only=True)["ctrl"]
    assert state["env_bins"]["bin_G"].tolist() == [0, 0, 9]
    assert ("a", "a1", 80) in history
    extended = library(
        tmp_path / "extended",
        [
            ("c", "c1", 20),
            ("b", "b1", 50),
            ("a", "a1", 80),
        ],
    )
    with native_training_resume(path, sampler_clips(extended)):
        state = torch.load(path, weights_only=True)["ctrl"]
    assert state["env_bins"]["bin_G"].tolist() == [0, 6, 4, 5]
    assert state["focus_q"].tolist() == [0, 0, 0.5, 0.5]
    changed = library(tmp_path / "changed", [("a", "new-hash", 80)])
    from source.tasks.blinkmimic.training.checkpoint.sampler import transfer_sampler
    assert transfer_sampler(None, source, changed, {}) is None
    robot = cli.TASKS["Cyclo-BlinkMimic-K1-Rev2-001"].robot_factory()
    with native_training_resume(path, sampler_clips(changed), target_robot=robot):
        reset = torch.load(path, weights_only=True)["ctrl"]
    assert reset["env_bins"]["bin_G"].tolist() == [0, 0]
    assert reset["focus_q"] is None
    updated = library(tmp_path / "updated", [("a", "new-hash", 120), ("b", "b1", 50)])
    with native_training_resume(path, sampler_clips(updated)):
        restored = torch.load(path, weights_only=True)
    assert restored["ctrl"]["decay_at"] == 125
    assert restored["ctrl"]["env_bins"]["bin_G"].tolist() == [0, 0, 0, 6]
    assert restored["ctrl"]["env_bins"]["bin_fail"].tolist() == [0, 0, 0, 9]
    assert restored["ctrl"]["ep_fail"].tolist() == [0, 0, 0, 3]


@pytest.mark.parametrize("numbered", [False, True])
def test_resume_reference_buffers_follow_target_robot(tmp_path, numbered):
    prepare()
    from blinkmimic.networks import Policy, policy_kwargs
    from source.tasks.blinkmimic.export import export_policy, native_dimensions, validate_policy_state

    target = cli.TASKS["Cyclo-BlinkMimic-K1-Rev2-001"].robot_factory()
    source = replace(
        target, default_joint_pos={".*": 0.125}, action_scale=dict.fromkeys(target.joint_names, 0.75)
    )
    args = dict(hist=1, goal_frames=1, enc_hidden="8", dec_hidden="8", pi_ref_skip=True)

    def policy(robot):
        return Policy(*native_dimensions(robot.num_joints), robot.num_joints, hist=1,
                      enc_hidden=(8,), dec_hidden=(8,), **policy_kwargs(args, robot))

    live = policy(source)
    ema = torch.optim.swa_utils.AveragedModel(live, use_buffers=True)
    with torch.no_grad():
        ema.module.dec[-1].bias.fill_(1)
    ema.n_averaged.fill_(7)
    ckpt = dict(args=args, pi=ema.module.state_dict(), pi_live=live.state_dict(), wm={}, critic={})
    if not numbered:
        ckpt.update(pi_avg=ema.state_dict(), opt_pi={}, opt_wm={}, opt_v={})
    path = tmp_path / ("model_1000.pt" if numbered else "last.pt")
    torch.save(ckpt, path)
    with native_training_resume(path, target_robot=target):
        saved = torch.load(path, weights_only=True)

    restored = policy(target)
    for key, original in (("pi_live", live), ("pi", ema.module)):
        restored.load_state_dict(saved[key])
        validate_policy_state(restored.state_dict(), args, target)
        torch.testing.assert_close(restored.dec[-1].bias, original.dec[-1].bias)
    if not numbered:
        restored_ema = torch.optim.swa_utils.AveragedModel(restored, use_buffers=True)
        restored_ema.load_state_dict(saved["pi_avg"])
        validate_policy_state(restored_ema.module.state_dict(), args, target)
        assert restored_ema.n_averaged.item() == 7
    resumed = tmp_path / "resumed.pt"
    torch.save(dict(args=args, pi=saved["pi"]), resumed)
    assert export_policy(resumed, tmp_path / "export", target).is_file()


def test_playback_reference_transform_matches_native_evaluator(tmp_path, monkeypatch):
    from blinkmimic.networks import Policy, policy_kwargs
    from source.tasks.blinkmimic import play, viewer
    from source.tasks.blinkmimic.export import native_dimensions

    source = cli.TASKS["Cyclo-BlinkMimic-K1-Rev2-001"].robot_factory()
    target = replace(source, default_joint_pos={".*": 0.125},
                     action_scale=dict.fromkeys(source.action_scale, 0.375))
    args = dict(hist=1, goal_frames=1, enc_hidden="8", dec_hidden="8", pi_ref_skip=True)

    def policy(robot):
        return Policy(*native_dimensions(robot.num_joints), robot.num_joints, hist=1,
                      enc_hidden=(8,), dec_hidden=(8,), **policy_kwargs(args, robot))

    saved = policy(source).state_dict()
    path = tmp_path / "last.pt"
    torch.save({"pi": saved, "args": args}, path)
    native_policy = policy(target)
    native_policy.load_state_dict(saved)
    torch.testing.assert_close(native_policy.ref_alpha, saved["ref_alpha"])
    torch.testing.assert_close(native_policy.ref_q0, saved["ref_q0"])
    received = []
    def playback(policy, policy_args, motion_dir, robot):
        assert robot is target
        return policy

    monkeypatch.setattr(play, "BlinkMimicPlayback", playback)
    monkeypatch.setattr(viewer, "run_viewer", lambda policy, *_a, **_kw: received.append(policy))
    play.main(["--checkpoint", str(path), "--motion-dir", str(tmp_path)], robot=target)
    torch.testing.assert_close(received[0].state_dict(), native_policy.state_dict())
    assert (tmp_path / "exported/policy.onnx").is_file()
    assert not (tmp_path / "play").exists()
    torch.testing.assert_close(torch.load(path, weights_only=True)["pi"], saved)


def test_world_model_only_initialization(tmp_path):
    prepare()
    from blinkmimic.networks import WorldModel

    torch.manual_seed(87)
    original = WorldModel(2, 3, hidden=(8,), structured=True)
    original.in_norm.mean.fill_(4)
    path = tmp_path / "model_1000.pt"
    torch.save(
        dict(
            args=dict(robot="unregistered", wm_structured=True), wm=original.state_dict()
        ),
        path,
    )

    def fresh_models():
        torch.manual_seed(13)
        return WorldModel(2, 3, hidden=(8,), structured=True), torch.nn.Linear(
            2, 2
        )

    _, expected_actor = fresh_models()
    with initialize_world_model(path):
        model, actor = fresh_models()
        assert callable(WorldModel.increments)
        assert not torch.optim.Adam(model.parameters()).state
    for key, value in original.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    assert torch.equal(actor.weight, expected_actor.weight)
    fresh, _ = fresh_models()
    assert not torch.equal(fresh.net[0].weight, model.net[0].weight)
    with pytest.raises(ValueError, match="representation"):
        with initialize_world_model(path):
            WorldModel(2, 3, hidden=(8,), structured=False)


@pytest.mark.parametrize("src_rev,dst_rev,has_sampler", [(1, 1, False), (1, 2, True), (2, 1, True)])
def test_resume_prepares_target_registered_motions(
    tmp_path, monkeypatch, src_rev, dst_rev, has_sampler
):
    src = f"Cyclo-BlinkMimic-K1-Rev{src_rev}-001"
    dst = f"Cyclo-BlinkMimic-K1-Rev{dst_rev}-001"
    run = tmp_path / "old"
    run.mkdir()
    if has_sampler:
        motions = library(run / "motions", [("dance", "hash", 50)])
        (run / "run-record.json").write_text(
            json.dumps(dict(task=src, manifest=str(motions / "manifest.json")))
        )
    path = run / "last.pt"
    torch.save(
        dict(
            args=dict(robot="k1" if src_rev == 1 else "k1_rev2", iterations=100),
            iteration=50,
            ctrl={"env_bins": {}} if has_sampler else None,
            pi={},
            wm={},
            critic={},
            wall_time=0,
            sim_steps=0,
        ),
        path,
    )
    (run / "params").mkdir()
    (run / "params/sim2real.yaml").write_text("invalid: [")
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
    assert flags[flags.index("--resume") + 1] == str(path)
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
