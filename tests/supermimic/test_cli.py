import sys

import pytest

from source.tasks.supermimic import cli


@pytest.mark.parametrize("requested,display,wayland,expected", [
    ("auto", ":0", None, "native"),
    ("auto", None, "wayland-0", "native"),
    ("auto", None, None, "viser"),
    ("auto", "", "", "viser"),
    ("viser", ":0", None, "viser"),
    ("native", None, None, "native"),
])
def test_play_viewer_selection_matches_upstream(monkeypatch, tmp_path, requested, display, wayland, expected):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import tyro
    from supermimic.scripts import viewers
    from source.tasks.supermimic import play, viewer

    for name, value in (("DISPLAY", display), ("WAYLAND_DISPLAY", wayland)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    motion = tmp_path / "dance1.npz"
    motion.touch()
    cfg = tyro.cli(play.PlayConfig, args=["--motion-file", str(motion), "--viewer", requested])
    constructors = {name: Mock() for name in ("native", "viser")}
    monkeypatch.setattr(viewers, "SuperMimicNativeViewer", constructors["native"])
    monkeypatch.setattr(viewer, "ClipPlaybackViewer", constructors["viser"])
    monkeypatch.setattr(viewer, "adapt_ghost_visualizer", Mock())
    monkeypatch.setattr(viewers, "create_wm_vis_model", lambda model: model)
    env = SimpleNamespace(unwrapped=SimpleNamespace(sim=SimpleNamespace(mj_model=object())))
    wm = SimpleNamespace(dynamics=object(), representation=object(), sim_context=object())
    actor = SimpleNamespace(task_specs=[SimpleNamespace(name="motion")])
    play.run_interactive_playback(env, wm, actor, cfg, {"motion": 1.0}, "motion")
    constructors[expected].assert_called_once()
    constructors[expected].return_value.run.assert_called_once_with()
    constructors["viser" if expected == "native" else "native"].assert_not_called()
    options = constructors[expected].call_args.kwargs
    assert options.get("motion_names") == (["dance1"] if expected == "viser" else None)


@pytest.mark.parametrize("revision, options", [
    ("K1_rev1", ["--robot", "k1_rev1"]),
    ("K1_rev2", ["--robot", "k1_rev2"]),
    ("K1_rev2", []),
])
def test_convert_dance_clip(revision, options, tmp_path):
    import numpy as np
    from source.tasks.supermimic import REPO_ROOT

    source = REPO_ROOT / "source/assets/motions" / revision / "dance1/dance1.npz"
    output = tmp_path / "dance1.npz"
    cli.main(["convert", *options, str(source), str(output)])
    with np.load(source) as reference, np.load(output) as converted:
        np.testing.assert_array_equal(converted["joint_pos"], reference["joint_pos"])
        assert converted["velocity_point"].item() == "link_origin"


@pytest.mark.parametrize("action", ["train", "play"])
@pytest.mark.parametrize("use_task_motion", [False, True])
def test_direct_body_reference_is_adapted_before_upstream(
    action, use_task_motion, monkeypatch, tmp_path
):
    import types

    import numpy as np

    import source.tasks.supermimic as native
    from source.tasks.supermimic import cli as native_cli
    from source.tasks.supermimic import config

    reference = native.REPO_ROOT / "source/assets/motions/K1_rev1/dance1/dance1.npz"
    monkeypatch.setattr(native, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(config, "register_tasks", lambda: None)
    task_id = "Cyclo-SuperMimic-GenCoord-K1-Rev1-Custom"
    monkeypatch.setitem(config.TASKS, task_id, config.TASKS["Cyclo-SuperMimic-GenCoord-K1-Rev1-001"])
    from mjlab.tasks import registry
    from source.tasks.supermimic.config.k1_rev1 import task_001

    env_cfg = task_001.make_env_cfg()
    env_cfg.commands["motion"].motion_file = [str(reference)]
    def load_env_cfg(task, *, play):
        assert play == (action == "play")
        return env_cfg

    monkeypatch.setattr(registry, "load_env_cfg", load_env_cfg)
    module_name = (
        "source.tasks.supermimic.train"
        if action == "train"
        else "source.tasks.supermimic.play"
    )
    module = types.ModuleType(module_name)
    called = []

    def upstream(argv):
        assert argv[0] == task_id
        path = argv[argv.index("--motion-file") + 1]
        assert path != str(reference)
        with np.load(path) as prepared:
            assert prepared["fps"].item() == 50
            assert "root_pos_w" in prepared
            assert prepared["velocity_point"].item() == "link_origin"
        called.append(True)

    module.main = upstream
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cyclo-supermimic",
            action,
            task_id,
            *([] if use_task_motion else ["--motion-file", str(reference)]),
        ],
    )
    original_args = list(sys.argv)
    native_cli.main()
    assert sys.argv == original_args
    assert called == [True]


@pytest.mark.parametrize(
    "options",
    [
        ["--motion-file", "first.npz", "--motion-file", "last.npz"],
        ["--motion-file=first.npz", "--motion-file=last.npz"],
        ["--motion-file", "first.npz", "--motion_file=last.npz"],
    ],
)
def test_duplicate_motion_options_cannot_bypass_adapter(monkeypatch, options):
    from source.tasks.supermimic import cli as native_cli
    from source.tasks.supermimic.reference import motion

    def must_not_adapt(*args, **kwargs):
        raise AssertionError("duplicate options must fail before adaptation")

    monkeypatch.setattr(motion, "prepare_motion_files", must_not_adapt)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "train", "Cyclo-SuperMimic-GenCoord-K1-Rev1-001", *options],
    )
    with pytest.raises(SystemExit) as error:
        native_cli.main()
    assert error.value.code == 2
