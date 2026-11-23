import sys

import pytest

from source.tasks.supermimic import cli


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
    monkeypatch.setattr(registry, "load_env_cfg", lambda task: env_cfg)
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
