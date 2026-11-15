import sys

import pytest
from cyclo_supermimic import cli


def test_k1_training_requires_local_motion(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["cyclo-supermimic", "train", "Cyclo-SuperMimic-GenCoord-K1-Rev1"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("action", ["train", "play"])
def test_direct_body_reference_is_adapted_before_upstream(
    action, monkeypatch, tmp_path
):
    import types

    import numpy as np

    import source.tasks.supermimic as native
    from source.tasks.supermimic import cli as native_cli
    from source.tasks.supermimic import tasks

    reference = native.REPO_ROOT / "source/assets/motions/K1_rev1/dance1/dance1.npz"
    monkeypatch.setattr(native, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tasks, "register_tasks", lambda: None)
    module_name = (
        "source.tasks.supermimic.train"
        if action == "train"
        else "supermimic.scripts.play"
    )
    module = types.ModuleType(module_name)
    called = []

    def upstream():
        path = sys.argv[sys.argv.index("--motion-file") + 1]
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
            "Cyclo-SuperMimic-GenCoord-K1-Rev1",
            "--motion-file",
            str(reference),
        ],
    )
    native_cli.main()
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
    from source.tasks.supermimic import motion

    def must_not_adapt(*args, **kwargs):
        raise AssertionError("duplicate options must fail before adaptation")

    monkeypatch.setattr(motion, "prepare_motion_files", must_not_adapt)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "train", "Cyclo-SuperMimic-GenCoord-K1-Rev1", *options],
    )
    with pytest.raises(SystemExit) as error:
        native_cli.main()
    assert error.value.code == 2
