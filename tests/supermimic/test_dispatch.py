import sys

import pytest

from scripts.reinforcement_learning import cli, dispatch as router


def test_task_family_names():
    assert router.task_family("Cyclo-Mimic-K1-Rev1-Dance1") is None
    assert router.task_family("Cyclo-BeyondMimic-K1-Rev1-Dance1") == "beyondmimic"
    assert router.task_family("Cyclo-SuperMimic-MaxCoord-K1-Rev1-001") == "supermimic"
    assert router.task_family("Cyclo-Velocity-Flat-K1-Rev1-v0") is None


def test_mismatched_algorithm_rejected_before_import(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--algorithm", "beyondmimic", "Cyclo-SuperMimic-GenCoord-K1-Rev1-001"],
    )
    with pytest.raises(SystemExit) as error:
        cli.prepare_args("train")
    assert error.value.code == 2


def test_wrong_runtime_has_setup_hint(monkeypatch):
    monkeypatch.setattr(cli, "version", lambda name: "1.6.0")
    with pytest.raises(SystemExit, match="locked Python 3.12 runtime"):
        cli.require_runtime()


def test_common_runtime_registers_existing_tasks():
    import source.tasks  # noqa: F401

    from mjlab.tasks.registry import list_tasks

    assert "Cyclo-BeyondMimic-K1-Rev1-Dance1" in list_tasks()
    assert not any(task.startswith("Cyclo-Mimic-") for task in list_tasks())
    assert "Cyclo-Velocity-Flat-K1-Rev1-v0" in list_tasks()


def test_supermimic_wrong_python_rejected(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(sys, "version_info", (3, 13))
        with pytest.raises(SystemExit, match="requires Python 3.12"):
            cli.require_runtime()


def test_shared_runtime_is_accepted(monkeypatch):
    monkeypatch.setattr(cli, "version", lambda name: "1.6.1")
    cli.require_runtime()


@pytest.mark.parametrize("family", ["blinkmimic", "supermimic"])
def test_task_listing_reads_registered_configs(family, monkeypatch, capsys):
    from importlib import import_module

    registry = import_module(f"source.tasks.{family}.config").TASKS
    name = f"Cyclo-{family}-Custom"
    monkeypatch.setitem(registry, name, next(iter(registry.values())))
    monkeypatch.setattr(sys, "argv", ["train.py", "--list-tasks", "--algorithm", family])
    assert cli.prepare_args("train") is None
    assert name in capsys.readouterr().out


def test_beyondmimic_listing_includes_new_registered_tasks(monkeypatch, capsys):
    from mjlab.tasks import registry
    import source.tasks  # noqa: F401

    name = "Cyclo-BeyondMimic-K1-Rev1-NewMotion"
    monkeypatch.setitem(
        registry._REGISTRY, name,
        registry._REGISTRY["Cyclo-BeyondMimic-K1-Rev1-Dance1"],
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--list-tasks", "--algorithm", "beyondmimic"])
    assert cli.prepare_args("train") is None
    assert name in capsys.readouterr().out


@pytest.mark.parametrize("action", ["train", "play"])
@pytest.mark.parametrize("family", ["blinkmimic", "supermimic"])
def test_dispatch_forwards_arguments_without_mutating_argv(action, family, monkeypatch):
    import types

    task = f"Cyclo-{'BlinkMimic' if family == 'blinkmimic' else 'SuperMimic'}-K1-Rev1"
    arguments = [task, "--help"]
    original = list(arguments)
    process_args = ["unrelated-entry.py", "untouched"]
    monkeypatch.setattr(sys, "argv", process_args)
    module = types.ModuleType(f"source.tasks.{family}.cli")
    calls = []
    module.main = lambda *args: calls.append(args)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    assert router.dispatch(action, arguments)
    expected = (action, arguments) if family == "blinkmimic" else ([action, *arguments],)
    assert calls == [expected]
    assert arguments == original
    assert sys.argv is process_args
    assert sys.argv == ["unrelated-entry.py", "untouched"]


@pytest.mark.parametrize("task", ["Cyclo-BeyondMimic-K1-Rev1-Dance1", "Cyclo-Velocity-Flat-K1-Rev1-v0"])
def test_shared_options_removed_before_default_backend(task, monkeypatch):
    monkeypatch.setattr(cli, "require_runtime", lambda: None)
    options = [task, "--help"]
    if "BeyondMimic" in task:
        options = ["--algorithm", "beyondmimic", *options]
    original = list(options)
    monkeypatch.setattr(sys, "argv", ["train.py", *options])
    process_args = sys.argv
    remaining = cli.prepare_args("train", options)

    assert remaining == [task, "--help"]
    assert not router.dispatch("train", remaining)
    assert options == original
    assert sys.argv is process_args
    assert sys.argv == ["train.py", *options]
