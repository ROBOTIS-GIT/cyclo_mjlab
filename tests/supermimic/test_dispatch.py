import sys

import pytest

from source.learning import dispatch as router


def test_task_family_names():
    assert router.task_family("Cyclo-Mimic-K1-Rev1-Dance1") == "beyondmimic"
    assert router.task_family("Cyclo-BeyondMimic-K1-Rev1-Dance1") == "beyondmimic"
    assert router.task_family("Cyclo-SuperMimic-MaxCoord-K1-Rev1") == "supermimic"
    assert router.task_family("Cyclo-Velocity-Flat-K1-Rev1-v0") is None


def test_mismatched_algorithm_rejected_before_import(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--algorithm", "beyondmimic", "Cyclo-SuperMimic-GenCoord-K1-Rev1"],
    )
    with pytest.raises(SystemExit) as error:
        router.dispatch("train")
    assert error.value.code == 2


def test_legacy_wrong_runtime_has_setup_hint(monkeypatch):
    monkeypatch.setattr(router, "version", lambda name: "1.6.0")
    with pytest.raises(SystemExit, match="existing Cyclo Docker"):
        router.require_runtime("beyondmimic")


def test_research_runtime_does_not_register_legacy():
    assert "source.tasks.mimic" not in sys.modules
    assert "source.tasks.velocity" not in sys.modules


def test_supermimic_wrong_python_rejected(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(sys, "version_info", (3, 12))
        with pytest.raises(SystemExit, match="requires Python 3.13"):
            router.require_runtime("supermimic")
