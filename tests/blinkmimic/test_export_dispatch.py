"""The common export CLI preserves each algorithm's checkpoint contract."""

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

from source.tasks.blinkmimic.runtime import ROOT


@pytest.mark.parametrize("algorithm", ["supermimic", "blinkmimic"])
def test_common_export_routes_to_selected_family(monkeypatch, algorithm):
    task = (
        "Cyclo-SuperMimic-GenCoord-K1-Rev1-001"
        if algorithm == "supermimic"
        else "Cyclo-BlinkMimic-K1-Rev1-001"
    )
    flags = [
        "--checkpoint-dir" if algorithm == "supermimic" else "--checkpoint",
        "synthetic",
        "--output-dir",
        "new-output",
    ]
    module = ModuleType(f"source.tasks.{algorithm}.export")
    called = []
    module.main = lambda args: called.append(args)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        sys, "argv", ["export.py", "--algorithm", algorithm, task, *flags]
    )
    runpy.run_path(
        str(ROOT / "scripts/reinforcement_learning/export.py"), run_name="__main__"
    )
    assert called == [[task, *flags]]


def test_blinkmimic_rejects_another_family_task(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["export.py", "--algorithm", "blinkmimic", "Cyclo-SuperMimic-GenCoord-K1-Rev1-001",
         "--checkpoint", "synthetic", "--output-dir", "new-output"],
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(
            str(ROOT / "scripts/reinforcement_learning/export.py"), run_name="__main__"
        )
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_blinkmimic_export_accepts_positional_task(monkeypatch):
    from source.tasks.blinkmimic import export
    from source.tasks.blinkmimic.config import TASKS

    task = next(iter(TASKS))
    called = []
    monkeypatch.setattr(export, "export_policy", lambda *args: called.append(args))
    export.main([task, "--checkpoint", "synthetic", "--output-dir", "new-output"])
    assert len(called) == 1
    assert called[0][:2] == (Path("synthetic"), Path("new-output"))
