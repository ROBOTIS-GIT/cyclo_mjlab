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
