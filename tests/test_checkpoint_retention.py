from source.utils.checkpoint_retention import prune_model_checkpoints


def test_retention_keeps_recent_five_and_milestones_only(tmp_path):
    for step in range(0, 27000, 1000):
        (tmp_path / f"model_{step}.pt").touch()
    for name in ("last.pt", "model_27000.pt.tmp", "model_bad.pt", "actor.pt"):
        (tmp_path / name).write_bytes(b"untouched")
    (tmp_path / "model_99999.pt").mkdir()
    prune_model_checkpoints(tmp_path)
    expected = {0, 10000, 20000, 22000, 23000, 24000, 25000, 26000}
    for step in range(0, 27000, 1000):
        assert (tmp_path / f"model_{step}.pt").exists() == (step in expected)
    for name in ("last.pt", "model_27000.pt.tmp", "model_bad.pt", "actor.pt"):
        assert (tmp_path / name).read_bytes() == b"untouched"
    assert (tmp_path / "model_99999.pt").is_dir()
    prune_model_checkpoints(tmp_path)
    assert all((tmp_path / f"model_{step}.pt").exists() for step in expected)
