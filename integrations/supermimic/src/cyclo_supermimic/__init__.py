"""Compatibility package for the original isolated-runtime entry point."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if not (REPO_ROOT / "source/tasks/supermimic/cli.py").is_file():
    raise ImportError(
        "cyclo-supermimic is a checkout-only compatibility package. "
        "Run uv sync --project integrations/supermimic from the Cyclo checkout; "
        "standalone wheel installation is unsupported."
    )
sys.path.insert(0, str(REPO_ROOT))
