"""Checkout-only compatibility imports for the common native runtime."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if not (REPO_ROOT / "source/tasks/supermimic/cli.py").is_file():
    raise ImportError(
        "cyclo-supermimic is a checkout-only compatibility package. "
        "Install it editable from the Cyclo checkout in the common Python 3.12 runtime; "
        "standalone wheel installation is unsupported."
    )
sys.path.insert(0, str(REPO_ROOT))
