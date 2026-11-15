"""Convert a Cyclo NPZ for the native SuperMimic task family."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from source.learning.dispatch import require_runtime

if __name__ == "__main__":
    require_runtime("supermimic")
    from source.tasks.supermimic.cli import main

    sys.argv.insert(1, "convert")
    main()
