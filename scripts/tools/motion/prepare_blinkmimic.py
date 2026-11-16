"""Create a portable BlinkMimic library from declared compact K1 references."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

if __name__ == "__main__":
    from source.tasks.blinkmimic.prepare import main

    main()
