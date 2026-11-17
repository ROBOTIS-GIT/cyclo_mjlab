"""Export a motion-independent policy graph from a supported local checkpoint."""

import argparse
import sys
from pathlib import Path

# Match train/play: prefer task modules from this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("blinkmimic",), required=True)
    parser.add_argument("task", choices=("Cyclo-BlinkMimic-K1-Rev1",))
    args, remaining = parser.parse_known_args()
    from source.tasks.blinkmimic.export import main as export_main

    export_main(remaining)


if __name__ == "__main__":
    main()
