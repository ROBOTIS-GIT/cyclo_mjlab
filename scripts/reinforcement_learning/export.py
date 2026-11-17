"""Export a motion-independent policy graph from a supported local checkpoint."""

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


EXPORTERS = {
    "supermimic": "source.tasks.supermimic.export",
    "blinkmimic": "source.tasks.blinkmimic.export",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--algorithm", choices=tuple(EXPORTERS), required=True)
    if len(sys.argv) == 1 or sys.argv[1:] in (["-h"], ["--help"]):
        parser.print_help()
        return
    args, remaining = parser.parse_known_args()
    importlib.import_module(EXPORTERS[args.algorithm]).main(remaining)


if __name__ == "__main__":
    main()
