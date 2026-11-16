"""Private subprocess boundary for upstream MJLab performance patches."""

import runpy
import sys

from .runtime import install_k1, prepare


def main():
    mode = sys.argv.pop(1)
    if mode not in ("train", "evaluate"):
        raise ValueError(mode)
    prepare()
    install_k1()
    runpy.run_module(f"blinkmimic.{mode}", run_name="__main__")


if __name__ == "__main__":
    main()
