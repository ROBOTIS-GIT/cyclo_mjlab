"""Private subprocess boundary for upstream MJLab performance patches."""

import json
import runpy
import sys
import traceback
from pathlib import Path

from .runtime import install_k1, prepare


def main():
    mode = sys.argv.pop(1)
    if mode not in ("train", "evaluate"):
        raise ValueError(mode)
    prepare()
    install_k1()
    try:
        runpy.run_module(f"blinkmimic.{mode}", run_name="__main__")
    except BaseException as error:
        # Read only Python bookkeeping from the failing frame; do not touch a failed
        # CUDA context or restart. Preserve the exact loop index absent from sparse logs.
        tb = error.__traceback__
        state = {}
        while tb:
            frame = tb.tb_frame
            if (
                frame.f_code.co_filename.endswith("/blinkmimic/train.py")
                and "args" in frame.f_globals
            ):
                g = frame.f_globals
                it = g.get("it")
                state = {
                    "loop_index_zero_based": it,
                    "last_completed_iterations": it,
                    "wm_update_calls": getattr(g.get("wm_update_g"), "calls", None),
                    "policy_update_calls": getattr(
                        g.get("policy_update_g"), "calls", None
                    ),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                out = Path(g["args"].log_dir)
            tb = tb.tb_next
        if state:
            (out / "failure.json").write_text(json.dumps(state, indent=2))
        raise


if __name__ == "__main__":
    main()
