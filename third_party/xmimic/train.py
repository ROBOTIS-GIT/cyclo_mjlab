"""Train XMimic.

    python train.py                       # one GPU
    python train.py --layout AWLP         # four GPUs (layouts: 1GPU, FP1, FP, AL, ALP, AWL, AWLP)

Every option of xmimic/config.py is a flag (--name value); the defaults are the published H100 configuration. Checkpoints,
train_log.csv and args.json go to --log_dir; evaluate them with python -m xmimic.evaluate <log_dir>.
"""

import os


def runtime_defaults():
    """XLA and JAX settings, unless set already (they must be in place before JAX starts, and spawned prober processes
    inherit them): cuBLAS for every GEMM (Triton GEMM fusions are slower for these sizes on Hopper), 90 % of the GPU memory
    preallocated, and a persistent compilation cache (with XLA's autotuning cache) so that later runs load their programs
    instead of compiling them. Python tracebacks are left out of the compiled programs' metadata, which halves cache loads."""
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_enable_triton_gemm" not in flags:
        os.environ["XLA_FLAGS"] = (flags + " --xla_gpu_enable_triton_gemm=false").strip()
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "xmimic", "jax"))
    os.environ.setdefault("JAX_INCLUDE_FULL_TRACEBACKS_IN_LOCATIONS", "false")


def main(argv=None):
    runtime_defaults()
    from xmimic.config import parse_args

    cfg = parse_args(argv)
    if cfg.layout == "1GPU":
        from xmimic import single_gpu

        single_gpu.run(cfg)
    else:
        from xmimic import multi_gpu

        multi_gpu.run(cfg)


if __name__ == "__main__":
    main()
