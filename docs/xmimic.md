# X-Mimic

X-Mimic runs the upstream JAX/MJX learner on a Cyclo robot scene. Cyclo prepares
the scene and exact 50 Hz reference states in its normal PyTorch/MJLab
environment. Training and evaluation run in a separate JAX environment so their
MuJoCo and Warp versions can coexist. The initial task uses the public K1 Rev.1
robot configuration and the same motion validation as BlinkMimic.

Install `third_party/xmimic/requirements.txt` in a dedicated Python environment
with CUDA support. From the Cyclo environment, pass the absolute path to that
environment's Python interpreter. Prepare the reference manifest using the
[BlinkMimic motion guide](blinkmimic.md).

```bash
python scripts/reinforcement_learning/train.py Cyclo-XMimic-K1-Rev1-001 \
  --manifest /path/to/motions/manifest.json --run-dir logs/xmimic/run \
  --jax-python /path/to/jax-env/bin/python
python scripts/reinforcement_learning/play.py Cyclo-XMimic-K1-Rev1-001 \
  --run-dir logs/xmimic/run --jax-python /path/to/jax-env/bin/python
```

The trainer reads all clips in manifest order and learns one policy. Additional
arguments are upstream trainer options. The default layout uses one GPU, 4096
environments and the published network widths. The evaluator uses one
checkpoint per batch and four replicas per clip by default. The run directory
contains the scene and prepared references, so keep it outside version control.

```bash
python scripts/reinforcement_learning/export.py --algorithm xmimic \
  Cyclo-XMimic-K1-Rev1-001 --checkpoint logs/xmimic/run/ckpt_003500.npz \
  --output-dir logs/xmimic/deploy
python -m source.tasks.xmimic.motion_csv Cyclo-XMimic-K1-Rev1-001 \
  /path/to/clip.npz logs/xmimic/deploy/params/clip.csv
```

Export loads the averaged checkpoint, checks the ordered robot contract and
transposes dense kernels for the shared BlinkMimic inference implementation.
It emits FP32 ONNX, Sim2Real YAML and software notices. With the default K1
recipe its input is 1420 raw features and its output is 23 normalized actions.
The BlinkMimic CSV runtime supplies the goal externally and computes velocities
from pose samples. This may differ from the supplied velocities used during
exact-state training. The CSV tool preserves poses and joint order and converts
quaternions from wxyz to xyzw. It takes an individual clip so one policy can be
paired with several motion files without embedding them in the graph.
