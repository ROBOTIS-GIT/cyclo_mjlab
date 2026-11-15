# ProArt validation, 2026-11-15

Repository: `/home/hc/Documents/Codex/k1-walking/cyclo_mjlab`

Worktree: `/home/hc/Documents/Codex/2026-11-15/task-3/feature-supermimic`

Origin: `https://github.com/ROBOTIS-GIT/cyclo_mjlab.git`

Fetched base: `e659f69af227d244a4edab0bb9228e7aeaadef45` (`origin/main`).
Branch: `feature-supermimic`. No pre-existing local or remote branch of that name
was found. Original main remains at `78a27ce` with its existing AMP edits and
untracked work preserved. No push, merge, PR, Isaac Sim edit, or real-robot action.
No `.agents/skills` or `AGENTS.md` exists in this checkout/base.

Runtime: isolated `integrations/supermimic/.venv`, CPython 3.13.18,
mjlab_research 1.6.0 at `dd51a9f`, MuJoCo 3.16.0, Warp 1.19.0,
PyTorch 2.15.0. Exact dependencies: `integrations/supermimic/uv.lock`.
Verification used `/tmp/supermimic-tools/uv`; its downloaded Python lives under
`/tmp/supermimic-python`. Recreate with the documented uv sync command if those
temporary tool directories are removed. Main project dependencies are unchanged.

All following commands ran from the worktree root. Let
`P=integrations/supermimic/.venv/bin/python` and
`C=integrations/supermimic/.venv/bin/cyclo-supermimic`.

| Check | Actual result |
| --- | --- |
| `$P -m pytest tests/supermimic third_party/supermimic/tests -q` | **227 passed**, one upstream torch.jit deprecation warning, 2.00 s (final rerun) |
| `integrations/supermimic/.venv/bin/ruff check integrations/supermimic/src tests/supermimic --select E4,E7,E9,F,I,B` | Passed |
| integration venv `ruff format --check integrations/supermimic/src tests/supermimic` | Passed |
| `python3 -m compileall -q integrations/supermimic/src third_party/supermimic/src tests/supermimic` | Passed |
| `git diff --check` | Passed |
| `$C convert source/assets/motions/K1_rev1/dance1/dance1.npz .supermimic-data/dance1.npz` | Passed with real committed K1 clip; source preserved |
| `$C train Cyclo-SuperMimic-GenCoord-K1-Rev1 --help` | Passed, original nested runner/learner options available |
| `$C smoke --device cpu` | Passed: actual sim transition, dynamics optimizer update, finite nonzero actor gradient through dynamics; loss 0.252228; 23 actions |
| `$C smoke --device cpu --representation maxcoord` | Passed: same path; loss 0.118190; 23 actions |
| `$C smoke --device cuda:0` with 150 s timeout | **Timed out during initial kernel compilation**, exit 124; not a passed GPU smoke |

CPU smoke used `MPLCONFIGDIR=/tmp/cyclo-mjlab-sm-mpl` and
`XDG_CACHE_HOME=/tmp/cyclo-mjlab-sm-cache`; first CPU execution compiled Warp
kernels within a 180-second bound, then MaxCoord reused the cache within a
90-second bound. GPU occupancy was checked outside the sandbox before use:
RTX 5070 Laptop GPU, driver 580.178.04, 13 / 8151 MiB, no compute processes.
The GPU test used only two environments and stopped at its timeout.

## Actual upstream training runner

A three-iteration bounded CPU run uses the real local dance1 clip, upstream
WorldModelLearner and ActorCriticPolicyLearner, without changing their algorithms:

```bash
CUDA_VISIBLE_DEVICES='' \
MPLCONFIGDIR=/tmp/cyclo-mjlab-sm-mpl \
XDG_CACHE_HOME=/tmp/cyclo-mjlab-sm-cache \
WANDB_MODE=offline TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=2 \
timeout 90 integrations/supermimic/.venv/bin/cyclo-supermimic train \
  Cyclo-SuperMimic-GenCoord-K1-Rev1 \
  --motion-file .supermimic-data/dance1.npz --device cpu --num-envs 2 \
  --runner.max-iterations 3 --runner.buffer-size 8 \
  --runner.wm.horizon 2 --runner.wm.batch-size 2 --runner.wm.num-batches 1 \
  --runner.policy.horizon 2 --runner.policy.batch-size 2 --runner.policy.num-batches 1 \
  --runner.dynamics-hidden-dims '(32,32)' --runner.actor-hidden-dims '(32,32)' \
  --runner.critic-hidden-dims '(32,)' --runner.log-interval 1 \
  --runner.save-interval 100 --runner.run-name k1-integration-smoke
```

GenCoord result: **passed, exit 0**, all 3 iterations, real WM losses and policy
updates logged; final `Loss/state=0.83131`, policy gradient norm 0.7537.
MaxCoord result (same command, MaxCoord task, distinct run name): **passed,
exit 0**, all 3 iterations, final `Loss/state=0.17293`, policy gradient norm
0.6230. Both runs produced `actor.pt`, `world_model.pt`, and `critic.pt` under
`logs/Cyclo-SuperMimic-<representation>-K1-Rev1/<timestamp>/`.
The step-0 checkpoint bundles are plumbing evidence, not trained trackers.
Initial sandbox execution failed because W&B's local service socket was denied;
the same offline CPU command passed outside that sandbox. No online W&B sync.
The first CLI attempt used space-separated tuple values; corrected to upstream
mjlab's required Python tuple syntax as shown above.

## Scope and remaining limits

Numerical tests verify all 23 MaxCoord IK angles and rates against MuJoCo FK and
finite gradients. The only vendored algorithm modification supports K1's
single-hinge waist chain; the complete upstream license is byte-identical.
The separate K1 robot adapter supplies explicit collision masks/default priority
required by the research API. Main Cyclo assets, PPO registration and environment
remain untouched.

No converged policy, long training, pretrained quality comparison, interactive
playback of a trained bundle, ONNX deployment validation, or real-robot rollout
was performed. Playback currently uses the original W&B bundle loader; offline
runs need explicit later sync to use that viewer path. GPU performance remains
unverified after the compilation timeout. CPU environment warnings about K1's
extra keyframe and XML integrator are inherited; research scene integration uses
the configured scene integrator, as upstream does.

Raw local evidence is retained under `logs/supermimic-validation/` (git-ignored),
and runner checkpoints under `logs/Cyclo-SuperMimic-*/`. No large artifacts or
datasets are committed.
