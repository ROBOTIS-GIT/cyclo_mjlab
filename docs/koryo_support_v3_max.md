# Koryo SupportV3-Max: scratch training with full disturbances

`Cyclo-Mimic-K1-Rev1-Koryo-SupportV3-Max` trains a new policy from random
initialization. Observation noise, reset noise, friction/COM/encoder randomization
and pushes stay at the existing ContactV2-Max stage 8, starting before the first
reset. Push ranges and intervals are unchanged. There is no disturbance ramp or
promotion gate. New contact objectives do not add termination conditions.

The original scratch PPO settings are retained: initial action std 1.0, adaptive
learning rate 0.001, entropy coefficient 0.005, 24 steps/update, action-rate weight
-0.1, and a 30-second training episode cap. Actor observations, action scaling,
and the mixed-start sampler are unchanged. At introduction, the reference motion
was also unchanged to isolate the support objective. As of 2026-10-15, the shared
Koryo default uses the corrected `koryo_support_v2` reference; see
[motion conversion and selection](koryo_support_v2_motion.md).

## Support objective

The task replaces V2's contact-transition term and quiet-stance ankle-motion term.
Existing impact/load-change costs and motion tracking stay active.

| Cost | Initial tuning value | Behavior |
|---|---:|---|
| Missing reference support | 0.45 | Mean debounced missing-contact occupancy across new physics substeps |
| Long unload | 0.15 | Squared `(unload duration - 50ms) / 150ms`, capped at 4; no expiry |
| Unexpected liftoff | 0.02 per event | Count once at debounced liftoff while reference expects support |
| Excess sole speed | 0.15 | Above reference low-point speed + 0.05m/s; scale 0.2m/s, squared and capped at 4 |
| Excess rotation | 0.05 | Above reference tilt/yaw speed + 0.5/1rad/s; scale 2rad/s, capped at 4 |

The combined cost has weight -1 and a 1.5 multiplier during reference single
support. The liftoff count is divided by policy dt because RewardManager scales
rewards by dt; its coefficient is an event cost, unlike the occupancy terms.

Contacts use 20N-on / 5N-off hysteresis and 10ms debounce. There is a 50ms
reference-landing allowance and 150ms reset/discontinuity warmup. Only new
substeps are consumed; history overlap is excluded. Per-environment resets and
reference jumps clear memory. Long gaps are penalized without requiring another
touchdown, and no 150ms recontact expiry makes extended hovering free.

Reference support uses V2's sphere-bottom geometry/hysteresis labels, including
pivots. These are heuristic labels, not hand-annotated contact truth. Motion costs
use low sphere-bottom point velocities computed from COM velocity and angular
velocity. They allow reference movement/pivots plus a tolerance, without the old
quiet-stance gate. A near-lowest sphere set is a support-patch approximation, not
a solver measurement of which individual sphere bears load. Angular costs limit
excess magnitudes, not direction errors; the existing tracking costs still handle
orientation fidelity. Missing-support costs remain active when the foot is unloaded.

The new combined objective ramps from 20% to 100% over 24,000 policy steps
(1,000 PPO updates). **Only reward strength ramps; disturbances remain 100%.**
Recovery steps are possible because the penalties are finite and no contact-based
termination is added. Their necessity is not explicitly classified; the policy
must trade contact quality against tracking/recovery. The coefficients above are
starting hypotheses, not validated optimal settings.

## Start from scratch

Inside the `cyclo_mjlab` container:

```bash
cd /workspace/cyclo_mjlab
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u scripts/reinforcement_learning/train.py \
  Cyclo-Mimic-K1-Rev1-Koryo-SupportV3-Max \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 60000 \
  --agent.run-name koryo_support_v3_max_from_scratch \
  --agent.upload-model False
```

Do not pass `--checkpoint` or enable `--agent.resume` on the first run. The new
runner rejects old checkpoints before loading their weights. A later continuation
may resume only this task's version-3 checkpoint with matching objective and
fixed stage 8, preserving PPO optimizer, iteration, contact-reward ramp and sampler.
Physics/contact memories restart on resume. The initial smoke run is separate and
is never used to initialize the main scratch run.

## Evaluate full disturbances and contact quality

Completion alone cannot detect support-foot hopping. Use multiple seeds and
matched motion windows (especially 10.52–11.82, 19.38–20.74, 49.78–51.16 and
59.82–61.20 seconds). Compare completion/tracking, low-load duty and longest gap,
unexpected liftoffs, sole motion and force peaks. An early fall is not a contact
quality improvement. Raw contact traces can be replayed with the same label and
threshold conventions; force thresholds are simulation diagnostics, not hardware
safety limits.

```bash
python scripts/reinforcement_learning/evaluate_koryo.py \
  --checkpoint logs/rsl_rl/k1_mimic/<run>/model_<iteration>.pt \
  --condition train --noise-scale 1 --push-scale 1 --full-motion --trials 32 \
  --seed 42 --output outputs/support_v3_full_42.json

python scripts/reinforcement_learning/evaluate_koryo_contacts.py \
  --checkpoint logs/rsl_rl/k1_mimic/<run>/model_<iteration>.pt \
  --condition train --seed 42 --output-dir outputs/support_v3_contacts_42
```

`evaluate_koryo_contacts.py --condition train` retains full base-task disturbances
while starting from frame zero. Its existing `clean`/`play` modes remain available
for diagnostics; a clean result does not establish performance under pushes.

The training command saves checkpoints every 500 iterations. It does not run
these full-motion evaluations automatically or select a contact-quality winner.
