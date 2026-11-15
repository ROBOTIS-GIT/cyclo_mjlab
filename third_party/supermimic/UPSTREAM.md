# Provenance

Vendored from https://github.com/HybridRobotics/supermimic at
`abc124c1bd5976341034668af3d12c087f2f1f55` (2026-11-15 checkout).
Copyright 2026 The SuperMimic Authors. Apache-2.0; original LICENSE and
README acknowledgements are preserved. Upstream acknowledges ControlVAE
(BSD-3-Clause), robotic_world_model (Apache-2.0), and DreamerV3 (MIT).

Local changes: `utils/analytic_ik.py` supports one-hinge K1 chains by supplying
an orthogonal auxiliary axis; unused padded coordinates are excluded as before.
The original world model, representations, replay buffer, differentiable
rollouts, policy learners, runner, and exporter are retained.
The sibling-only uv mjlab source in pyproject.toml is changed to the pinned
research git revision above. Python and algorithm dependencies are unchanged.
