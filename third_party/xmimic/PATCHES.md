X-Mimic is vendored at the revision in UPSTREAM. LICENSE and
THIRD_PARTY_NOTICES.md are the original upstream files. This checkout includes
the learner and evaluator. Upstream data, G1 meshes, tools, and trained policies
are obtained separately if needed.

Cyclo adds an optional scene directory, an explicit contact-body mapping, and
an ordered manifest path for native states with reference hash checks. The original CSV loader
and training recipe are retained. Checkpoints record the robot's joint order,
nominal pose and action scales so evaluation and export can check the selected
task. The evaluator also rejects mixed checkpoint contracts in one run.
