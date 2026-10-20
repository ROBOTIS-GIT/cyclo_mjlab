"""Readiness check: source policy rollout; no weight updates."""
from dataclasses import asdict
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from refinement import env_cfg, agent_cfg, ArmRunner, SOURCE

configure_torch_backends()
cfg = env_cfg(); cfg.scene.num_envs = 32
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
wrapped = RslRlVecEnvWrapper(env)
runner = ArmRunner(wrapped, asdict(agent_cfg()), None, 'cuda:0')
runner.load(SOURCE)
env.common_step_counter = 24000
policy = runner.get_inference_policy('cuda:0')
cmd = env.command_manager.get_term('reference_trajectory')
term = env.reward_manager.get_term_cfg('arm_stability').func
names = [cmd.robot.joint_names[j] for j in term.arms]
coverage = []
for name, lo, hi in [('right_shoulder_roll_joint',23.5,24),
                     ('left_shoulder_roll_joint',30.25,30.75),
                     ('left_shoulder_roll_joint',33.7,34.2),
                     ('left_wrist_roll_joint',35.2,36.15),
                     ('left_shoulder_roll_joint',39,39.6),
                     ('right_wrist_roll_joint',41.375,41.875)]:
    j = names.index(name); gate = term.hold[round(lo*50):round(hi*50),j]
    coverage.append(dict(joint=name, interval=[lo,hi], gate_mean=float(gate.mean()),
                         gate_max=float(gate.max())))
print('HOLD_GATE_COVERAGE', json.dumps(coverage), flush=True)
obs = wrapped.get_observations(); maximum_age = 0.; max_frame = 0; total = 0.; peak = 0.
for step in range(2400):
    with torch.inference_mode():
        obs, rewards, dones, extras = wrapped.step(policy(obs))
    assert torch.isfinite(rewards).all() and torch.isfinite(term.cached).all()
    maximum_age = max(maximum_age, float(term.age.max()))
    max_frame = max(max_frame, int(cmd.frame_ids.max()))
    total += float(term.cached.mean()); peak = max(peak, float(term.cached.max()))
    if step % 600 == 0:
        print('STEP', step, 'max_age', maximum_age, 'max_frame', max_frame, flush=True)
assert maximum_age > 1. and max_frame > 2200 and total > 0
result = dict(policy_steps=2400, parallel_envs=32, maximum_continuous_age_s=maximum_age,
              maximum_reference_frame=max_frame, full_ramp_penalty_mean=total/2400,
              full_ramp_penalty_max=peak, gate_coverage=coverage, weights_updated=False)
print('PREFLIGHT_OK', json.dumps(result), flush=True)
(Path(__file__).parent/'preflight.json').write_text(json.dumps(result, indent=2))
env.close()
