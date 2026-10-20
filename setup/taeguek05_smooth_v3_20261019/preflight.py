"""GPU validation of actual masks and policy rollouts before user training."""
from dataclasses import asdict
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from refinement import env_cfg, agent_cfg, SmoothRunner, SOURCE

configure_torch_backends()
cfg = env_cfg()
cfg.scene.num_envs = 32
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
wrapped = RslRlVecEnvWrapper(env)
runner = SmoothRunner(wrapped, asdict(agent_cfg()), None, 'cuda:0')
runner.load(SOURCE)
env.common_step_counter = 24000  # Inspect the final reward strength, without training.
policy = runner.get_inference_policy('cuda:0')
cmd = env.command_manager.get_term('reference_trajectory')
support = env.reward_manager.get_term_cfg('taeguek_support_quality').func
smooth = env.reward_manager.get_term_cfg('reference_residual_smoothness').func
coverage = []
for seconds in (14.07, 35.39, 15.38, 19.22, 29.04):
    f = round(seconds * cmd.reference.fps)
    coverage.append(dict(seconds=seconds, height=support.reference_bottom[f, 1].item(),
        lowpoint_speed=support.reference_lowpoint_speed[f, 1].item(),
        moving_eligible=bool(support.moving_eligible[f, 1]), support=bool(support.reference_support[f, 1])))
print('MASK_COVERAGE', json.dumps(coverage), flush=True)
obs = wrapped.get_observations()
maximum_age = 0.
max_frame = 0
cost_sum = 0.
cost_max = 0.
for step in range(2050):
    if step < 3:
        print('BEFORE_STEP', step, 'frame', cmd.frame_ids[:3].tolist(), 'z_actual',
            cmd.robot_body_pos_w[:1, :, 2].tolist(), 'z_aligned', cmd.aligned_body_position_w[:1, :, 2].tolist(), flush=True)
    with torch.inference_mode():
        obs, reward, done, extras = wrapped.step(policy(obs))
    assert torch.isfinite(reward).all()
    maximum_age = max(maximum_age, float(smooth.age.max()))
    max_frame = max(max_frame, int(cmd.frame_ids.max()))
    cost_sum += float(smooth.cached.mean())
    cost_max = max(cost_max, float(smooth.cached.max()))
    assert torch.isfinite(smooth.cached).all()
    if step < 5 or step % 512 == 0:
        print('STEP', step, 'done', int(done.sum()), 'age', float(smooth.age.max()),
            'reward', float(reward.mean()), 'frame', cmd.frame_ids[:3].tolist(), flush=True)
assert maximum_age > .5, ('All environments reset too early to exercise smoothness', maximum_age)
assert max_frame >= 2000, ('Full reference tail was not reached', max_frame)
assert cost_sum > 0, 'New reward did not activate'
assert (support.reference_kick & support.moving_eligible).sum() == 0
result = dict(maximum_continuous_age_s=maximum_age, maximum_reference_frame=max_frame,
    policy_steps=2050, parallel_envs=32, full_ramp_smoothness_mean=cost_sum/2050,
    full_ramp_smoothness_max=cost_max, coverage=coverage)
print('PREFLIGHT_OK', json.dumps(result), flush=True)
(Path(__file__).parent/'preflight.json').write_text(json.dumps(result, indent=2))
env.close()
