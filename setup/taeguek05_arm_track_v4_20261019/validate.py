"""Bounded integration check; never considered a production resume candidate."""
import argparse
from dataclasses import asdict
import datetime
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / f'cyclo-matplotlib-{os.getuid()}'))
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from refinement import env_cfg, agent_cfg, ArmRunner, SOURCE, validate_inputs, validate_resume, revision
from launch import choose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--steps', type=int, default=256)
    args = parser.parse_args()
    validate_inputs(); configure_torch_backends()
    os.chdir(ROOT)
    tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logdir = ROOT / 'logs/rsl_rl/k1_mimic' / f'taeguek05_arm_track_v4_smoke_{tag}'
    cfg = env_cfg(); cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env)
    runner = ArmRunner(wrapped, asdict(agent_cfg()), str(logdir), args.device)
    runner.load(SOURCE)
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    for model, key in ((runner.alg.actor, 'actor_state_dict'), (runner.alg.critic, 'critic_state_dict')):
        for name, value in source[key].items():
            torch.testing.assert_close(model.state_dict()[name].cpu(), value, rtol=0, atol=0)
    runner.learn(num_learning_iterations=3, init_at_random_ep_len=True)
    checkpoint = logdir / 'model_2.pt'
    first = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert validate_resume(first, args.num_envs) == 3
    # Use a fresh runner/environment, as a real stop-and-resume would.
    env.close()
    cfg = env_cfg(); cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env)
    resumed = ArmRunner(wrapped, asdict(agent_cfg()), str(logdir / 'resume'), args.device)
    resumed.load(checkpoint)
    assert resumed.current_learning_iteration == 3 and env.common_step_counter == 72
    resumed.learn(num_learning_iterations=1, init_at_random_ep_len=True)
    second = torch.load(logdir / 'resume/model_3.pt', map_location='cpu', weights_only=False)
    assert validate_resume(second, args.num_envs) == 4
    step1 = min(float(v['step']) for v in first['optimizer_state_dict']['state'].values())
    step2 = min(float(v['step']) for v in second['optimizer_state_dict']['state'].values())
    assert step2 > step1 > 0
    term = env.reward_manager.get_term_cfg('arm_recovery').func
    env.common_step_counter = 24000
    policy = resumed.get_inference_policy(args.device)
    obs = wrapped.get_observations()
    totals, peaks = {}, {}
    max_age = 0.
    for _ in range(args.steps):
        with torch.inference_mode():
            obs, rewards, dones, extras = wrapped.step(policy(obs))
        assert torch.isfinite(rewards).all() and torch.isfinite(term.cached).all()
        max_age = max(max_age, float(term.age.max()))
        for name, value in term.cmd.metrics.items():
            if name.startswith('arm_v4_'):
                assert torch.isfinite(value).all(), name
                totals[name] = totals.get(name, 0.) + float(value.mean())
                peaks[name] = max(peaks.get(name, 0.), float(value.max()))
    assert max_age > .15 and totals['arm_v4_joint'] > 0
    # Production chooser must not pick these smoke updates.
    assert choose(args.num_envs)[0] == SOURCE
    result = dict(device=args.device, num_envs=args.num_envs, policy_steps=args.steps,
        completed_updates=[3, 4], optimizer_steps=[step1, step2], revision=revision(),
        maximum_continuous_age_s=max_age, metrics_mean={k:v/args.steps for k,v in totals.items()},
        metrics_max=peaks, source_actor_critic_exact=True, smoke_excluded=True,
        logdir=str(logdir), cuda_peak_memory_mb=(torch.cuda.max_memory_allocated()/2**20
                                               if args.device.startswith('cuda') else None))
    (HERE / f'validation_{args.device.replace(":", "_")}.json').write_text(json.dumps(result, indent=2))
    print('VALIDATION_OK', json.dumps(result), flush=True)
    env.close()


if __name__ == '__main__':
    main()
