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
import yaml
from source.utils import export_sim2real_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from refinement import env_cfg, agent_cfg, BalancedRunner, SOURCE, validate_inputs, validate_resume, revision
from launch import choose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--steps', type=int, default=256)
    args = parser.parse_args()
    validate_inputs(); configure_torch_backends(); torch.set_num_threads(4)
    os.chdir(ROOT)
    tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logdir = ROOT / 'logs/rsl_rl/k1_mimic' / f'taeguek06_precision_support_v4_finetune_smoke_{tag}'
    cfg = env_cfg(); cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg, device=args.device)
    export_sim2real_cfg(env,logdir)
    actual=yaml.safe_load((logdir/'params/sim2real.yaml').read_text())
    expected=yaml.safe_load((HERE/'source/params/sim2real.yaml').read_text())
    assert actual==expected,'Policy interface/PD changed'
    support=env.reward_manager.get_term_cfg('t6_support').func
    fps=float(support.cmd.reference.fps)
    assert fps==50 and abs(env.step_dt-.02)<1e-10 and cfg.decimation==4
    coverage={}
    for name,lo,hi,foot in [('kick1',5.90,5.92,0),('kick2',8.96,8.98,1),('kick3',11.68,11.74,0)]:
        mask=support.kick_support[round(lo*fps):round(hi*fps)+1,foot]
        coverage[name]=dict(start_s=lo,end_s=hi,fraction=float(mask.float().mean()))
    assert all(v['fraction']==1 for v in coverage.values()),coverage
    torch.save({'kick_support':support.kick_support.cpu(),'base_support':support.reference_support.cpu()},HERE/'coverage_masks.pt')
    assert sum(k=='t6_support' for k in cfg.rewards)==1
    wrapped = RslRlVecEnvWrapper(env)
    runner = BalancedRunner(wrapped, asdict(agent_cfg()), str(logdir), args.device)
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
    resumed = BalancedRunner(wrapped, asdict(agent_cfg()), str(logdir / 'resume'), args.device)
    resumed.load(checkpoint)
    assert resumed.current_learning_iteration == 3 and env.common_step_counter == 72
    cmd=env.command_manager.get_term('reference_trajectory')
    torch.testing.assert_close(cmd.failure_score_by_bin.cpu(),first['infos']['sampling'],rtol=0,atol=0)
    for k,v in resumed.alg.actor.state_dict().items():
        torch.testing.assert_close(v.cpu(),first['actor_state_dict'][k],rtol=0,atol=0)
    resumed.learn(num_learning_iterations=1, init_at_random_ep_len=True)
    second = torch.load(logdir / 'resume/model_3.pt', map_location='cpu', weights_only=False)
    assert validate_resume(second, args.num_envs) == 4
    step1 = min(float(v['step']) for v in first['optimizer_state_dict']['state'].values())
    step2 = min(float(v['step']) for v in second['optimizer_state_dict']['state'].values())
    assert step2 > step1 > 0
    term = env.reward_manager.get_term_cfg('t6_arm_recovery').func
    env.common_step_counter = 120000
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
            if name.startswith(('t6_',)):
                assert torch.isfinite(value).all(), name
                totals[name] = totals.get(name, 0.) + float(value.mean())
                peaks[name] = max(peaks.get(name, 0.), float(value.max()))
    assert max_age > .15 and totals['t6_arm_joint_error_rad'] > 0
    assert totals['t6_kick_support_extra']>0,totals
    assert peaks['t6_kick_support_ramp']==1.
    assert peaks['t6_rebound_ramp']==1. and totals['t6_rebound_extra']>0
    assert totals['t6_grf_tail_extra']>0
    assert totals['t6_precision_extra']>0 and totals['t6_path_precision_cost']>0
    assert peaks['t6_precision_ramp']==1.
    support=env.reward_manager.get_term_cfg('t6_support').func
    with torch.inference_mode():
        torch.testing.assert_close(support(env),support(env),rtol=0,atol=0)
        support.reset(torch.tensor([0],device=args.device))
    assert support.rebound_frame[0]==-1 and not support.previous_expected[0].any()
    # Production chooser must not pick these smoke updates.
    assert choose(args.num_envs)[0] == SOURCE
    result = dict(device=args.device, num_envs=args.num_envs, policy_steps=args.steps,
        completed_updates=[3, 4], optimizer_steps=[step1, step2], revision=revision(),
        maximum_continuous_age_s=max_age, metrics_mean={k:v/args.steps for k,v in totals.items()},
        metrics_max=peaks, source_actor_critic_exact=True, smoke_excluded=True, sim2real_matches_source=True, coverage=coverage, production_started=False,
        logdir=str(logdir), cuda_peak_memory_mb=(torch.cuda.max_memory_allocated()/2**20
                                               if args.device.startswith('cuda') else None))
    (HERE / f'validation_{args.device.replace(":", "_")}.json').write_text(json.dumps(result, indent=2))
    print('VALIDATION_OK', json.dumps(result), flush=True)
    env.close()


if __name__ == '__main__':
    main()
