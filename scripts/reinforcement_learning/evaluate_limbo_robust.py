"""Fixed nominal, deployed-entry and held-out noisy/disturbed entry evaluations."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse,sys,json
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from source.tasks.mimic.config.k1_rev1.limbo_robust_env_cfg import k1_rev1_limbo_robust_env_cfg,k1_rev1_limbo_robust_runner_cfg
from source.tasks.mimic.limbo_robust_runner import RobustRunner
from source.tasks.mimic.mdp import limbo_prepare_obs124 as prep,limbo_1p2_v2
from limbo_eval_utils import reset_bank


def evaluate(checkpoint,output):
    torch.set_num_threads(2)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reports=[]
    for profile,level in [('nominal',0.),('runtime_entry',0.),('heldout_medium',.5),('heldout_full',1.)]:
        heldout=profile.startswith('heldout')
        cfg=k1_rev1_limbo_robust_env_cfg(play=not heldout)
        cfg.scene.num_envs=32 if heldout else 65
        cfg.seed=20261030 if heldout else 42
        c=cfg.commands['reference_trajectory']
        c.debug_vis=False;c.start_from_zero=True;c.evaluation_difficulty=level
        c.reset_pool_seed=20261129  # Disjoint from the training reset pool.
        c.randomize_entry=heldout;c.enable_entry_push=heldout;c.push_probability=1.
        def capture(env):
            cmd=prep.command(env)
            env.eval_terminal={
                'success':prep.full_success(env).cpu().numpy().copy(),
                'physical':limbo_1p2_v2.full_success(env).cpu().numpy().copy(),
                'prepared':cmd.prefix_ok.cpu().numpy().copy(),
                'push':cmd.push_done.cpu().numpy().copy()}
            return prep.full_success(env).float()
        cfg.rewards['limbo_full_success'].func=capture
        env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
        agent=k1_rev1_limbo_robust_runner_cfg()
        wrap=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        runner=RobustRunner(wrap,asdict(agent),device='cuda:0')
        try:
            runner.load(str(checkpoint),load_cfg={'actor':True},strict=True,map_location='cuda:0')
            seed=20261030 if heldout else 42
            if heldout:
                env.reset(seed=seed);cmd=prep.command(env);obs=wrap.get_observations()
            else:
                cmd,obs=reset_bank(env,wrap,seed)
            cmd.entry_duration[:]=0. if profile=='nominal' else 1.
            cmd.entry_start_gain[:]=1.
            active=np.ones(env.num_envs,bool);peak=np.zeros(env.num_envs);jump=np.zeros(env.num_envs);cases=[]
            with torch.no_grad():
                for step in range(552):
                    action=runner.alg.actor(obs,stochastic_output=False)
                    if not torch.isfinite(action).all():
                        raise RuntimeError(f'Nonfinite actor output in {profile}, step {step}')
                    if step<100:
                        peak=np.maximum(peak,np.where(active,action.abs().amax(-1).cpu().numpy(),0.))
                    obs,reward,done,extras=wrap.step(action)
                    if step<100:
                        delta=env.action_manager.get_term('joint_pos').goal_delta.abs().amax(-1).cpu().numpy()
                        jump=np.maximum(jump,np.where(active,delta,0.))
                    for i in np.flatnonzero(active & done.cpu().numpy().astype(bool)):
                        row=env.eval_terminal
                        cases.append({'case':int(i),'success':bool(row['success'][i]),'physical':bool(row['physical'][i]),
                            'prepared':bool(row['prepared'][i]),'push_delivered':bool(row['push'][i]),
                            'elapsed_s':(step+1)*.02,'first2s_peak_raw_action':float(peak[i]),'first2s_peak_target_jump_rad':float(jump[i]),
                            'reasons':[name for name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(name)[i])]})
                        active[i]=False
                    if not active.any():break
            if active.any():raise RuntimeError(f'Incomplete evaluation: {profile}')
            r={'profile':profile,'seed':seed,'difficulty':level,'cases':cases,'num_cases':len(cases),
               'successes':sum(x['success'] for x in cases),'prepared':sum(x['prepared'] for x in cases),
               'success_rate':sum(x['success'] for x in cases)/len(cases),
               'first2s_peak_raw_action':max(x['first2s_peak_raw_action'] for x in cases),
               'first2s_peak_target_jump_rad':max(x['first2s_peak_target_jump_rad'] for x in cases)}
            reports.append(r)
            (output/'evaluation.json').write_text(json.dumps(reports,indent=2))
            print('ROBUST_EVAL',json.dumps({k:v for k,v in r.items() if k!='cases'}),flush=True)
        finally:
            env.close();del runner,wrap,env;torch.cuda.empty_cache()
    return reports


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();evaluate(a.checkpoint,a.output)
