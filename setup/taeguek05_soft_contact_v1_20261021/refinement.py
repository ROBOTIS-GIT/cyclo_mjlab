"""Taegeuk 5: mature ArmTrackV4 plus gradual soft touchdown and settling."""
from pathlib import Path
from collections import deque
import hashlib,json,math
import numpy as np
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from arm_recovery import ArmRecovery
from contact import SoftLanding
import smooth_v3_base as base

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=HERE/'source/model_9999.pt'
MOTION=HERE/'motion/k1_taeguek05_left_arm_refined_50hz_converted.npz'
MARKER='taeguek05_soft_contact_v1'
RUN_NAME=MARKER+'_finetune'
TOTAL_UPDATES=10000
RAMP_STEPS=120000

def revision():
    h=hashlib.sha256()
    paths=[HERE/n for n in ['refinement.py','contact.py','arm_recovery.py','smooth_v3_base.py','inputs.json']]
    paths += [ROOT/'source/tasks/mimic/mdp'/n for n in ['support_contacts.py','taeguek_support.py','taeguek_support_v2.py','performance_contacts.py','contact_transitions.py','commands.py']]
    paths += [ROOT/'source/tasks/mimic/config/k1_rev1'/n for n in ['base_env_cfg.py','taeguek_env_cfg.py','taeguek_support_env_cfg.py','taeguek_support_v2_env_cfg.py','agents/rsl_rl_ppo_cfg.py']]
    paths += [ROOT/'source/tasks/mimic/tracking_env_cfg.py',ROOT/'source/assets/robots/robotis_k1/k1_rev1.py']
    for p in paths:h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()

def validate_inputs():
    for name,h in json.loads((HERE/'inputs.json').read_text()).items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=h:raise ValueError('Input changed: '+name)
    z=np.load(MOTION,allow_pickle=False);a=np.loadtxt(MOTION.with_suffix('.csv'),delimiter=',')
    assert z['joint_pos'].shape==(2017,23) and float(z['fps'][0])==50.
    assert np.isfinite(a).all() and np.max(abs(a[:,7:]-z['joint_pos']))<1e-6
    for k in ['joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:assert np.isfinite(z[k]).all()

def env_cfg(play=False):
    cfg=base.env_cfg(play)
    cfg.commands['reference_trajectory'].trajectory_file=str(MOTION)
    cfg.scene.num_envs=1 if play else 4096
    cfg.rewards['taeguek_support_quality'].params['new_ramp_steps']=1
    cfg.rewards['reference_residual_smoothness'].params['ramp_steps']=1
    cfg.rewards['arm_recovery']=RewardTermCfg(func=ArmRecovery,weight=-1.,params={'ramp_steps':1})
    # Existing support/clearance/kick objectives remain single and fully mature.
    cfg.rewards['soft_landing']=RewardTermCfg(func=SoftLanding,weight=-1.,params={'ramp_steps':RAMP_STEPS})
    return cfg

def agent_cfg():
    cfg=base.agent_cfg();cfg.algorithm.learning_rate=2e-5
    cfg.algorithm.schedule='fixed';cfg.algorithm.entropy_coef=.001
    cfg.max_iterations=TOTAL_UPDATES;cfg.save_interval=500;cfg.run_name=RUN_NAME
    cfg.resume=False;cfg.upload_model=False
    return cfg

def guard_update(state,iteration,values):
    s=dict(state)
    if iteration<=s.get('last_iteration',-1):return s
    if not all(math.isfinite(x) for x in values.values()):
        return {**s,'last_iteration':iteration,'halt':True,'reasons':['nonfinite_metrics']}
    # Calibrate to this 5th-chapter policy, not absolute thresholds from chapter 7.
    if 'baseline' not in s and iteration>=500:s['baseline']=dict(values)
    reasons=[];base=s.get('baseline',values)
    if iteration>=1000:
        if values['length']<max(600,.60*base['length']):reasons.append('episode_length')
        for k in ['arm','hand','joint']:
            if values[k]>1.8*base[k]+(.02 if k=='hand' else .05):reasons.append(k)
    s.update(last_iteration=iteration,values=values,reasons=reasons,
             bad_checks=s.get('bad_checks',0)+1 if reasons else 0)
    s['halt']=s['bad_checks']>=2
    return s


def validate_resume(d,num_envs=None):
    m=(d.get('infos') or {}).get(MARKER)
    if not m or m.get('revision')!=revision():raise ValueError('Reward/config/input revision mismatch')
    n=int(d['iter'])+1
    # The launcher owns the requested target; completed runs can exceed the default.
    if n<=0:raise ValueError('Invalid completed update count')
    if d['infos']['env_state']['common_step_counter']!=n*24:raise ValueError('Environment counter mismatch')
    if num_envs is not None and m['num_envs']!=num_envs:raise ValueError('Environment count changed')
    if not d.get('optimizer_state_dict',{}).get('state'):raise ValueError('Missing optimizer state')
    if not all(torch.isfinite(v).all() for k in ['actor_state_dict','critic_state_dict'] for v in d[k].values()):raise ValueError('Nonfinite checkpoint')
    if d['infos'].get('tracking_guard',{}).get('halt'):raise ValueError('Tracking guard halted; inspect before resume')
    return n


class SoftContactRunner(MjlabOnPolicyRunner):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.guard_state={};self.window=deque(maxlen=50)
        self.provenance={'revision':revision(),'num_envs':self.env.unwrapped.num_envs,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
        original=self.logger.log
        def log(*args,**kwargs):
            vals={}
            for name,key in {'arm':'arm_v4_joint_error_rad','hand':'arm_v4_hand_error_m','joint':'error_joint_pos'}.items():
                ts=[torch.as_tensor(e['Metrics/reference_trajectory/'+key],device=self.device).float().reshape(-1) for e in self.logger.ep_extras if 'Metrics/reference_trajectory/'+key in e]
                if ts:vals[name]=torch.cat(ts).mean().item()
            if self.logger.lenbuffer:vals['length']=sum(self.logger.lenbuffer)/len(self.logger.lenbuffer)
            if len(vals)==4:self.window.append(vals)
            return original(*args,**kwargs)
        self.logger.log=log

    def load(self,path,load_cfg=None,strict=True,map_location=None):
        if load_cfg is not None:raise ValueError('Use managed loading')
        d=torch.load(path,map_location='cpu',weights_only=False)
        if (d.get('infos') or {}).get(MARKER):
            n=validate_resume(d,self.env.unwrapped.num_envs)
            result=super().load(path,None,strict,map_location or self.device)
            self.current_learning_iteration=n;self.guard_state=d['infos'].get('tracking_guard',{})
            self.provenance=d['infos'][MARKER]
            print('T5_SOFT_BALANCED_RESUME',n,flush=True);return result
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=self.provenance['source_sha256']:raise ValueError('Wrong warmstart source')
        if not all(torch.isfinite(v).all() for k in ['actor_state_dict','critic_state_dict'] for v in d[k].values()):raise ValueError('Nonfinite source')
        result=super().load(path,dict(actor=True,critic=True,optimizer=False,iteration=False,rnd=False),strict,map_location or self.device)
        for model,key in [(self.alg.actor,'actor_state_dict'),(self.alg.critic,'critic_state_dict')]:
            if not all(torch.equal(v.cpu(),d[key][k]) for k,v in model.state_dict().items()):raise RuntimeError('Warmstart weight mismatch')
        self.current_learning_iteration=0;self.env.unwrapped.common_step_counter=0
        if self.alg.optimizer.state:raise RuntimeError('Warmstart optimizer must be fresh')
        self.provenance['source_iteration']=int(d['iter']);self.provenance['fresh_optimizer']=True
        print('T5_SOFT_BALANCED_WARMSTART',self.provenance,flush=True);return result

    def save(self,path,infos=None):
        if len(self.window)>=25:
            v={k:sum(x[k] for x in self.window)/len(self.window) for k in self.window[0]}
            self.guard_state=guard_update(self.guard_state,self.current_learning_iteration,v)
        elif self.current_learning_iteration>=1000:
            self.guard_state={'halt':True,'reasons':['missing_tracking_metrics']}
        super().save(path,{**(infos or {}),MARKER:self.provenance,'tracking_guard':self.guard_state})
        print('T5_SOFT_TRACKING_GUARD',json.dumps(self.guard_state),flush=True)
        if self.guard_state.get('halt'):
            # Keep guard files per run: a failed smoke experiment must not block production.
            Path(path).parent.joinpath('tracking_guard_halt.json').write_text(json.dumps(self.guard_state,indent=2))
            raise RuntimeError('Tracking regression: checkpoint saved, inspect before resuming')
