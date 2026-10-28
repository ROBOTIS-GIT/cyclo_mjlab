"""Taegeuk 6: bounded support plus arm pose recovery, isolated from other tasks."""
from pathlib import Path
from collections import deque
import hashlib,json,math
import numpy as np
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from source.tasks.mimic.config.k1_rev1.taeguek_env_cfg import k1_rev1_taeguek_env_cfg
from source.tasks.mimic.config.k1_rev1.agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg
from source.tasks.mimic.mdp.performance_contacts import add_foot_sensors,foot_load_history
from source.tasks.mimic.mdp.taeguek_support import TaegeukFootSupportQuality
from source.tasks.mimic.mdp.taeguek_support_v2 import sole_bottom_height,kick_height_cost
from support import SupportStability,support_ramp,huber
from arms import ArmRecovery
from rebound import impact_tail
from precision import PrecisionSupport, path_precision

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=HERE/'source/model_6250.pt'
MOTION=HERE/'motion/k1_taeguek06_collision_only_50hz_converted.npz'
MARKER='taeguek06_precision_support_v4'
RUN_NAME=MARKER+'_finetune'
TOTAL_UPDATES=10000
RAMP_STEPS=120000  # 5000 PPO updates x 24 environment steps

def revision():
    h=hashlib.sha256()
    paths=[HERE/n for n in ['refinement.py','support.py','arms.py','kick_support.py','rebound.py','precision.py','inputs.json']]
    paths += [ROOT/'source/tasks/mimic/mdp'/n for n in ['support_contacts.py','taeguek_support.py','taeguek_support_v2.py','performance_contacts.py','contact_transitions.py','commands.py']]
    paths += [ROOT/'source/tasks/mimic/config/k1_rev1'/n for n in ['base_env_cfg.py','taeguek_env_cfg.py','agents/rsl_rl_ppo_cfg.py']]
    paths += [ROOT/'source/tasks/mimic/tracking_env_cfg.py',ROOT/'source/assets/robots/robotis_k1/k1_rev1.py']
    for p in paths:h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()

def validate_inputs():
    for name,h in json.loads((HERE/'inputs.json').read_text()).items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=h:raise ValueError('Input changed: '+name)
    z=np.load(MOTION,allow_pickle=False);a=np.loadtxt(MOTION.with_suffix('.csv'),delimiter=',')
    assert z['joint_pos'].shape==(2124,23) and float(z['fps'][0])==50.
    assert np.isfinite(a).all() and np.max(abs(a[:,7:]-z['joint_pos']))<1e-6
    for k in ['joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']:assert np.isfinite(z[k]).all()


def arm_pose_tracking(env,std=.35):
    c=env.command_manager.get_term('reference_trajectory')
    ids=[i for i,n in enumerate(c.robot.joint_names) if any(x in n for x in ['shoulder','elbow','wrist'])]
    return torch.exp(-((c.robot_joint_pos[:,ids]-c.joint_pos[:,ids])/std).square()).mean(-1)


class KickPreservation(TaegeukFootSupportQuality):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.pelvis=self.cmd.cfg.body_names.index('pelvis')
        self.kick=((self.reference_bottom-self.reference_bottom.amin(0))>.25)&~self.reference_support

    def __call__(self,env,ramp_steps=RAMP_STEPS):
        c=self.cmd;f=c.frame_ids
        b=sole_bottom_height(c.robot_body_pos_w[:,self.feet],c.robot_body_quat_w[:,self.feet],self.offsets,self.radii)-env.scene.env_origins[:,2,None]
        p=c.robot_body_pos_w[:,self.pelvis,2]-env.scene.env_origins[:,2]
        rp=c.reference.body_position_w[f,self.pelvis,2]
        cost=kick_height_cost(b,self.reference_bottom[f],p,rp,self.kick[f]).sum(-1)
        valid=(env.episode_length_buf>8).float()
        c.metrics['t6_kick_height_abs_error_m']=((b-self.reference_bottom[f]).abs()*self.kick[f]).sum(-1)*valid
        return support_ramp(env.common_step_counter,ramp_steps)*cost*valid


def impact_terms(loads,weight_n,dt):
    x=loads[:,:,:-1]
    peak=huber((x.amax(-1)/weight_n-2.2).clamp_min(0)).sum(-1)
    peak+=huber((x.sum(1).amax(-1)/weight_n-3.).clamp_min(0)/1.5)
    rise=((loads[:,:,:-1]-loads[:,:,1:])/(weight_n*dt)).clamp_min(0)
    rate=huber((rise-80.).clamp_min(0)/120.).amax(-1).sum(-1)
    return peak,rate


def soft_contact(env,ramp_steps=RAMP_STEPS):
    c=env.command_manager.get_term('reference_trajectory');loads=foot_load_history(env)
    peak,rate=impact_terms(loads,float(env.sim.mj_model.body_mass.sum())*9.81,env.cfg.sim.mujoco.timestep)
    valid=(env.episode_length_buf>8).float()
    c.metrics['t6_contact_peak_n']=loads[:,:,:-1].sum(1).amax(-1)*valid
    # At most .15 total cost after the RewardTerm weight; auxiliary to tracking.
    raw=peak+.08*rate
    c.metrics['t6_grf_tail_extra']=impact_tail(raw,48000)*valid
    return support_ramp(env.common_step_counter,ramp_steps)*(torch.tanh(raw)+impact_tail(raw,48000))*valid


def env_cfg(play=False):
    cfg=add_foot_sensors(k1_rev1_taeguek_env_cfg(6,play))
    c=cfg.commands['reference_trajectory'];c.trajectory_file=str(MOTION)
    c.first_frame_env_fraction=.3 if not play else 0.;c.start_sampling_uniform_mix=.2
    cfg.episode_length_s=44.48 if not play else int(1e9)
    cfg.scene.num_envs=4096 if not play else 1
    # Exactly one support penalty; do not stack the stronger old SupportV1 term.
    cfg.rewards['t6_support']=RewardTermCfg(func=PrecisionSupport,weight=-1.,params={'new_ramp_steps':48000})
    cfg.rewards['t6_arm_pose']=RewardTermCfg(func=arm_pose_tracking,weight=.4,params={})
    cfg.rewards['t6_arm_recovery']=RewardTermCfg(func=ArmRecovery,weight=-1.,params={
        'joint_weight':.35,'elbow_weight':.15,'hand_weight':.30,'orientation_weight':.10,
        'hold_velocity_weight':.08,'hold_acceleration_weight':.01,'ramp_steps':1})
    cfg.rewards['t6_kick_height']=RewardTermCfg(func=KickPreservation,weight=-.25,params={'ramp_steps':1})
    cfg.rewards['t6_soft_contact']=RewardTermCfg(func=soft_contact,weight=-.15,params={'ramp_steps':1})
    cfg.rewards['t6_path_precision']=RewardTermCfg(func=path_precision,weight=-.15,params={'ramp_steps':48000})
    return cfg


def agent_cfg():
    c=k1_rev1_mimic_ppo_runner_cfg();c.algorithm.learning_rate=3e-6
    c.algorithm.schedule='fixed';c.algorithm.entropy_coef=.0005
    c.max_iterations=TOTAL_UPDATES;c.save_interval=250;c.run_name=RUN_NAME
    c.resume=False;c.upload_model=False
    return c


def guard_update(state,iteration,values):
    s=dict(state)
    if iteration<=s.get('last_iteration',-1):return s
    if not all(math.isfinite(x) for x in values.values()):
        return {**s,'last_iteration':iteration,'halt':True,'reasons':['nonfinite_metrics']}
    # Calibrate to this 6th-chapter policy, not absolute thresholds from chapter 7.
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
    if 'sampling' not in d['infos'] or not torch.isfinite(d['infos']['sampling']).all():raise ValueError('Missing/nonfinite sampler state')
    if not d.get('optimizer_state_dict',{}).get('state'):raise ValueError('Missing optimizer state')
    if not all(torch.isfinite(v).all() for k in ['actor_state_dict','critic_state_dict'] for v in d[k].values()):raise ValueError('Nonfinite checkpoint')
    if d['infos'].get('tracking_guard',{}).get('halt'):raise ValueError('Tracking guard halted; inspect before resume')
    return n


class BalancedRunner(MjlabOnPolicyRunner):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.guard_state={};self.window=deque(maxlen=50)
        self.provenance={'revision':revision(),'num_envs':self.env.unwrapped.num_envs,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
        original=self.logger.log
        def log(*args,**kwargs):
            vals={}
            for name,key in {'arm':'t6_arm_joint_error_rad','hand':'t6_arm_hand_error_m','joint':'error_joint_pos'}.items():
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
            cmd=self.env.unwrapped.command_manager.get_term('reference_trajectory')
            cmd.failure_score_by_bin.copy_(d['infos']['sampling'].to(self.device))
            print('T6_PRECISION_SUPPORT_RESUME',n,flush=True);return result
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=self.provenance['source_sha256']:raise ValueError('Wrong warmstart source')
        if not all(torch.isfinite(v).all() for k in ['actor_state_dict','critic_state_dict'] for v in d[k].values()):raise ValueError('Nonfinite source')
        result=super().load(path,dict(actor=True,critic=True,optimizer=False,iteration=False,rnd=False),strict,map_location or self.device)
        for model,key in [(self.alg.actor,'actor_state_dict'),(self.alg.critic,'critic_state_dict')]:
            if not all(torch.equal(v.cpu(),d[key][k]) for k,v in model.state_dict().items()):raise RuntimeError('Warmstart weight mismatch')
        self.current_learning_iteration=0;self.env.unwrapped.common_step_counter=0
        if self.alg.optimizer.state:raise RuntimeError('Warmstart optimizer must be fresh')
        self.provenance['source_iteration']=int(d['iter']);self.provenance['fresh_optimizer']=True
        print('T6_PRECISION_SUPPORT_WARMSTART',self.provenance,flush=True);return result

    def save(self,path,infos=None):
        if len(self.window)>=25:
            v={k:sum(x[k] for x in self.window)/len(self.window) for k in self.window[0]}
            self.guard_state=guard_update(self.guard_state,self.current_learning_iteration,v)
        elif self.current_learning_iteration>=1000:
            self.guard_state={'halt':True,'reasons':['missing_tracking_metrics']}
        cmd=self.env.unwrapped.command_manager.get_term('reference_trajectory')
        super().save(path,{**(infos or {}),MARKER:self.provenance,'tracking_guard':self.guard_state,
            'sampling':cmd.failure_score_by_bin.detach().cpu().clone()})
        print('T6_TRACKING_GUARD',json.dumps(self.guard_state),flush=True)
        if self.guard_state.get('halt'):
            # Keep guard files per run: a failed smoke experiment must not block production.
            Path(path).parent.joinpath('tracking_guard_halt.json').write_text(json.dumps(self.guard_state,indent=2))
            raise RuntimeError('Tracking regression: checkpoint saved, inspect before resuming')
