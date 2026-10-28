"""Warm-start LandingV3 final; target one hard landing, preserve its baseline."""
from pathlib import Path
from collections import deque
import hashlib,json,shutil,math
import numpy as np
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import MjlabOnPolicyRunner
import base_v2
from landing import TargetedLanding

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
MARKER='taeguek07_landing_v4';TASK='Cyclo-Mimic-K1-Rev1-Taeguek07-LandingV4'
TOTAL=10000
SOURCE=HERE/'source/model_9999.pt'
SOURCE_SHA='c140a2b2843bd09a52d945fc3d9d600ba6ac7c56bdbdaac22f423627f356fdd8'
MOTION=HERE/'motion/k1_taeguek07_arm_corrected_v1_50hz_converted.npz'

def revision():
    h=hashlib.sha256()
    for p in sorted(HERE.glob('*.py')):
        if p.name.startswith(('test_','validate_','audit_')):continue
        h.update(p.name.encode());h.update(p.read_bytes())
    for folder in ('source','scripts'):
        for p in sorted((ROOT/folder).rglob('*.py')):
            h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    h.update((HERE/'inputs.json').read_bytes())
    return h.hexdigest()

def validate_inputs():
    manifest=json.loads((HERE/'inputs.json').read_text())
    for n,h in manifest['files_sha256'].items():
        if hashlib.sha256((HERE/n).read_bytes()).hexdigest()!=h:raise ValueError('Frozen input changed: '+n)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA:raise ValueError('Wrong source checkpoint')
    with np.load(MOTION,allow_pickle=False) as z:
        a=np.loadtxt(MOTION.with_suffix('.csv'),delimiter=',')
        assert a.shape==(2320,30) and z['joint_pos'].shape==(2320,23) and float(z['fps'][0])==50
        assert np.isfinite(a).all() and np.max(abs(a[:,7:]-z['joint_pos']))<1e-6
        for k in z.files:
            if np.issubdtype(z[k].dtype,np.number):assert np.isfinite(z[k]).all(),k
    return manifest

def env_cfg(play=False):
    cfg=base_v2.env_cfg(play)
    # Keep ALL previously learned arm, support, and global impact objectives mature.
    cfg.rewards['support_micro_motion'].params['ramp_steps']=1
    cfg.rewards['soft_ground_contact'].params['ramp_steps']=1
    cfg.rewards['targeted_right_landing']=RewardTermCfg(func=TargetedLanding,weight=-1.,params={'ramp_steps':48000})
    c=cfg.commands['reference_trajectory'];c.trajectory_file=str(MOTION)
    # 30% whole-motion + 10% existing focus + 20% pre-landing + 40% adaptive.
    # Pre-landing starts stay 0.7–1.2 s before impact.
    c.focus_windows=tuple(c.focus_windows)+((26.70,27.20),)*6
    c.focus_env_fraction=.3
    cfg.scene.num_envs=4096 if not play else 1
    return cfg

def agent_cfg():
    cfg=base_v2.agent_cfg();cfg.run_name=MARKER+'_finetune'
    cfg.algorithm.learning_rate=5e-6;cfg.algorithm.entropy_coef=.0003;cfg.algorithm.schedule='fixed'
    cfg.max_iterations=TOTAL;cfg.save_interval=250;cfg.resume=False;cfg.upload_model=False
    return cfg

def guard_update(state,iteration,values):
    s=dict(state)
    if iteration<=s.get('last_iteration',-1):return s
    keys={'arm_rmse','arm_max','joint_error','length'}
    if set(values)!=keys or not all(math.isfinite(v) for v in values.values()):
        return {**s,'last_iteration':iteration,'halt':True,'reasons':['invalid_metrics']}
    if iteration>=250 and 'baseline' not in s:s['baseline']=dict(values)
    b=s.get('baseline',values)
    limits={'arm_rmse':max(.12,min(.16,1.2*b['arm_rmse']+.01)),
            'arm_max':max(.28,min(.4,1.25*b['arm_max']+.02)),
            'joint_error':max(.8,min(1.,1.25*b['joint_error']+.05)),
            'length':max(1400.,.75*b['length'])}
    reasons=[]
    if iteration>=1000:
        reasons=[k for k in keys-{'length'} if values[k]>limits[k]]
        if values['length']<limits['length']:reasons.append('episode_length')
    bad=s.get('bad_checks',0)+1 if reasons else 0
    s.update(last_iteration=iteration,values=dict(values),limits=limits,reasons=reasons,bad_checks=bad,halt=bad>=2)
    return s

def completed_updates(data,num_envs,smoke=False):
    meta=(data.get('infos') or {}).get(MARKER,{})
    if meta.get('revision')!=revision() or meta.get('source_sha256')!=SOURCE_SHA:raise ValueError('Revision/source mismatch')
    if meta.get('num_envs')!=num_envs or meta.get('smoke')!=smoke or meta.get('total_updates')!=TOTAL:raise ValueError('Environment/smoke/budget mismatch')
    n=int(data['iter'])+1
    if not 0<n<=TOTAL or data['infos']['env_state']['common_step_counter']!=n*24:raise ValueError('Counter mismatch')
    if not data.get('optimizer_state_dict',{}).get('state'):raise ValueError('Missing optimizer state')
    if not all(torch.isfinite(v).all() for k in ('actor_state_dict','critic_state_dict') for v in data[k].values()):raise ValueError('Nonfinite weights')
    if data['infos'].get('tracking_guard',{}).get('halt'):raise ValueError('Tracking guard halted; inspect before resume')
    return n

class ContactRunner(MjlabOnPolicyRunner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.guard_state={};self.window=deque(maxlen=100)
        original=self.logger.log
        def log(*a,**kw):
            vals={}
            for name,key in {'arm_rmse':'t7_v2_arm_rmse','arm_max':'t7_v2_arm_max_error','joint_error':'error_joint_pos'}.items():
                ts=[torch.as_tensor(e['Metrics/reference_trajectory/'+key],device=self.device).float().reshape(-1) for e in self.logger.ep_extras if 'Metrics/reference_trajectory/'+key in e]
                if ts:vals[name]=torch.cat(ts).mean().item()
            if self.logger.lenbuffer:vals['length']=sum(self.logger.lenbuffer)/len(self.logger.lenbuffer)
            if len(vals)==4:self.window.append(vals)
            return original(*a,**kw)
        self.logger.log=log
        self.provenance=dict(revision=revision(),num_envs=self.env.unwrapped.num_envs,smoke='smoke' in self.cfg['run_name'],total_updates=TOTAL,
            source_sha256=SOURCE_SHA,source_iteration=9999,initial_state='warmstart',fresh_optimizer=True,inputs=validate_inputs())
        p=Path(args[2] if len(args)>2 else kwargs['log_dir'])/'params';p.mkdir(exist_ok=True,parents=True)
        for f in (MOTION.with_suffix('.csv'),HERE/'inputs.json'):shutil.copy2(f,p/f.name)
        for f in HERE.glob('*.py'):
            if not f.name.startswith(('test_','validate_','audit_')):shutil.copy2(f,p/('training_'+f.name))
        (p/'reward_manifest.json').write_text(json.dumps(self.provenance,indent=2))

    def load(self,path,load_cfg=None,strict=True,map_location=None):
        if load_cfg is not None:raise ValueError('Managed checkpoint loading only')
        d=torch.load(path,map_location='cpu',weights_only=False)
        if MARKER in (d.get('infos') or {}):
            n=completed_updates(d,self.env.unwrapped.num_envs,self.provenance['smoke'])
            result=super().load(path,None,strict,map_location or self.device)
            self.current_learning_iteration=n;self.provenance=d['infos'][MARKER]
            self.guard_state=d['infos'].get('tracking_guard',{});self.window.extend(d['infos'].get('guard_window',[]))
            print('T7_LANDING_V4_RESUME_COMPLETED',n,flush=True);return result
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=SOURCE_SHA:raise ValueError('Expected evaluated LandingV3 final')
        if d['iter']!=9999 or d['infos']['env_state']['common_step_counter']!=240000:raise ValueError('Source completion mismatch')
        if d['infos'].get('tracking_guard',{}).get('halt'):raise ValueError('Source guard halted')
        result=super().load(path,dict(actor=True,critic=True,optimizer=False,iteration=False,rnd=False),strict,map_location or self.device)
        self.current_learning_iteration=0;self.env.unwrapped.common_step_counter=0
        if self.alg.optimizer.state:raise RuntimeError('Source optimizer leaked')
        for model,key in ((self.alg.actor,'actor_state_dict'),(self.alg.critic,'critic_state_dict')):
            if not all(torch.isfinite(v).all() and torch.equal(v.cpu(),d[key][k]) for k,v in model.state_dict().items()):raise RuntimeError('Warm-start mismatch')
        print('T7_LANDING_V4_WARMSTART',SOURCE_SHA,'fresh optimizer',flush=True)
        return result

    def save(self,path,infos=None):
        path=Path(path);tmp=path.with_suffix('.pt.tmp')
        if not self.provenance['smoke']:
            if len(self.window)>=50:
                values={k:sum(v[k] for v in self.window)/len(self.window) for k in self.window[0]}
                self.guard_state=guard_update(self.guard_state,self.current_learning_iteration,values)
            elif self.current_learning_iteration>=1000:self.guard_state={'halt':True,'reasons':['missing_tracking_metrics']}
        super().save(str(tmp),{**(infos or {}),MARKER:self.provenance,'tracking_guard':self.guard_state,'guard_window':list(self.window)})
        tmp.replace(path)
        print('T7_LANDING_V4_GUARD',json.dumps(self.guard_state),flush=True)
        if self.guard_state.get('halt'):
            (path.parent/'tracking_guard_halt.json').write_text(json.dumps(self.guard_state,indent=2))
            raise RuntimeError('Tracking regressed: checkpoint saved; inspect before resuming')
