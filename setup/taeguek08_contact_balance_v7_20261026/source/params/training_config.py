"""T8 contact precision from the screened V5.1 checkpoint at iteration 5000."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from source.tasks.mimic.config.k1_rev1.taeguek_env_cfg import k1_rev1_taeguek_env_cfg
from source.tasks.mimic.config.k1_rev1.agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg
from source.tasks.mimic.mdp.performance_contacts import add_foot_sensors
from rewards import SupportQuality, SoftLanding, SupportHold, SwingTracking, torso_tilt_error
from arm_recovery import ArmRecovery
from collections import deque
from guard import update_guard
from refinement import LandingSupportRefinement
from precision import ContactPrecision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MOTION = HERE/'motion/k1_taeguek08_contact_v2_50hz.npz'
MARKER = 'taeguek08_contact_precision_v6'
TASK = 'Cyclo-Mimic-K1-Rev1-Taeguek08-ContactPrecisionV6'
TOTAL = 20000
SOURCE = HERE/'source/model_5000.pt'
SOURCE_SHA = '1ffd0058aba47b0f2a1e2bd5e56f38ba946099dce04cd146826b21008f97175c'


def revision():
    h = hashlib.sha256()
    for p in sorted(HERE.glob('*.py')):
        if p.name.startswith('test_') or p.name.startswith('validate_'): continue
        h.update(p.name.encode()); h.update(p.read_bytes())
    # Includes imported model, configuration, command and contact code.
    for directory in ('source/tasks/mimic','source/robots','source/assets/robots'):
        for p in sorted((ROOT/directory).rglob('*.py')):
            h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    h.update((HERE/'inputs.json').read_bytes())
    h.update(SOURCE_SHA.encode())
    return h.hexdigest()


def validate_inputs():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError('Warm-start checkpoint hash mismatch')
    manifest = json.loads((HERE/'inputs.json').read_text())
    for name,sha in manifest['files_sha256'].items():
        if hashlib.sha256((HERE/'motion'/name).read_bytes()).hexdigest()!=sha:
            raise ValueError(f'Frozen input changed: {name}')
    with np.load(MOTION,allow_pickle=False) as z:
        for k in z.files:
            if np.issubdtype(z[k].dtype,np.number) and not np.isfinite(z[k]).all():
                raise ValueError(f'Non-finite motion field: {k}')
        csv=np.loadtxt(MOTION.with_suffix('.csv'),delimiter=',')
        assert csv.shape==(2519,30) and z['joint_pos'].shape==(2519,23)
        assert float(z['fps'][0])==50 and np.max(abs(csv[:,7:]-z['joint_pos']))<1e-6
        pelvis=list(z['body_names']).index('pelvis')
        assert np.max(abs(csv[:,:3]-z['body_pos_w'][:,pelvis]))<1e-6
        quats=z['body_quat_w'][:,pelvis];target=csv[:,[6,3,4,5]]
        assert np.min(abs(np.sum(quats*target,axis=1)))>1-1e-6
    with np.load(HERE/'motion/contact_phase_50hz.npz',allow_pickle=False) as z:
        assert z['phase'].shape==(2519,2) and np.isin(z['phase'],[0,1,2,3]).all()
        assert np.max(abs(z['time']-np.arange(2519)/50))<1e-8
        assert np.all(~z['expected_support'] | np.isin(z['phase'],[1,2]))
    return manifest


def env_cfg(play=False):
    cfg = add_foot_sensors(k1_rev1_taeguek_env_cfg(8,play=play))
    cmd = cfg.commands['reference_trajectory']
    cmd.trajectory_file = str(MOTION)
    cmd.first_frame_env_fraction = 0.4 if not play else 0.
    cmd.start_sampling_uniform_mix = .2
    cfg.scene.num_envs = 4096 if not play else 1
    cfg.episode_length_s = 52.38 if not play else int(1e9)
    # Do not ramp the already learned contact/impact objectives back down.
    cfg.rewards['t8_support_quality'] = RewardTermCfg(func=SupportQuality,weight=-1.,params={'ramp_steps':1})
    cfg.rewards['t8_soft_landing'] = RewardTermCfg(func=SoftLanding,weight=-1.,params={'ramp_steps':1})
    cfg.rewards['t8_support_hold'] = RewardTermCfg(func=SupportHold,weight=-1.,params={'drift_weight':.12,'motion_weight':.06,'tilt_weight':.20,'ramp_steps':1})
    cfg.rewards['t8_swing_tracking'] = RewardTermCfg(func=SwingTracking,weight=-1.,params={'xy_weight':.15,'height_weight':.35,'ramp_steps':1})
    cfg.rewards['t8_arm_recovery'] = RewardTermCfg(func=ArmRecovery,weight=-1.,params={'joint_weight':1.8,'elbow_weight':.6,'hand_weight':1.2,'orientation_weight':.5,'hold_velocity_weight':.04,'hold_acceleration_weight':.01,'ramp_steps':1})
    cfg.rewards['t8_torso_tilt'] = RewardTermCfg(func=torso_tilt_error,weight=-.85,params={})
    cfg.rewards['t8_landing_support_refinement'] = RewardTermCfg(func=LandingSupportRefinement,weight=-1.,params={'ramp_steps':1})
    cfg.rewards['t8_contact_precision'] = RewardTermCfg(func=ContactPrecision,weight=-1.,params={'ramp_steps':48000})
    cfg.rewards['reference_pelvis_xy'].weight=1.3
    # Keep noise, disturbances, PD and 200Hz physics/50Hz policy.
    return cfg


def agent_cfg():
    cfg = k1_rev1_mimic_ppo_runner_cfg()
    cfg.run_name = MARKER+'_finetune'
    cfg.max_iterations = TOTAL
    cfg.save_interval = 250
    cfg.algorithm.learning_rate = 5e-6
    cfg.algorithm.schedule = 'fixed'
    cfg.algorithm.entropy_coef = .0003
    cfg.resume = False
    cfg.upload_model = False
    return cfg


def completed_updates(data, num_envs, smoke=False):
    meta = data.get('infos',{}).get(MARKER,{})
    if meta.get('revision')!=revision() or meta.get('smoke')!=smoke:
        raise ValueError('Reward revision or smoke/main checkpoint mismatch')
    if meta.get('num_envs')!=num_envs or meta.get('total_updates')!=TOTAL:
        raise ValueError('Environment count or training budget mismatch')
    completed=int(data['iter'])+1
    if not 0<completed<=TOTAL or data['infos']['env_state']['common_step_counter']!=completed*24:
        raise ValueError('Checkpoint update count/environment counter mismatch')
    for key in ('actor_state_dict','critic_state_dict'):
        if not all(torch.isfinite(v).all() for v in data[key].values()):
            raise ValueError('Non-finite checkpoint weights')
    if not data.get('optimizer_state_dict',{}).get('state'):
        raise ValueError('Missing optimizer state')
    if data['infos'].get('tracking_guard',{}).get('halt'):
        raise ValueError('Tracking guard stopped this run; inspect before resuming')
    return completed


class ContactRunner(MjlabOnPolicyRunner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.guard_state = {}
        self.window = deque(maxlen=100)
        original_log = self.logger.log
        def collect(*a,**kw):
            vals={}
            for name,key in {'arm':'t8_arm_rmse_rad','hand':'t8_arm_hand_error_m','tilt':'t8_torso_tilt_rad','joint':'error_joint_pos'}.items():
                ts=[torch.as_tensor(e['Metrics/reference_trajectory/'+key],device=self.device).float().reshape(-1) for e in self.logger.ep_extras if 'Metrics/reference_trajectory/'+key in e]
                if ts: vals[name]=torch.cat(ts).mean().item()
            if self.logger.lenbuffer: vals['length']=sum(self.logger.lenbuffer)/len(self.logger.lenbuffer)
            if len(vals)==5: self.window.append(vals)
            return original_log(*a,**kw)
        self.logger.log=collect
        self.provenance = dict(revision=revision(),num_envs=self.env.unwrapped.num_envs,
            smoke='smoke' in self.cfg['run_name'],total_updates=TOTAL,
            initial_state='warmstart',inputs=validate_inputs(),save_interval=250,
            source_sha256=SOURCE_SHA,source_iteration=5000,fresh_optimizer=True)
        log_dir = Path(args[2] if len(args)>2 else kwargs['log_dir'])
        self.saved_params_dir=log_dir/'params'
        p=log_dir/'params';p.mkdir(exist_ok=True,parents=True)
        for file in (MOTION.with_suffix('.csv'),HERE/'motion/contact_phase_50hz.npz',HERE/'inputs.json'):
            shutil.copy2(file,p/file.name)
        for file in ('config.py','rewards.py','costs.py','launch.py','arm_recovery.py','guard.py','refinement.py','refinement_costs.py','precision.py','precision_costs.py'):
            shutil.copy2(HERE/file,p/('training_'+file))
        (p/'reward_manifest.json').write_text(json.dumps(self.provenance,indent=2))

    def load(self,path,load_cfg=None,strict=True,map_location=None):
        if load_cfg is not None: raise ValueError('Managed checkpoint loading only')
        data=torch.load(path,map_location='cpu',weights_only=False)
        if MARKER not in (data.get('infos') or {}):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=SOURCE_SHA:
                raise ValueError('Warm start requires the evaluated T8 model_5000')
            if int(data['iter'])!=5000 or data['infos']['env_state']['common_step_counter']!=120024:
                raise ValueError('Source completion metadata mismatch')
            result=super().load(path,dict(actor=True,critic=True,optimizer=False,iteration=False,rnd=False),strict,map_location or self.device)
            source_guard=data['infos'].get('tracking_guard',{})
            if source_guard.get('halt') or not source_guard.get('values'):
                raise ValueError('Expected healthy source tracking metrics')
            self.guard_state={'baseline':dict(source_guard.get('baseline',source_guard['values']))}
            self.current_learning_iteration=0
            self.env.unwrapped.common_step_counter=0
            if self.alg.optimizer.state:raise RuntimeError('Source optimizer leaked into finetune')
            for model,key in ((self.alg.actor,'actor_state_dict'),(self.alg.critic,'critic_state_dict')):
                if not all(torch.isfinite(v).all() and torch.equal(v.cpu(),data[key][k]) for k,v in model.state_dict().items()):
                    raise RuntimeError('Warm-start weights invalid or differ from source')
            print('T8_CONTACT_PRECISION_V6_WARMSTART',SOURCE_SHA,'fresh optimizer; completed=0',flush=True)
            return result
        n=completed_updates(data,self.env.unwrapped.num_envs,self.provenance['smoke'])
        result=super().load(path,None,strict,map_location or self.device)
        self.current_learning_iteration=n
        self.provenance=data['infos'][MARKER]
        self.guard_state=data['infos'].get('tracking_guard',{})
        self.window.extend(data['infos'].get('guard_window',[]))
        (self.saved_params_dir/'reward_manifest.json').write_text(json.dumps(self.provenance,indent=2))
        print('T8_CONTACT_PRECISION_V6_RESUMED_COMPLETED_UPDATES',n,flush=True)
        return result

    def save(self,path,infos=None):
        path=Path(path);tmp=path.with_suffix('.pt.tmp')
        if not self.provenance['smoke']:
            if len(self.window)>=50:
                values={k:sum(v[k] for v in self.window)/len(self.window) for k in self.window[0]}
                self.guard_state=update_guard(self.guard_state,self.current_learning_iteration,values)
            elif self.current_learning_iteration>=1000:
                self.guard_state={**self.guard_state,'halt':False,'warning':True,'reasons':['missing_tracking_metrics']}
        live=self.env.unwrapped.episode_length_buf
        health=dict(live_length_max=int(live.max()),live_length_mean=float(live.float().mean()),
                    live_environments_over24=int((live>24).sum()))
        super().save(str(tmp),{**(infos or {}),MARKER:self.provenance,'tracking_guard':self.guard_state,'guard_window':list(self.window),'live_training_health':health})
        print('T8_V6_LIVE_HEALTH',json.dumps(health),flush=True)
        tmp.replace(path)
        print('T8_CONTACT_PRECISION_V6_GUARD',json.dumps(self.guard_state),flush=True)
        if self.guard_state.get('warning'):
            path.parent.joinpath('tracking_guard_warning.json').write_text(json.dumps(self.guard_state,indent=2))
        if self.guard_state.get('halt'):
            path.parent.joinpath('training_error.json').write_text(json.dumps(self.guard_state,indent=2))
            raise RuntimeError('Invalid non-finite training metrics; checkpoint saved')
