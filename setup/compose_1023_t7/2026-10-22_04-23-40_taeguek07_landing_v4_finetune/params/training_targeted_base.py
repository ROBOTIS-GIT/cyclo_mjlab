"""Isolated training refinements for Taegeuk 3, 4, 7 and 8.

No observation/action/CSV changes. These tasks do not replace existing tasks.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import numpy as np
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.lab_api.math import quat_apply, quat_mul, quat_from_euler_xyz
from source.tasks.mimic.mdp.commands import ReferenceTrajectoryCommand, ReferenceTrajectoryCommandCfg
from source.tasks.mimic.mdp.taeguek_support import TaegeukFootSupportQuality
from source.tasks.mimic.mdp.taeguek_support_v2 import sole_bottom_height
from source.tasks.mimic.config.k1_rev1.taeguek_support_env_cfg import k1_rev1_taeguek_support_env_cfg
from source.tasks.mimic.config.k1_rev1.agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg

HERE=Path(__file__).resolve().parent
CHAPTERS=(3,4,7,8)
FOCUS={3:((7.0,7.5),(21.7,22.2)),4:((3.8,4.2),(11.9,12.3),(13.3,13.8)),
       7:((4.4,4.9),(5.7,6.1),(31.2,31.7)),8:((6.1,6.5),(25.5,26.0),(36.8,37.3))}

def wrap(a):
    return torch.atan2(torch.sin(a),torch.cos(a))

def yaw(q):
    w,x,y,z=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))

def huber(x):
    """Smooth linear tail: severe failures remain distinguishable, no hard cap."""
    return torch.sqrt(1+x.square())-1

def turn_components(actual_delta,reference_delta,progress,dt):
    actual_rate=actual_delta/dt;reference_rate=reference_delta/dt
    turning=(reference_rate.abs()>.5).float()
    reverse=(-actual_rate*reference_rate.sign()-.25).clamp_min(0)*turning
    return (huber(progress/.5),huber((actual_rate-reference_rate)/2.)*turning,
            huber(reverse/2.))

class DirectionalTurnTracking:
    def __init__(self,cfg,env):
        self.cmd=env.command_manager.get_term('reference_trajectory')
        self.prev_yaw=torch.zeros(env.num_envs,device=env.device)
        self.prev_frame=torch.full((env.num_envs,),-1,device=env.device,dtype=torch.long)
        self.progress=torch.zeros_like(self.prev_yaw)
        self.age=torch.zeros_like(self.prev_yaw);self.cached=torch.zeros_like(self.prev_yaw)
        self.reference_yaw=yaw(self.cmd.reference.body_orientation_w[:,self.cmd.reference_anchor_body_id])
        self.turn_id=torch.zeros(len(self.reference_yaw),device=env.device,dtype=torch.long)
        seconds=torch.arange(len(self.reference_yaw),device=env.device)/float(self.cmd.reference.fps)
        for i,(start,end) in enumerate(((7.4,8.6),(22.2,23.4)),1):
            self.turn_id[(seconds>=start)&(seconds<=end)]=i
        self.last_step=-1

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.prev_frame[ids]=-1;self.progress[ids]=0;self.age[ids]=0;self.cached[ids]=0

    def __call__(self,env,progress_weight=1.,rate_weight=.5,reverse_weight=.5,
                 heading_weight=.5,ramp_steps=12000):
        if self.last_step==env.common_step_counter:return self.cached
        frame=self.cmd.frame_ids;actual=yaw(self.cmd.robot_anchor_quat_w)
        continuous=(frame==self.prev_frame+1)&(self.prev_frame>=0)&(env.episode_length_buf>1)
        active=self.turn_id[frame]>0
        same_turn=active&(self.turn_id[frame]==self.turn_id[self.prev_frame.clamp_min(0)])
        ad=wrap(actual-self.prev_yaw)
        rd=wrap(self.reference_yaw[frame]-self.reference_yaw[self.prev_frame.clamp_min(0)])
        # Do NOT wrap accumulated error: an extra 360-degree turn is not correct.
        self.progress=torch.where(continuous&same_turn,self.progress+ad-rd,
                                  wrap(actual-self.reference_yaw[frame]))*active
        self.age=torch.where(continuous,self.age+env.step_dt,env.step_dt)
        pc,rc,reverse=turn_components(ad,rd,self.progress,env.step_dt)
        heading=1-torch.cos(wrap(actual-self.reference_yaw[frame]))
        valid=(self.age>.15).float()
        ramp=min(1.,.2+.8*env.common_step_counter/max(1,ramp_steps))
        # Each suspect turn is independent: no incentive to unwind a previous
        # extra revolution later in the choreography.
        self.cached=ramp*(progress_weight*pc+rate_weight*rc+reverse_weight*reverse+heading_weight*heading)*valid*active
        for name,value in [('progress_rad',self.progress.abs()),('rate_cost',rc),('reverse_cost',reverse),('heading_cost',heading)]:
            self.cmd.metrics['target_turn_'+name]=value*valid
        self.prev_yaw.copy_(actual);self.prev_frame.copy_(frame);self.last_step=env.common_step_counter
        return self.cached

def support_geometry_cost(bottom,target,vertical_speed,target_vertical_speed,support,
                          pelvis,pelvis_target,air_mask):
    height=huber((bottom-target.clamp_min(0)-.003).clamp_min(0)/.008)*support
    # Penalize upward departure while support is expected. Reference motion is allowed.
    rising=huber((vertical_speed-target_vertical_speed-.05).clamp_min(0)/.25)*support
    world=bottom-target;relative=world-(pelvis-pelvis_target)[:,None]
    flight=.5*(huber(world/.08)+huber(relative/.08))*air_mask
    return height,rising,flight

class SupportRefinement(TaegeukFootSupportQuality):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.pelvis=self.cmd.cfg.body_names.index('pelvis')
        # Includes both feet in intended jumps; support geometry never penalizes them.
        self.air_mask=(self.reference_bottom-self.reference_bottom.amin(0)>.25)&~self.reference_support
        n=len(self.reference_bottom);fps=float(self.cmd.reference.fps)
        self.target_vz=torch.empty_like(self.reference_bottom)
        self.target_vz[1:-1]=(self.reference_bottom[2:]-self.reference_bottom[:-2])*fps*.5
        self.target_vz[0]=(self.reference_bottom[1]-self.reference_bottom[0])*fps
        self.target_vz[-1]=(self.reference_bottom[-1]-self.reference_bottom[-2])*fps
        self.prev_bottom=torch.zeros(env.num_envs,2,device=env.device)
        self.geom_frame=torch.full((env.num_envs,),-1,device=env.device,dtype=torch.long)
        self.geom_cached=torch.zeros(env.num_envs,device=env.device);self.geom_step=-1

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'geom_frame'):
            ids=slice(None) if env_ids is None else env_ids
            self.geom_frame[ids]=-1;self.geom_cached[ids]=0

    def __call__(self,env,height_weight=1.2,rising_weight=.35,flight_weight=.6,ramp_steps=12000):
        if self.geom_step==env.common_step_counter:return self.geom_cached
        ramp=min(1.,.2+.8*env.common_step_counter/max(1,ramp_steps))
        base=super().__call__(env,support_weight=.45+.45*ramp,long_unload_weight=.15+.15*ramp,
            liftoff_weight=.02+.02*ramp,slip_weight=.15,rotation_weight=.05,
            single_support_multiplier=1.5,ramp_steps=1)
        frame=self.cmd.frame_ids;support=self.reference_support[frame]
        bottom=sole_bottom_height(self.cmd.robot_body_pos_w[:,self.feet],
            self.cmd.robot_body_quat_w[:,self.feet],self.offsets,self.radii)-env.scene.env_origins[:,2,None]
        continuous=(frame==self.geom_frame+1)&(self.geom_frame>=0)
        vz=torch.where(continuous[:,None],(bottom-self.prev_bottom)/env.step_dt,self.target_vz[frame])
        pelvis=self.cmd.robot_body_pos_w[:,self.pelvis,2]-env.scene.env_origins[:,2]
        ptarget=self.cmd.reference.body_position_w[frame,self.pelvis,2]
        height,rising,flight=support_geometry_cost(bottom,self.reference_bottom[frame],vz,self.target_vz[frame],
            support,pelvis,ptarget,self.air_mask[frame])
        valid=(self.age>.15).float();single=torch.where(support.sum(-1)==1,1.5,1.)
        extra=(height_weight*height.sum(-1)+rising_weight*rising.sum(-1))*single+flight_weight*flight.sum(-1)
        self.geom_cached=base+ramp*extra*valid
        for name,value in [('height',height),('rising',rising),('flight_height',flight)]:
            self.cmd.metrics['target_support_'+name]=value.sum(-1)*valid
        self.prev_bottom.copy_(bottom);self.geom_frame.copy_(frame);self.geom_step=env.common_step_counter
        return self.geom_cached

@dataclass(kw_only=True)
class TargetedCommandCfg(ReferenceTrajectoryCommandCfg):
    focus_windows: tuple=()
    focus_env_fraction: float=.2
    entry_bank_file: str=''
    entry_env_fraction: float=0.

    def build(self,env):
        if not 0<=self.entry_env_fraction<=self.first_frame_env_fraction<=1:raise ValueError('Invalid entry fractions')
        if not 0<=self.focus_env_fraction<=1-self.first_frame_env_fraction:raise ValueError('Invalid focus fraction')
        return TargetedCommand(self,env)

class TargetedCommand(ReferenceTrajectoryCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.entry_bank=None
        if cfg.entry_bank_file:
            with np.load(cfg.entry_bank_file,allow_pickle=False) as b:
                names=list(b['joint_names']);order=[names.index(n) for n in self.robot.joint_names]
                self.entry_bank={k:torch.tensor(b[k],device=self.device,dtype=torch.float32)
                    for k in ['root_pose','root_com_velocity','joint_pos','joint_vel']}
                self.entry_bank['joint_pos']=self.entry_bank['joint_pos'][:,order]
                self.entry_bank['joint_vel']=self.entry_bank['joint_vel'][:,order]

    def _sample_start_frames(self,env_ids):
        super()._sample_start_frames(env_ids)
        if self.cfg.start_from_zero or not self.cfg.focus_windows:return
        ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        lo=int(self.num_envs*self.cfg.first_frame_env_fraction)
        hi=lo+int(self.num_envs*self.cfg.focus_env_fraction)
        ids=ids[(ids>=lo)&(ids<hi)]
        if not len(ids):return
        ranges=torch.tensor(self.cfg.focus_windows,device=self.device)
        which=torch.randint(len(ranges),(len(ids),),device=self.device)
        seconds=ranges[which,0]+torch.rand(len(ids),device=self.device)*(ranges[which,1]-ranges[which,0])
        self.frame_ids[ids]=(seconds*float(self.reference.fps)).long().clamp(0,self.reference.num_frames-1)

    def _resample_command(self,env_ids):
        super()._resample_command(env_ids)
        # First call during base construction may precede bank initialization.
        if getattr(self,'entry_bank',None) is None or self.cfg.start_from_zero:return
        ids=torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        ids=ids[ids<int(self.num_envs*self.cfg.entry_env_fraction)]
        if not len(ids):return
        bank=self.entry_bank;k=torch.randint(len(bank['root_pose']),(len(ids),),device=self.device)
        root=bank['root_pose'][k].clone();vel=bank['root_com_velocity'][k].clone()
        # Match reference initial heading, preserve tilt/height/joints/velocities.
        angle=wrap(yaw(self.body_quat_w[ids,0])-yaw(root[:,3:]))
        zeros=torch.zeros_like(angle);rot=quat_from_euler_xyz(zeros,zeros,angle)
        root[:,3:]=quat_mul(rot,root[:,3:]);root[:,:2]=self.body_pos_w[ids,0,:2]
        root[:,2]+=self._env.scene.env_origins[ids,2]
        vel[:,:3]=quat_apply(rot,vel[:,:3]);vel[:,3:]=quat_apply(rot,vel[:,3:])
        pos=bank['joint_pos'][k].clone();limits=self.robot.data.soft_joint_pos_limits[ids]
        pos=torch.clamp(pos,limits[:,:,0],limits[:,:,1])
        self.robot.write_joint_state_to_sim(pos,bank['joint_vel'][k],env_ids=ids)
        self.robot.write_root_link_pose_to_sim(root,env_ids=ids)
        ipos=self.robot.data.model.body_ipos[:,self.robot.indexing.root_body_id]
        if ipos.ndim==3:ipos=ipos.squeeze(1)
        com_w=quat_apply(root[:,3:],ipos[ids])
        vel[:,:3]-=torch.cross(vel[:,3:],com_w,dim=-1)
        self.robot.write_root_link_velocity_to_sim(vel,env_ids=ids);self.robot.clear_state(env_ids=ids)

def env_cfg(chapter,play=False):
    if chapter not in CHAPTERS:raise ValueError('Only chapters 3, 4, 7, 8')
    cfg=k1_rev1_taeguek_support_env_cfg(chapter,play=play)
    old=cfg.commands['reference_trajectory']
    cmd=TargetedCommandCfg(**vars(old),focus_windows=FOCUS[chapter],
        entry_bank_file=str(HERE/f'entry_t{chapter}.npz') if chapter!=3 else '',
        entry_env_fraction=.15 if chapter!=3 and not play else 0.)
    cmd.first_frame_env_fraction=.3 if not play else 0.
    cmd.start_sampling_uniform_mix=.2
    cfg.commands['reference_trajectory']=cmd
    # Full-motion fixed-start episodes must reach both turns / the second jump.
    frames=np.load(cmd.trajectory_file,allow_pickle=False)['joint_pos'].shape[0]
    cfg.episode_length_s=frames/50+2 if not play else int(1e9)
    if chapter==3:
        cfg.rewards['targeted_turn']=RewardTermCfg(func=DirectionalTurnTracking,weight=-1.,params={})
        cfg.rewards['taeguek_support_quality'].params['ramp_steps']=1
    else:
        cfg.rewards['taeguek_support_quality']=RewardTermCfg(func=SupportRefinement,weight=-1.,params={})
    return cfg

def agent_cfg():
    cfg=k1_rev1_mimic_ppo_runner_cfg();cfg.run_name='targeted_3478_refinement'
    cfg.algorithm.learning_rate=1e-4;cfg.algorithm.schedule='fixed';cfg.algorithm.entropy_coef=.001
    cfg.max_iterations=10000;cfg.save_interval=500;cfg.resume=False;cfg.upload_model=False
    return cfg

class TargetedRunner(MjlabOnPolicyRunner):
    def load(self,path,load_cfg=None,strict=True,map_location=None):
        data=torch.load(path,map_location='cpu',weights_only=False)
        marker=(data.get('infos') or {}).get('targeted_3478')
        if marker is not None:
            self.provenance=marker
            return super().load(path,load_cfg,strict,map_location)
        if load_cfg is not None or self.alg.optimizer.state:raise ValueError('Warm start requires fresh optimizer')
        infos=super().load(path,dict(actor=True,critic=True,optimizer=False,iteration=False,rnd=False),strict,map_location)
        self.current_learning_iteration=0;self.env.unwrapped.common_step_counter=0
        for model,key in [(self.alg.actor,'actor_state_dict'),(self.alg.critic,'critic_state_dict')]:
            for name,value in data[key].items():
                if not torch.equal(model.state_dict()[name].cpu(),value):raise RuntimeError('Warm-start weight mismatch')
        if self.alg.optimizer.state:raise RuntimeError('Source optimizer restored unexpectedly')
        self.provenance=dict(source=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            iteration=int(data['iter']),fresh_optimizer=True)
        print('TARGETED_WARMSTART',self.provenance,flush=True)
        return infos

    def save(self,path,infos=None):
        super().save(path,{**(infos or {}),'targeted_3478':getattr(self,'provenance',{})})

def register_tasks():
    from mjlab.tasks.registry import register_mjlab_task
    for c in CHAPTERS:
        register_mjlab_task(task_id=f'Cyclo-Mimic-K1-Rev1-Taeguek{c:02d}-TargetedV3',
            env_cfg=env_cfg(c),play_env_cfg=env_cfg(c,True),rl_cfg=agent_cfg(),runner_cls=TargetedRunner)
