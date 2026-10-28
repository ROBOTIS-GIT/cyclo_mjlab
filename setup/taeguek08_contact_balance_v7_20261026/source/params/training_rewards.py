"""Contact-phase-aware stability and soft landings for corrected Taegeuk 8."""
from pathlib import Path
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat
from source.tasks.mimic.mdp.taeguek_support import TaegeukFootSupportQuality
from source.tasks.mimic.mdp.performance_contacts import foot_load_history
from costs import phase_drift, geometric_costs, impact_costs, approach_cost, compress, robust, hold_costs, swing_costs

HERE = Path(__file__).resolve().parent


class ContactGeometry(TaegeukFootSupportQuality):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        with np.load(HERE/'motion/contact_phase_50hz.npz', allow_pickle=False) as z:
            assert z['phase'].shape == (self.cmd.reference.num_frames, 2)
            assert abs(float(z['fps'])-float(self.cmd.reference.fps)) < 1e-6
            self.phase = torch.tensor(z['phase'], device=env.device)
            self.reference_support = torch.tensor(z['expected_support'], device=env.device)
            target = torch.tensor(z['sole_bottom_target'], device=env.device)
            if (target-self.reference_bottom).abs().max().item() > 1e-4:
                raise ValueError('Contact labels and runtime foot geometry do not match')
        self.point = torch.tensor([.0325, 0., -.065], device=env.device)
        ref = self.cmd.reference
        self.ref_sole = ref.body_position_w[:,self.feet] + quat_apply(
            ref.body_orientation_w[:,self.feet], self.point.expand(ref.num_frames,2,3))
        self.weight_n = float(env.sim.mj_model.body_mass.sum())*9.81

    def geometry(self, env):
        q = self.cmd.robot_body_quat_w[:,self.feet]
        p = self.cmd.robot_body_pos_w[:,self.feet]
        count = self.offsets.shape[1]
        rotated = quat_apply(q[:,:,None,:].expand(-1,-1,count,-1),
                            self.offsets[None].expand(env.num_envs,-1,-1,-1))
        surface = rotated.clone(); surface[...,2] -= self.radii[None]
        com = quat_apply(q, self.com[None].expand(env.num_envs,-1,-1))
        v = self.cmd.robot_body_lin_vel_w[:,self.feet,None,:] + torch.cross(
            self.cmd.robot_body_ang_vel_w[:,self.feet,None,:].expand_as(surface),
            surface-com[:,:,None,:], dim=-1)
        low = surface[...,2] <= surface[...,2].amin(-1,keepdim=True)+.005
        speed_each = v[...,:2].norm(dim=-1)
        speed_mean = (speed_each*low).sum(-1)/low.sum(-1).clamp_min(1)
        speed_min = torch.where(low, speed_each, torch.inf).amin(-1)
        speed_down = torch.where(low,-v[...,2],-torch.inf).amax(-1).clamp_min(0)
        bottom = p[...,2]+surface[...,2].amin(-1)-env.scene.env_origins[:,2,None]
        sole = p+quat_apply(q,self.point.expand(env.num_envs,2,3))
        return sole, bottom, speed_mean, speed_min, speed_down

    def advance(self, env):
        frame = self.cmd.frame_ids
        continuous = (frame == self.last_frame+1) & (env.episode_length_buf > 1)
        ids = (~continuous).nonzero(as_tuple=False).flatten()
        if len(ids): self.reset(ids)
        self.age += env.step_dt
        return frame, continuous, (self.age > .15).float()


class SupportQuality(ContactGeometry):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.phase_offset = torch.zeros(env.num_envs,2,2,device=env.device)
        self.previous = torch.zeros(env.num_envs,2,dtype=torch.bool,device=env.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'phase_offset'):
            ids = slice(None) if env_ids is None else env_ids
            self.phase_offset[ids] = 0; self.previous[ids] = False

    def __call__(self, env, missing_weight=.7, unload_weight=.15, event_weight=.02,
                 drift_weight=.20, motion_weight=.15, height_weight=.35,
                 rock_weight=.05, ramp_steps=120000):
        if self.last_step == env.common_step_counter: return self.cached
        frame, continuous, valid = self.advance(env)
        phase = self.phase[frame]; expected = self.reference_support[frame]
        loads = foot_load_history(env)[:,:,:env.cfg.decimation].flip(-1)
        liftoffs, missing, long_unload, occupancy = self.tracker.update(loads,expected)
        sole,bottom,mean_speed,min_speed,_ = self.geometry(env)
        planted = expected & (phase==1)
        error = sole[...,:2]-env.scene.env_origins[:,None,:2]-self.ref_sole[frame,:,:2]
        # Reset at reference phase entry, NEVER at actual contact loss.
        drift,self.phase_offset = phase_drift(error,self.phase_offset,planted,self.previous,continuous)
        speed = torch.where(phase==2,min_speed,mean_speed)
        dc,mc,hc,rc = geometric_costs(drift,speed,self.ref_speed[frame],bottom,
            self.reference_bottom[frame],self.cmd.robot_body_ang_vel_w[:,self.feet]-
            self.cmd.body_ang_vel_w[:,self.feet],phase,expected)
        raw = (missing_weight*missing+unload_weight*long_unload+
               event_weight*liftoffs/env.step_dt+drift_weight*dc+motion_weight*mc+
               height_weight*hc+rock_weight*rc).sum(-1)*valid
        ramp = min(1.,.2+.8*env.common_step_counter/max(1,ramp_steps))
        self.cached = ramp*compress(raw)
        for name,value in [('missing',missing.sum(-1)),('unload',long_unload.sum(-1)),
            ('liftoffs',liftoffs.sum(-1)),('drift_cost',dc.sum(-1)),('motion_cost',mc.sum(-1)),
            ('height_cost',hc.sum(-1)),('rock_cost',rc.sum(-1)),
            ('drift_m',(drift.norm(dim=-1)*planted).amax(-1)),
            ('support_speed_m_s',(speed*expected).amax(-1)),('cost',self.cached)]:
            self.cmd.metrics['t8_support_'+name] = value*valid
        self.cmd.metrics['t8_reward_ramp'] = torch.full_like(valid,ramp)
        self.previous.copy_(planted);self.last_frame.copy_(frame);self.last_step=env.common_step_counter
        return self.cached


class SoftLanding(ContactGeometry):
    def __call__(self, env, peak_weight=.4, rise_weight=.04, excess_weight=.06,
                 approach_weight=.12, ramp_steps=120000):
        if self.last_step == env.common_step_counter: return self.cached
        frame,continuous,valid = self.advance(env)
        loads = foot_load_history(env)[:,:,:env.cfg.decimation+1]
        peak,rise,excess = impact_costs(loads,self.weight_n,env.cfg.sim.mujoco.timestep)
        _,bottom,_,_,down = self.geometry(env)
        approach = approach_cost(down,bottom,self.reference_bottom[frame])
        ramp = min(1.,.2+.8*env.common_step_counter/max(1,ramp_steps))
        self.cached = ramp*compress((peak_weight*peak+rise_weight*rise+
            excess_weight*excess+approach_weight*approach)*valid)
        for name,value in [('peak_cost',peak),('rise_cost',rise),('excess_cost',excess),
                           ('approach_cost',approach),('cost',self.cached),
                           ('peak_force_n',loads[:,:,:-1].sum(1).amax(-1))]:
            self.cmd.metrics['t8_landing_'+name] = value*valid
        self.last_frame.copy_(frame);self.last_step=env.common_step_counter
        return self.cached


def arm_pose_error(env):
    cmd = env.command_manager.get_term('reference_trajectory')
    if not hasattr(cmd,'_t8_arms'):
        cmd._t8_arms = [i for i,n in enumerate(cmd.robot.joint_names)
                       if any(s in n for s in ('shoulder','elbow','wrist'))]
        if len(cmd._t8_arms)!=10: raise ValueError('Expected ten arm joints')
    error = cmd.robot_joint_pos[:,cmd._t8_arms]-cmd.joint_pos[:,cmd._t8_arms]
    cmd.metrics['t8_arm_rmse_rad'] = error.square().mean(-1).sqrt()
    return robust(error/.35).mean(-1)


class SupportHold(ContactGeometry):
    """Additional support objective; the established landing objective is retained."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.phase_offset=torch.zeros(env.num_envs,2,2,device=env.device)
        self.previous=torch.zeros(env.num_envs,2,dtype=torch.bool,device=env.device)
        self.up=torch.tensor([0.,0.,1.],device=env.device)
        ref=self.cmd.reference
        self.ref_gravity=quat_apply_inverse(ref.body_orientation_w[:,self.feet],
            self.up.expand(ref.num_frames,2,3))
        # A local-gravity derivative ignores yaw about the vertical axis.
        self.ref_tilt_rate=torch.gradient(self.ref_gravity,spacing=(1/float(ref.fps),),dim=0)[0].norm(dim=-1)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'phase_offset'):
            ids=slice(None) if env_ids is None else env_ids
            self.phase_offset[ids]=0;self.previous[ids]=False

    def __call__(self,env,drift_weight=.25,motion_weight=.12,tilt_weight=.30,
                 ramp_steps=24000):
        if self.last_step==env.common_step_counter:return self.cached
        frame,continuous,valid=self.advance(env)
        phase=self.phase[frame];expected=self.reference_support[frame]
        sole,bottom,speed,_,_=self.geometry(env)
        planted=expected&(phase==1)
        error=sole[...,:2]-env.scene.env_origins[:,None,:2]-self.ref_sole[frame,:,:2]
        drift,self.phase_offset=phase_drift(error,self.phase_offset,planted,self.previous,continuous)
        gravity=quat_apply_inverse(self.cmd.robot_body_quat_w[:,self.feet],self.up.expand(env.num_envs,2,3))
        dc,mc,tc,quiet=hold_costs(drift,speed,gravity,self.ref_gravity[frame],
            self.ref_speed[frame],self.ref_tilt_rate[frame],phase,expected,self.reference_bottom[frame])
        ramp=min(1.,env.common_step_counter/max(1,ramp_steps))
        self.cached=ramp*compress((drift_weight*dc+motion_weight*mc+tilt_weight*tc).sum(-1))*valid
        for name,value in [('drift_cost',dc.sum(-1)),('speed_cost',mc.sum(-1)),('tilt_cost',tc.sum(-1)),
                           ('quiet_feet',quiet.sum(-1)),('cost',self.cached)]:
            self.cmd.metrics['t8_hold_'+name]=value*valid
        self.cmd.metrics['t8_hold_ramp']=torch.full_like(valid,ramp)
        self.previous.copy_(planted);self.last_frame.copy_(frame);self.last_step=env.common_step_counter
        return self.cached


class SwingTracking(ContactGeometry):
    def __call__(self,env,xy_weight=.35,height_weight=.80,ramp_steps=24000):
        if self.last_step==env.common_step_counter:return self.cached
        frame,_,valid=self.advance(env)
        sole,bottom,_,_,_=self.geometry(env)
        # Compare shape in each pelvis's yaw frame: no penalty for global drift,
        # but a short/low kick cannot hide behind good average joint tracking.
        actual=quat_apply_inverse(yaw_quat(self.cmd.robot_anchor_quat_w)[:,None,:].expand(-1,2,-1),
            sole-self.cmd.robot_anchor_pos_w[:,None,:])
        ref=quat_apply_inverse(yaw_quat(self.cmd.anchor_quat_w)[:,None,:].expand(-1,2,-1),
            self.ref_sole[frame]-self.cmd.reference.body_position_w[frame,self.cmd.reference_anchor_body_id,None,:])
        xy,h,gate=swing_costs(actual[...,:2],ref[...,:2],bottom,self.reference_bottom[frame],self.phase[frame])
        ramp=min(1.,.1+.9*env.common_step_counter/max(1,ramp_steps))
        self.cached=ramp*compress((xy_weight*xy+height_weight*h).sum(-1))*valid
        self.cmd.metrics['t8_swing_height_error_m']=((bottom-self.reference_bottom[frame]).abs()*gate).amax(-1)*valid
        self.cmd.metrics['t8_swing_xy_error_m']=((actual[...,:2]-ref[...,:2]).norm(dim=-1)*gate).amax(-1)*valid
        self.cmd.metrics['t8_swing_cost']=self.cached.clone()
        self.last_frame.copy_(frame);self.last_step=env.common_step_counter
        return self.cached


def torso_tilt_error(env):
    cmd=env.command_manager.get_term('reference_trajectory')
    if not hasattr(cmd,'_t8_torso_index'):
        cmd._t8_torso_index=cmd.cfg.body_names.index('torso_link')
    i=cmd._t8_torso_index
    up=torch.tensor([0.,0.,1.],device=env.device).expand(env.num_envs,3)
    actual=quat_apply_inverse(cmd.robot_body_quat_w[:,i],up)
    target=quat_apply_inverse(cmd.body_quat_w[:,i],up)
    chord=(actual-target).norm(dim=-1)
    angle=2*torch.asin((chord/2).clamp(0,1))
    cmd.metrics['t8_torso_tilt_rad']=angle
    return robust((chord-.025).clamp_min(0)/.15)
