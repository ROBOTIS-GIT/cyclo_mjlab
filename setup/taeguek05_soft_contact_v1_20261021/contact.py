"""Substep GRF costs and reference-derived post-landing settling, without step pinning."""
import math
import torch
from mjlab.utils.lab_api.math import quat_apply
from source.tasks.mimic.mdp.taeguek_support import TaegeukFootSupportQuality
from source.tasks.mimic.mdp.taeguek_support_v2 import sole_bottom_height
from source.tasks.mimic.mdp.performance_contacts import foot_load_history
from source.tasks.mimic.mdp.support_contacts import SupportTracker


def robust(x):
    return torch.sqrt(1+x.square())-1


def compress(raw):
    # Smooth compression, with no hard saturation of large impact errors.
    return 2*torch.log1p(raw.clamp_min(0)/2)


def landing_masks(bottom, grounded, lowpoint_speed, fps):
    """Offline labels only. No hardcoded chapter timestamps or actual-state gate.

    A touchdown must follow genuine reference clearance; low fast translation
    is not a landing. Normal next liftoffs clear the settling objective.
    """
    device=bottom.device
    height=(bottom-bottom.amin(0)).cpu();ground=grounded.cpu();speed=lowpoint_speed.cpu()
    approach=torch.zeros_like(ground);settle=torch.zeros_like(ground)
    before=math.ceil(.20*fps);after=math.ceil(.45*fps);lookback=math.ceil(.4*fps)
    for foot in range(2):
        for i in range(1,len(height)):
            if bool(ground[i,foot]) and not bool(ground[i-1,foot]):
                if not bool((height[max(0,i-lookback):i,foot]>.04).any()):continue
                approach[max(0,i-before):i+1,foot]=True
                end=min(len(height),i+after+1)
                for j in range(i,end):
                    if not bool(ground[j,foot]):break
                    # Allow a pivot; exclude fast low relocation.
                    if speed[j,foot]<.6:settle[j,foot]=True
    return approach.to(device),settle.to(device)


def impact_costs(loads,weight_n,dt):
    current=loads[:,:,:-1]  # the previous overlapping history sample is not a new peak
    perfoot=current.amax(-1)/weight_n
    total=current.sum(1).amax(-1)/weight_n
    peak=robust((perfoot-2.2).clamp_min(0)).sum(-1)
    peak+=robust((total-3.).clamp_min(0)/1.5)
    rise=((loads[:,:,:-1]-loads[:,:,1:])/(weight_n*dt)).clamp_min(0)
    rate=robust((rise-80.).clamp_min(0)/120.).amax(-1).sum(-1)
    return peak,rate


def approach_cost(down,bottom,reference_landing):
    # Early ground approach is covered even if reference foot is still >10cm.
    # Airborne kicks far from the ground and reference departures stay free.
    gate=reference_landing&(bottom<.06)
    return (robust((down-.6).clamp_min(0)/.6)*gate).sum(-1)


def body_support_deficit(loads,weight_n,reference_support):
    # Distinguish a weight transfer from losing all support; intended flight is free.
    ratio=loads.sum(1)/weight_n
    return (((.2-ratio).clamp_min(0)/.2).square().mean(-1)
            *reference_support.any(-1))


class SoftLanding(TaegeukFootSupportQuality):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.reference_approach,self.reference_settle=landing_masks(
            self.reference_bottom,self.original_reference_support,
            self.reference_lowpoint_speed,float(self.cmd.reference.fps))
        # Low translating references may never reach 4 cm clearance. They still
        # need a gentle downward ground approach, without lateral foot pinning.
        self.reference_approach |= self.original_reference_support
        self.settle_tracker=SupportTracker(env.num_envs,env.device,env.cfg.sim.mujoco.timestep)
        self.weight_n=float(env.sim.mj_model.body_mass.sum())*float(abs(env.sim.mj_model.opt.gravity[2]))

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'settle_tracker'):self.settle_tracker.reset(env_ids)

    def __call__(self,env,peak_weight=.40,loading_rate_weight=.04,approach_weight=.12,
                 settling_missing_weight=.20,settling_liftoff_weight=.01,
                 settling_height_weight=.05,body_support_weight=.20,ramp_steps=120000):
        if self.last_step==env.common_step_counter:return self.cached
        frame=self.cmd.frame_ids
        discontinuity=(frame!=self.last_frame+1)|(env.episode_length_buf<=1)
        self.reset(discontinuity.nonzero(as_tuple=False).flatten())
        self.age+=env.step_dt;valid=(self.age>.15).float()
        loads=foot_load_history(env)[:,:,:env.cfg.decimation+1]
        peak,rate=impact_costs(loads,self.weight_n,env.cfg.sim.mujoco.timestep)
        q=self.cmd.robot_body_quat_w[:,self.feet];p=self.cmd.robot_body_pos_w[:,self.feet]
        count=self.offsets.shape[1]
        rotated=quat_apply(q[:,:,None,:].expand(-1,-1,count,-1),self.offsets[None].expand(env.num_envs,-1,-1,-1))
        surface=rotated.clone();surface[...,2]-=self.radii[None]
        com=quat_apply(q,self.com[None].expand(env.num_envs,-1,-1))
        velocity=self.cmd.robot_body_lin_vel_w[:,self.feet,None,:]+torch.cross(
            self.cmd.robot_body_ang_vel_w[:,self.feet,None,:].expand_as(surface),surface-com[:,:,None,:],dim=-1)
        low=surface[...,2]<=surface[...,2].amin(-1,keepdim=True)+.005
        down=torch.where(low,-velocity[...,2],-torch.inf).amax(-1).clamp_min(0)
        bottom=sole_bottom_height(p,q,self.offsets,self.radii)-env.scene.env_origins[:,2,None]
        approach=approach_cost(down,bottom,self.reference_approach[frame])
        expected=self.reference_settle[frame]
        liftoffs,missing,_,_=self.settle_tracker.update(loads[:,:,:env.cfg.decimation].flip(-1),expected)
        # A low unloaded foot still incurs missing cost: hovering is not a loophole.
        height=(robust((bottom-self.reference_bottom[frame].clamp_min(0)-.003).clamp_min(0)/.01)*expected).sum(-1)
        unsupported=body_support_deficit(loads[:,:,:env.cfg.decimation],self.weight_n,self.reference_support[frame])
        raw=(peak_weight*peak+loading_rate_weight*rate+approach_weight*approach
             +settling_missing_weight*missing.sum(-1)
             +settling_liftoff_weight*liftoffs.sum(-1)/env.step_dt
             +settling_height_weight*height+body_support_weight*unsupported)
        ramp=min(1.,.1+.9*env.common_step_counter/max(1,ramp_steps))
        self.cached=ramp*compress(raw)*valid
        metrics={'peak_cost':peak,'rate_cost':rate,'approach_cost':approach,
                 'settle_missing':missing.sum(-1),'settle_liftoffs':liftoffs.sum(-1),
                 'settle_height_cost':height,'body_support_deficit':unsupported,'peak_total_force_n':loads[:,:,:-1].sum(1).amax(-1),
                 'peak_foot_force_n':loads[:,:,:-1].amax(dim=(1,2)),'cost':self.cached}
        for name,value in metrics.items():self.cmd.metrics['t5_soft_'+name]=value*valid
        self.cmd.metrics['t5_soft_ramp']=torch.full_like(valid,ramp)
        self.last_frame.copy_(frame);self.last_step=env.common_step_counter
        return self.cached
