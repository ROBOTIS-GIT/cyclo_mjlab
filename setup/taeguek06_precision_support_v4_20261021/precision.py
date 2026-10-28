"""Small-angle support precision and world-path recovery, added to mature V3."""
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from rebound import KickReboundSupport
from kick_support import correction_ramp
from support import huber


def filtered_gravity(gravity,mask):
    """100 ms binomial reference filter; exclude non-support neighbours."""
    n=len(gravity);frames=torch.arange(n,device=gravity.device)
    result=torch.zeros_like(gravity);weight=torch.zeros_like(mask,dtype=gravity.dtype)
    for offset,w in zip(range(-2,3),[1.,4.,6.,4.,1.]):
        ids=(frames+offset).clamp(0,n-1);eligible=mask[ids].to(gravity.dtype)*w
        result+=gravity[ids]*eligible[...,None];weight+=eligible
    result=result/weight.clamp_min(1)[...,None]
    result=result/result.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    return torch.where(mask[...,None],result,gravity)


def path_cost(actual,reference):
    return huber(((actual[...,:2]-reference[...,:2]).norm(dim=-1)-.025).clamp_min(0)/.10)


def path_precision(env,ramp_steps=48000):
    c=env.command_manager.get_term('reference_trajectory');i=c.cfg.body_names.index('pelvis')
    cost=path_cost(c.robot_body_pos_w[:,i],c.body_pos_w[:,i])
    valid=(env.episode_length_buf>8).float()
    cost=cost*valid*correction_ramp(env.common_step_counter,ramp_steps)
    c.metrics['t6_path_precision_cost']=cost
    return cost


def precision_cost(error,velocity,expected):
    # Existing V2 attitude deadband is ~2 degrees. Target remaining sub-degree
    # rocking without forcing yaw or flattening the intended reference tilt.
    pose=huber((error.norm(dim=-1)-.010).clamp_min(0)/.0523)*expected
    rate=huber((velocity.norm(dim=-1)-.15).clamp_min(0)/.60)*expected
    return .08*pose+.06*rate


class PrecisionSupport(KickReboundSupport):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.precision_target=filtered_gravity(self.target_gravity,self.kick_support)
        self.precision_frame=torch.full_like(self.cmd.frame_ids,-1)
        self.precision_previous=torch.zeros(env.num_envs,2,3,device=env.device)
        self.precision_expected=torch.zeros(env.num_envs,2,dtype=torch.bool,device=env.device)
        self.precision_cache=torch.zeros(env.num_envs,device=env.device)
        self.precision_step=-1

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'precision_frame'):
            ids=slice(None) if env_ids is None else env_ids
            self.precision_frame[ids]=-1;self.precision_previous[ids]=0
            self.precision_expected[ids]=False;self.precision_cache[ids]=0
            self.precision_step=-1

    def __call__(self,env,new_ramp_steps=48000):
        if self.precision_step==env.common_step_counter:return self.precision_cache
        base=super().__call__(env,new_ramp_steps=1)
        c=self.cmd;f=c.frame_ids;expected=self.kick_support[f]
        gravity=quat_apply_inverse(c.robot_body_quat_w[:,self.feet],self.up.expand(env.num_envs,2,3))
        error=gravity-self.precision_target[f]
        continuous=(f==self.precision_frame+1)&(env.episode_length_buf>1)
        smooth=continuous[:,None]&expected&self.precision_expected
        velocity=torch.where(smooth[...,None],(error-self.precision_previous)/env.step_dt,0.)
        ramp=correction_ramp(env.common_step_counter,new_ramp_steps)
        extra=precision_cost(error,velocity,expected).sum(-1)*(self.age>.15).float()*ramp
        self.precision_cache=base+extra
        c.metrics['t6_precision_extra']=extra
        c.metrics['t6_precision_ramp']=torch.full_like(extra,ramp)
        self.precision_frame.copy_(f);self.precision_previous.copy_(error)
        self.precision_expected.copy_(expected);self.precision_step=env.common_step_counter
        return self.precision_cache
