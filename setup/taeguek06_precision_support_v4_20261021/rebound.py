"""Anticipate kick-support rebounds; preserve the mature V2 objective."""
import torch
from mjlab.utils.lab_api.math import quat_apply,quat_apply_inverse
from kick_support import KickSupportStability,correction_ramp
from support import huber


def impact_tail(raw, step, ramp_steps=48000):
    # Same GRF thresholds/weight. Gradually restore the tail lost to tanh,
    # rather than raising all contact penalty coefficients.
    return correction_ramp(step,ramp_steps)*(raw-torch.tanh(raw)).clamp_min(0)


def rebound_cost(vertical_speed,gravity_error_velocity,expected):
    up=huber((vertical_speed-.04).clamp_min(0)/.20)*expected
    down=huber((-vertical_speed-.30).clamp_min(0)/.40)*expected
    wobble=huber((gravity_error_velocity.norm(dim=-1)-.20).clamp_min(0)/1.)*expected
    return up,down,wobble


class KickReboundSupport(KickSupportStability):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.rebound_frame=torch.full_like(self.cmd.frame_ids,-1)
        self.previous_gravity_error=torch.zeros(env.num_envs,2,3,device=env.device)
        self.previous_expected=torch.zeros(env.num_envs,2,dtype=torch.bool,device=env.device)
        self.rebound_cache=torch.zeros(env.num_envs,device=env.device)
        self.rebound_step=-1

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'rebound_frame'):
            ids=slice(None) if env_ids is None else env_ids
            self.rebound_frame[ids]=-1;self.previous_gravity_error[ids]=0
            self.previous_expected[ids]=False;self.rebound_cache[ids]=0
            self.rebound_step=-1

    def __call__(self,env,new_ramp_steps=48000):
        if self.rebound_step==env.common_step_counter:return self.rebound_cache
        base=super().__call__(env,new_ramp_steps=1)
        c=self.cmd;f=c.frame_ids;q=c.robot_body_quat_w[:,self.feet]
        expected=self.kick_support[f]
        continuous=(f==self.rebound_frame+1)&(env.episode_length_buf>1)
        count=self.offsets.shape[1]
        rotated=quat_apply(q[:,:,None,:].expand(-1,-1,count,-1),self.offsets[None].expand(env.num_envs,-1,-1,-1))
        surface=rotated.clone();surface[...,2]-=self.radii[None]
        com=quat_apply(q,self.com[None].expand(env.num_envs,-1,-1))
        velocity=c.robot_body_lin_vel_w[:,self.feet,None,:]+torch.cross(
            c.robot_body_ang_vel_w[:,self.feet,None,:].expand_as(surface),surface-com[:,:,None,:],dim=-1)
        # Follow the lowest material point, not the ankle/heel when pivoting.
        low=surface[...,2].argmin(-1,keepdim=True)
        vertical=velocity[...,2].gather(-1,low).squeeze(-1)
        gravity=quat_apply_inverse(q,self.up.expand(env.num_envs,2,3))
        error=gravity-self.target_gravity[f]
        smooth=continuous[:,None]&expected&self.previous_expected
        rate=torch.where(smooth[...,None],(error-self.previous_gravity_error)/env.step_dt,0.)
        up,down,wobble=rebound_cost(vertical,rate,expected)
        valid=(self.age>.15).float();ramp=correction_ramp(env.common_step_counter,new_ramp_steps)
        extra=(.10*up+.05*down+.05*wobble).sum(-1)*valid*ramp
        self.rebound_cache=base+extra
        for name,value in [('rise',up),('descent',down),('wobble',wobble)]:
            c.metrics['t6_rebound_'+name]=value.sum(-1)*valid
        c.metrics['t6_rebound_extra']=extra
        c.metrics['t6_rebound_ramp']=torch.full_like(extra,ramp)
        self.rebound_frame.copy_(f);self.previous_gravity_error.copy_(error)
        self.previous_expected.copy_(expected);self.rebound_step=env.common_step_counter
        return self.rebound_cache
