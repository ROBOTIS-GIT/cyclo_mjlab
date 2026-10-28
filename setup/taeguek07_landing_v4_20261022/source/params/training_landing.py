"""Local right-foot landing costs for the measured 27.94 s impact.

Phase eligibility depends only on the frozen reference. Costs do not turn off
when physical contact disappears. No changes to the CSV or control interface.
"""
import torch
from mjlab.utils.lab_api.math import quat_apply
from source.tasks.mimic.mdp.taeguek_support import TaegeukFootSupportQuality
from source.tasks.mimic.mdp.taeguek_support_v2 import sole_bottom_height
from source.tasks.mimic.mdp.performance_contacts import foot_load_history

WEIGHTS = dict(braking=.35, early_height=.12, early_load=.08,
               peak=.25, load_rate=.025, late_height=.15, late_unload=.12)

def robust(x):
    return torch.sqrt(1 + x.square()) - 1

def smoothstep(x):
    x=x.clamp(0,1)
    return x.square()*(3-2*x)

def phase_gate(time):
    return smoothstep((time-27.70)/.06)*smoothstep((28.35-time)/.08)

def speed_envelope(height, touchdown_speed=.5, braking_acceleration=3.):
    # A soft target, not a hard speed clamp or hardware safety limit.
    return torch.sqrt(touchdown_speed**2+2*braking_acceleration*height.clamp_min(0))

def landing_costs(time,bottom,speed_down,reference_bottom,reference_vz,
                  loads,weight_n,physics_dt):
    """One selected foot: vectors [N], newest-first physics load history [N,K]."""
    if loads.shape[-1]<2 or physics_dt<=0 or weight_n<=0:
        raise ValueError('Physical force history and positive units required')
    phase=phase_gate(time)
    descending=(reference_vz<-.05)&(time<=28.04)
    early=smoothstep((reference_bottom-.015)/.035)
    capture=(time>=28.04)&(reference_bottom<.012)
    current=loads[:,:-1]
    peak=current.amax(-1)/weight_n
    rise=((loads[:,:-1]-loads[:,1:])/(weight_n*physics_dt)).clamp_min(0)
    terms={
        'braking':robust((speed_down-speed_envelope(bottom)).clamp_min(0)/.75)*descending,
        'early_height':robust((reference_bottom-bottom-.015).clamp_min(0)/.04)*early,
        'early_load':robust((peak-.1).clamp_min(0))*early,
        'peak':robust((peak-2.2).clamp_min(0)),
        'load_rate':robust((rise-80.).clamp_min(0)/120.).amax(-1),
        'late_height':robust((bottom-reference_bottom-.01).clamp_min(0)/.025)*capture,
        'late_unload':(1-current.mean(-1)/(.25*weight_n)).clamp(0,1).square()*capture,
    }
    return {k:v*phase for k,v in terms.items()}

class TargetedLanding(TaegeukFootSupportQuality):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        assert self.cmd.cfg.body_names[self.feet[1]]=='right_ankle_roll_link'
        ref=self.cmd.reference
        self.seconds=torch.arange(ref.num_frames,device=env.device)/float(ref.fps)
        self.reference_vz=torch.gradient(self.reference_bottom,spacing=(1/float(ref.fps),),dim=0)[0]
        self.weight_n=float(env.sim.mj_model.body_mass.sum())*9.81

    def __call__(self,env,ramp_steps=72000):
        if self.last_step==env.common_step_counter:return self.cached
        c=self.cmd;frame=c.frame_ids
        continuous=(frame==self.last_frame+1)&(env.episode_length_buf>1)
        self.age=torch.where(continuous,self.age+env.step_dt,0.)
        valid=(self.age>.15).float()
        pos=c.robot_body_pos_w[:,self.feet];q=c.robot_body_quat_w[:,self.feet]
        count=self.offsets.shape[1]
        rotated=quat_apply(q[:,:,None,:].expand(-1,-1,count,-1),self.offsets[None].expand(env.num_envs,-1,-1,-1))
        surface=rotated.clone();surface[...,2]-=self.radii[None]
        com=quat_apply(q,self.com[None].expand(env.num_envs,-1,-1))
        velocity=c.robot_body_lin_vel_w[:,self.feet,None,:]+torch.cross(c.robot_body_ang_vel_w[:,self.feet,None,:].expand_as(surface),surface-com[:,:,None,:],dim=-1)
        low=surface[...,2]<=surface[...,2].amin(-1,keepdim=True)+.005
        down=torch.where(low,-velocity[...,2],-torch.inf).amax(-1).clamp_min(0)[:,1]
        bottom=(sole_bottom_height(pos,q,self.offsets,self.radii)-env.scene.env_origins[:,2,None])[:,1]
        loads=foot_load_history(env)[:,1,:env.cfg.decimation+1]
        time=self.seconds[frame]
        terms=landing_costs(time,bottom,down,self.reference_bottom[frame,1],self.reference_vz[frame,1],loads,self.weight_n,env.cfg.sim.mujoco.timestep)
        ramp=min(1.,.1+.9*env.common_step_counter/max(1,ramp_steps))
        self.cached=ramp*valid*sum(WEIGHTS[k]*v for k,v in terms.items())
        for k,v in terms.items():c.metrics['t7_landing_'+k]=v*valid
        active=(phase_gate(time)>0).float()*valid
        c.metrics['t7_landing_active']=active
        c.metrics['t7_landing_cost']=self.cached.clone()
        c.metrics['t7_landing_peak_n']=loads[:,:-1].amax(-1)*active
        c.metrics['t7_landing_down_m_s']=down*active
        c.metrics['t7_landing_ramp']=torch.full_like(valid,ramp)
        self.last_frame.copy_(frame);self.last_step=env.common_step_counter
        return self.cached
