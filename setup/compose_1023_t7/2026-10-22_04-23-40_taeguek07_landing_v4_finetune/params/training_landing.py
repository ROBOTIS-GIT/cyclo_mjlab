"""Reference-aware braking and both-foot impact costs for one T7 landing.

The frozen descending reference supplies both phase and height-matched velocity
limits, so following its fast airborne descent does not incur braking costs.
"""
import torch
from mjlab.utils.lab_api.math import quat_apply
from source.tasks.mimic.mdp.taeguek_support import TaegeukFootSupportQuality
from source.tasks.mimic.mdp.taeguek_support_v2 import sole_bottom_height
from source.tasks.mimic.mdp.performance_contacts import foot_load_history
from landing_v3 import landing_costs as old_costs,WEIGHTS as OLD_WEIGHTS
from landing_v3 import robust,smoothstep,phase_gate

WEIGHTS=dict(braking=.6,early_height=.2,early_load=.12,peak=.25,load_rate=.025,
             late_height=.15,late_unload=.12,total_peak=.25,total_rate=.05,total_excess=.15)

def interpolate_height_speed(height,reference_heights,reference_speeds):
    """Frozen heights strictly ascending; edge values extend outside the table."""
    h=height.clamp(reference_heights[0],reference_heights[-1])
    i=torch.searchsorted(reference_heights,h.contiguous()).clamp(1,len(reference_heights)-1)
    alpha=(h-reference_heights[i-1])/(reference_heights[i]-reference_heights[i-1])
    return reference_speeds[i-1]+alpha*(reference_speeds[i]-reference_speeds[i-1])

def speed_target(reference_vz,height_matched_down):
    # Lower bound permits soft touchdown. No arbitrary envelope opposing the
    # recorded 1.9 m/s descent 19 cm above the floor.
    phase_down=(-reference_vz).clamp_min(0)
    return (torch.minimum(phase_down,height_matched_down.clamp_min(0))+.2).clamp_min(.6)

def landing_costs(time,bottom,speed_down,reference_bottom,reference_vz,
                  loads,weight_n,physics_dt,height_matched_down):
    if loads.ndim!=3 or loads.shape[1]!=2 or loads.shape[2]<2:
        raise ValueError('Expected both feet [N,2,K], newest-first, K>=2')
    terms=old_costs(time,bottom,speed_down,reference_bottom,reference_vz,loads[:,1],weight_n,physics_dt)
    gate=phase_gate(time)
    descending=(reference_vz<-.05)&(time<=28.04)
    terms['braking']=robust((speed_down-speed_target(reference_vz,height_matched_down)).clamp_min(0)/.75)*descending*gate
    total=loads.sum(1);current=total[:,:-1]/weight_n
    rise=((total[:,:-1]-total[:,1:])/(weight_n*physics_dt)).clamp_min(0)
    excess=(current-3.).clamp_min(0)
    terms['total_peak']=robust(excess.amax(-1))*gate
    terms['total_rate']=robust((rise-80.).clamp_min(0)/120.).amax(-1)*gate
    # RewardManager multiplies costs by policy dt. Mean over the current four
    # physical samples then yields exactly sum(excess)*physics_dt; no double dt.
    terms['total_excess']=excess.mean(-1)*gate
    return terms

class TargetedLanding(TaegeukFootSupportQuality):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        assert self.cmd.cfg.body_names[self.feet[1]]=='right_ankle_roll_link'
        ref=self.cmd.reference
        self.seconds=torch.arange(ref.num_frames,device=env.device)/float(ref.fps)
        self.reference_vz=torch.gradient(self.reference_bottom,spacing=(1/float(ref.fps),),dim=0)[0]
        segment=(self.seconds>=27.7-1e-5)&(self.seconds<=28.04+1e-5)
        self.height_table=self.reference_bottom[segment,1].flip(0).contiguous()
        self.speed_table=(-self.reference_vz[segment,1]).clamp_min(0).flip(0).contiguous()
        if len(self.height_table)<5 or not torch.all(torch.diff(self.height_table)>0):
            raise ValueError('Expected frozen monotonic T7 landing reference')
        self.weight_n=float(env.sim.mj_model.body_mass.sum())*9.81
        # Confirm reference-following is not penalized at the table samples.
        allowed=speed_target(-self.speed_table,interpolate_height_speed(self.height_table,self.height_table,self.speed_table))
        assert torch.all(self.speed_table<=allowed+1e-6)

    def __call__(self,env,ramp_steps=48000):
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
        loads=foot_load_history(env)[:,:,:env.cfg.decimation+1]
        time=self.seconds[frame];rb=self.reference_bottom[frame,1];rv=self.reference_vz[frame,1]
        physical_dt=env.cfg.sim.mujoco.timestep
        matched=interpolate_height_speed(bottom,self.height_table,self.speed_table)
        terms=landing_costs(time,bottom,down,rb,rv,loads,self.weight_n,physical_dt,matched)
        old=old_costs(time,bottom,down,rb,rv,loads[:,1],self.weight_n,physical_dt)
        old_total=sum(OLD_WEIGHTS[k]*v for k,v in old.items())
        new_total=sum(WEIGHTS[k]*v for k,v in terms.items())
        ramp=min(1.,env.common_step_counter/max(1,ramp_steps))
        self.cached=valid*((1-ramp)*old_total+ramp*new_total)
        for k,v in terms.items():c.metrics['t7_landing_'+k]=v*valid
        active=(phase_gate(time)>0).float()*valid
        for k,v in dict(active=active,cost=self.cached.clone(),peak_n=loads[:,1,:-1].amax(-1)*active,
                        total_peak_n=loads[:,:,:-1].sum(1).amax(-1)*active,down_m_s=down*active,
                        allowed_down_m_s=speed_target(rv,matched)*active,ramp=torch.full_like(valid,ramp)).items():
            c.metrics['t7_landing_'+k]=v
        self.last_frame.copy_(frame);self.last_step=env.common_step_counter
        return self.cached
