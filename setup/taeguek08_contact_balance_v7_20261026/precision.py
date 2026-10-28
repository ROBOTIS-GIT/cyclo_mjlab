import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from source.tasks.mimic.mdp.performance_contacts import foot_load_history
from rewards import ContactGeometry
from costs import phase_drift
from precision_costs import stable_mask,costs,WEIGHTS

class ContactPrecision(ContactGeometry):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        ref=self.cmd.reference;fps=float(ref.fps)
        self.time=torch.arange(ref.num_frames,device=env.device)/fps
        sole_speed=torch.gradient(self.ref_sole[:,:,:2],spacing=(1/fps,),dim=0)[0].norm(dim=-1)
        up=torch.tensor([0.,0.,1.],device=env.device)
        gravity=quat_apply_inverse(ref.body_orientation_w[:,self.feet],up.expand(ref.num_frames,2,3))
        tilt_rate=torch.gradient(gravity,spacing=(1/fps,),dim=0)[0].norm(dim=-1)
        yaw_rate=ref.body_angular_velocity_w[:,self.feet,2]
        self.locked=stable_mask(self.reference_bottom,sole_speed,yaw_rate,tilt_rate,
                               self.phase,self.reference_support,fps)
        self.offset=torch.zeros(env.num_envs,2,2,device=env.device)
        self.previous=torch.zeros(env.num_envs,2,dtype=torch.bool,device=env.device)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'offset'):
            ids=slice(None) if env_ids is None else env_ids
            self.offset[ids]=0;self.previous[ids]=False

    def __call__(self,env,ramp_steps=48000):
        if self.last_step==env.common_step_counter:return self.cached
        frame,continuous,valid=self.advance(env)
        sole,_,speed,_,_=self.geometry(env);locked=self.locked[frame]
        # Hold displacement error relative to the recorded reference from quiet entry.
        # This removes the competing world-fixed target; actual contact loss cannot reset it.
        position=sole[...,:2]-env.scene.env_origins[:,None,:2]-self.ref_sole[frame,:,:2]
        drift,self.offset=phase_drift(position,self.offset,locked,self.previous,continuous)
        loads=foot_load_history(env)[:,:,:env.cfg.decimation+1]
        terms=costs(self.time[frame],loads,self.weight_n,env.cfg.sim.mujoco.timestep,locked,drift,speed)
        ramp=min(1.,env.common_step_counter/max(1,ramp_steps))
        self.cached=ramp*valid*sum(WEIGHTS[k]*terms[k] for k in WEIGHTS)
        for k,v in terms.items():self.cmd.metrics['t8_v6_'+k]=v*valid
        self.cmd.metrics['t8_v6_cost']=self.cached.clone()
        self.cmd.metrics['t8_v6_ramp']=torch.full_like(valid,ramp)
        self.cmd.metrics['t8_v6_locked_feet']=locked.sum(-1).float()*valid
        self.previous.copy_(locked);self.last_frame.copy_(frame);self.last_step=env.common_step_counter
        return self.cached
