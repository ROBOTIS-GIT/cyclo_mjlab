"""T8 targeted whole-landing and quiet-support refinement, leaving V3 intact."""
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from source.tasks.mimic.mdp.performance_contacts import foot_load_history
from rewards import ContactGeometry
from costs import phase_drift
from refinement_costs import WEIGHTS, landing_gates, total_impact, approach_terms, quiet_terms, support_gap

class LandingSupportRefinement(ContactGeometry):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        ref=self.cmd.reference
        self.time=torch.arange(ref.num_frames,device=env.device)/float(ref.fps)
        self.ref_vz=torch.gradient(self.reference_bottom,spacing=(1/float(ref.fps),),dim=0)[0]
        self.pelvis=self.cmd.cfg.body_names.index('pelvis')
        self.up=torch.tensor([0.,0.,1.],device=env.device)
        gravity=quat_apply_inverse(ref.body_orientation_w[:,self.feet],self.up.expand(ref.num_frames,2,3))
        self.ref_tilt_rate=torch.gradient(gravity,spacing=(1/float(ref.fps),),dim=0)[0].norm(dim=-1)
        self.phase_offset=torch.zeros(env.num_envs,2,2,device=env.device)
        self.previous=torch.zeros(env.num_envs,2,dtype=torch.bool,device=env.device)
        # Precompute ages from reference labels, never from actual contact.
        stable=(self.reference_support & (self.phase==1)).cpu()
        ages=torch.zeros_like(stable,dtype=torch.float32)
        for i in range(1,ref.num_frames):
            ages[i]=torch.where(stable[i]&stable[i-1],ages[i-1]+1/float(ref.fps),0.)
        self.reference_age=ages.to(env.device)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'phase_offset'):
            ids=slice(None) if env_ids is None else env_ids
            self.phase_offset[ids]=0;self.previous[ids]=False

    def __call__(self,env,ramp_steps=72000):
        if self.last_step==env.common_step_counter:return self.cached
        frame,continuous,valid=self.advance(env)
        c=self.cmd;t=self.time[frame];gates=landing_gates(t)
        sole,bottom,speed,_,down=self.geometry(env)
        loads=foot_load_history(env)[:,:,:env.cfg.decimation+1]
        p,r,e=total_impact(loads,self.weight_n,env.cfg.sim.mujoco.timestep)
        active=gates.amax(-1)
        terms=dict(total_peak=p*active,total_rate=r*active,total_excess=e*active)
        expected=self.reference_support[frame];phase=self.phase[frame]
        # Contact occupancy over the entire policy interval, not one endpoint.
        occupancy=(loads[:,:,:-1]>5.).float().mean(-1)
        terms.update(approach_terms(bottom,down,self.reference_bottom[frame],self.ref_vz[frame],
            c.robot_body_pos_w[:,self.pelvis,2]-env.scene.env_origins[:,2],
            c.reference.body_position_w[frame,self.pelvis,2],
            c.robot_body_lin_vel_w[:,self.pelvis,2],c.body_lin_vel_w[:,self.pelvis,2],
            gates,occupancy,expected))
        planted=expected&(phase==1)
        error=sole[...,:2]-env.scene.env_origins[:,None,:2]-self.ref_sole[frame,:,:2]
        drift,self.phase_offset=phase_drift(error,self.phase_offset,planted,self.previous,continuous)
        terms.update(quiet_terms(t,drift,speed,self.ref_speed[frame],
            c.robot_body_ang_vel_w[:,self.feet]-c.body_ang_vel_w[:,self.feet],
            self.ref_tilt_rate[frame],phase,expected))
        terms['support_gap']=support_gap(loads,self.weight_n,self.ref_speed[frame],
            self.ref_tilt_rate[frame],phase,expected,self.reference_age[frame])
        ramp=min(1.,env.common_step_counter/max(1,ramp_steps))
        self.cached=ramp*valid*sum(WEIGHTS[k]*terms[k] for k in WEIGHTS)
        for k,v in terms.items():c.metrics['t8_v5_'+k]=v*valid
        c.metrics['t8_v5_cost']=self.cached.clone()
        c.metrics['t8_v5_ramp']=torch.full_like(valid,ramp)
        c.metrics['t8_v5_active']=active*valid
        c.metrics['t8_v5_total_peak_n']=loads[:,:,:-1].sum(1).amax(-1)*active*valid
        self.previous.copy_(planted);self.last_frame.copy_(frame);self.last_step=env.common_step_counter
        return self.cached
