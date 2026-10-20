"""Taegeuk 7: retain mature arm/support tracking, add micro-motion and soft contact."""
from pathlib import Path
import hashlib
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.lab_api.math import quat_apply
from source.tasks.mimic.mdp.performance_contacts import foot_load_history
from source.tasks.mimic.mdp.taeguek_support import TaegeukFootSupportQuality
from source.tasks.mimic.mdp.taeguek_support_v2 import sole_bottom_height
from foot_micro import SupportStability, huber
import arm_base

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'source/model_9999.pt'
SOURCE_SHA = '2228c9f5e3f6e3bd8849a46894066347700f208e6a0c9426016ad88d72cfc7cc'
MOTION = HERE / 'motion/k1_taeguek07_arm_corrected_v1_50hz_converted.npz'
MOTION_SHA = '877390d22ab136a8fc99726e240c09e73efdd2d2b9ff4d4524a6c1c90c0346df'
MARKER = 'taeguek07_soft_contact_v1'
RUN_NAME = MARKER + '_finetune'
TOTAL_UPDATES = 10000
arm_base.MOTION = MOTION


def revision():
    h = hashlib.sha256()
    for name in ('refinement.py', 'arm_base.py', 'targeted_base.py', 'foot_micro.py', 'entry_t7.npz'):
        h.update(name.encode()); h.update((HERE/name).read_bytes())
    for name in ('support_contacts.py','taeguek_support.py','taeguek_support_v2.py','performance_contacts.py','contact_transitions.py'):
        h.update(name.encode()); h.update((ROOT/'source/tasks/mimic/mdp'/name).read_bytes())
    return h.hexdigest()


def validate_inputs():
    for path, expected in ((SOURCE, SOURCE_SHA), (MOTION, MOTION_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen input changed: {path}')


def impact_costs(loads, weight_n, physics_dt):
    # Histories are newest-first and contain decimation+1 samples.
    current = loads[:, :, :-1]
    per_foot = current.amax(-1) / weight_n
    both = current.sum(1).amax(-1) / weight_n
    peak = huber((per_foot - 2.2).clamp_min(0)).sum(-1)
    peak += huber((both - 3.).clamp_min(0) / 1.5)
    # Penalize rapid load application, not unloading to begin a kick.
    rise = ((loads[:, :, :-1] - loads[:, :, 1:]) / (weight_n * physics_dt)).clamp_min(0)
    loading_rate = huber((rise - 80.).clamp_min(0) / 120.).amax(-1).sum(-1)
    return peak, loading_rate


def approach_cost(speed_down, bottom, reference_bottom):
    # A fast airborne kick away from the floor is not a landing event.
    gate = (bottom < .04) & (reference_bottom < .06)
    return (huber((speed_down - .6).clamp_min(0) / .6) * gate).sum(-1)


class SoftContact(TaegeukFootSupportQuality):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.weight_n = float(env.sim.mj_model.body_mass.sum()) * 9.81
        print('T7_SOFT_CONTACT_NOMINAL_WEIGHT_N', self.weight_n, flush=True)

    def __call__(self, env, peak_weight=.5, loading_rate_weight=.08,
                 approach_weight=.12, ramp_steps=24000):
        if self.last_step == env.common_step_counter:
            return self.cached
        frame = self.cmd.frame_ids
        continuous = (frame == self.last_frame + 1) & (env.episode_length_buf > 1)
        self.age = torch.where(continuous, self.age + env.step_dt, 0.)
        valid = (self.age > .15).float()
        loads = foot_load_history(env)[:, :, :env.cfg.decimation+1]
        peak, rate = impact_costs(loads, self.weight_n, env.cfg.sim.mujoco.timestep)
        q = self.cmd.robot_body_quat_w[:, self.feet]
        pos = self.cmd.robot_body_pos_w[:, self.feet]
        count = self.offsets.shape[1]
        rotated = quat_apply(q[:,:,None,:].expand(-1,-1,count,-1),
                             self.offsets[None].expand(env.num_envs,-1,-1,-1))
        surface = rotated.clone(); surface[...,2] -= self.radii[None]
        com = quat_apply(q, self.com[None].expand(env.num_envs,-1,-1))
        v = self.cmd.robot_body_lin_vel_w[:,self.feet,None,:] + torch.cross(
            self.cmd.robot_body_ang_vel_w[:,self.feet,None,:].expand_as(surface),
            surface-com[:,:,None,:],dim=-1)
        low = surface[...,2] <= surface[...,2].amin(-1,keepdim=True)+.005
        speed_down = torch.where(low, -v[...,2], -torch.inf).amax(-1).clamp_min(0)
        bottom = sole_bottom_height(pos,q,self.offsets,self.radii)-env.scene.env_origins[:,2,None]
        approach = approach_cost(speed_down,bottom,self.reference_bottom[frame])
        ramp = min(1.,.2+.8*env.common_step_counter/max(1,ramp_steps))
        self.cached = ramp*valid*(peak_weight*peak+loading_rate_weight*rate+approach_weight*approach)
        for name,value in [('peak_cost',peak),('loading_rate_cost',rate),('approach_cost',approach),
                           ('max_foot_force_n',loads[:,:,:-1].amax(dim=(1,2))),
                           ('total_peak_force_n',loads[:,:,:-1].sum(1).amax(-1))]:
            self.cmd.metrics['t7_soft_'+name] = value*valid
        self.last_frame.copy_(frame); self.last_step=env.common_step_counter
        return self.cached


def env_cfg(play=False):
    cfg = arm_base.env_cfg(play)
    # Keep the already trained contact/flight-height and arm objectives mature.
    cfg.rewards['arm_stability'].params = dict(ramp_steps=1)
    cfg.rewards['support_micro_motion'] = RewardTermCfg(func=SupportStability,weight=-1.,params={
        'drift_weight':.35,'rocking_weight':.25,'rocking_acc_weight':.04,
        'clearance_weight':0.,'rising_weight':0.,'ramp_steps':24000})
    cfg.rewards['soft_ground_contact'] = RewardTermCfg(func=SoftContact,weight=-1.,params={})
    return cfg


def agent_cfg():
    cfg = arm_base.agent_cfg()
    cfg.algorithm.learning_rate = 5e-5
    cfg.algorithm.schedule = 'fixed'
    cfg.algorithm.entropy_coef = .001
    cfg.max_iterations = TOTAL_UPDATES
    cfg.run_name = RUN_NAME
    cfg.resume = False
    cfg.upload_model = False
    return cfg


def validate_resume(data, num_envs=None):
    meta = (data.get('infos') or {}).get(MARKER)
    if not meta or meta.get('revision') != revision() or meta.get('sha256') != SOURCE_SHA:
        raise ValueError('Checkpoint source/reward revision mismatch')
    completed = int(data['iter']) + 1
    if not 0 < completed <= TOTAL_UPDATES:
        raise ValueError('Invalid update count')
    if data['infos']['env_state']['common_step_counter'] != completed * 24:
        raise ValueError('Iteration/environment counter mismatch')
    if num_envs is not None and meta['num_envs'] != num_envs:
        raise ValueError('Resume environment count mismatch')
    if not data.get('optimizer_state_dict', {}).get('state'):
        raise ValueError('Missing optimizer state')
    return completed


class SupportRunner(MjlabOnPolicyRunner):
    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if load_cfg is not None:
            raise ValueError('Managed checkpoint loading only')
        data = torch.load(path, map_location='cpu', weights_only=False)
        if not all(torch.isfinite(v).all() for k in ('actor_state_dict', 'critic_state_dict') for v in data[k].values()):
            raise ValueError('Non-finite source policy')
        if (data.get('infos') or {}).get(MARKER):
            completed = validate_resume(data, self.env.unwrapped.num_envs)
            result = super().load(path, None, strict, map_location or self.device)
            self.current_learning_iteration = completed
            self.provenance = data['infos'][MARKER]
            print('T7_SOFT_CONTACT_RESUME', completed, flush=True)
            return result
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SOURCE_SHA:
            raise ValueError('Warm start requires the evaluated model_9999.pt')
        result = super().load(path, dict(actor=True, critic=True, optimizer=False,
                             iteration=False, rnd=False), strict, map_location or self.device)
        self.current_learning_iteration = 0
        self.env.unwrapped.common_step_counter = 0
        if self.alg.optimizer.state:
            raise RuntimeError('Expected fresh optimizer')
        for model, key in ((self.alg.actor, 'actor_state_dict'), (self.alg.critic, 'critic_state_dict')):
            if not all(torch.equal(v.cpu(), data[key][k]) for k, v in model.state_dict().items()):
                raise RuntimeError('Warm-start weights differ from source')
        self.provenance = dict(sha256=SOURCE_SHA, source_iteration=9999,
                              revision=revision(), num_envs=self.env.unwrapped.num_envs,
                              fresh_optimizer=True)
        print('T7_SOFT_CONTACT_WARMSTART', self.provenance, flush=True)
        return result

    def save(self, path, infos=None):
        super().save(path, {**(infos or {}), MARKER: self.provenance})
