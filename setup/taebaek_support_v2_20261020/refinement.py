"""Reference-gated foot stabilization for the evaluated Taebaek arm policy."""
from pathlib import Path
import hashlib
import json
import math
from collections import deque
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.lab_api.math import quat_apply
from source.tasks.mimic.config.k1_rev1.base_env_cfg import k1_rev1_mimic_env_cfg
from source.tasks.mimic.config.k1_rev1.agents.rsl_rl_ppo_cfg import k1_rev1_mimic_ppo_runner_cfg
from source.tasks.mimic.mdp.performance_contacts import add_foot_sensors
from source.tasks.mimic.mdp.pelvis_xy_rewards import reference_pelvis_xy
from source.tasks.mimic.mdp.taeguek_support import TaegeukFootSupportQuality
from source.tasks.mimic.mdp.taeguek_support_v2 import sole_bottom_height

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'source/model_17000.pt'
SOURCE_SHA = '6653bc0311b6de47fcec59232af125d781a991efb4b44eed86e09608112d3564'
MOTION = HERE / 'motion/k1_taebaek_arm_refined_50hz_converted.npz'
MOTION_SHA = '6145604652ebb3aaa8fabf7be7682ff04ddea9ae1d33c11f17936127b04c47f9'
MARKER = 'taebaek_support_v2'
RUN_NAME = MARKER + '_finetune'
TOTAL_UPDATES = 10000


def revision():
    digest = hashlib.sha256()
    paths = [HERE/'refinement.py'] + [
        ROOT/'source/tasks/mimic/mdp'/f for f in
        ('support_contacts.py', 'taeguek_support.py', 'taeguek_support_v2.py',
         'performance_contacts.py', 'contact_transitions.py')]
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_inputs():
    for path, expected in ((SOURCE, SOURCE_SHA), (MOTION, MOTION_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen input changed: {path}')


MAX_SUPPORT_COST = .6

def support_ramp(steps, ramp_steps=120000):
    return min(1., .1 + .9 * steps / max(1, ramp_steps))

def bounded_support_cost(raw):
    return MAX_SUPPORT_COST * torch.tanh(raw / MAX_SUPPORT_COST)

def guard_update(state, iteration, mean_steps):
    state = dict(state)
    if iteration <= state.get("last_iteration", -1):
        return state
    best = max(state.get("best_steps", 0.), mean_steps)
    bad = iteration >= 500 and (mean_steps < 600 or (best >= 1200 and mean_steps < .5 * best))
    state.update(last_iteration=iteration, best_steps=best, mean_steps=mean_steps,
                 bad_checks=state.get("bad_checks", 0)+1 if bad else 0)
    state["halt"] = state["bad_checks"] >= 2
    return state

def huber(x):
    return torch.sqrt(1 + x.square()) - 1


def foot_costs(drift, angular_error, angular_acc_error, bottom, target_bottom,
               vertical_speed_error, support):
    """All geometric terms stay active without contact; no hovering loophole."""
    return {
        'drift': huber((drift.norm(dim=-1) - .02).clamp_min(0) / .05) * support,
        'rocking': huber((angular_error[..., :2].norm(dim=-1) - .35).clamp_min(0)) * support,
        'rocking_acc': huber((angular_acc_error[..., :2].norm(dim=-1) - 5).clamp_min(0) / 20) * support,
        'clearance': huber((bottom - target_bottom.clamp_min(0) - .01).clamp_min(0) / .02) * support,
        'rising': huber((vertical_speed_error - .05).clamp_min(0) / .25) * support,
    }


def phase_drift(error, offset, support, previous_support, continuous):
    start = support & (~previous_support | ~continuous[:, None])
    offset = torch.where(start[..., None], error, offset)
    return error - offset, offset


class SupportStability(TaegeukFootSupportQuality):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.point = torch.tensor([.0325, 0, -.065], device=env.device)
        ref = self.cmd.reference
        self.ref_sole = ref.body_position_w[:, self.feet] + quat_apply(
            ref.body_orientation_w[:, self.feet],
            self.point.expand(ref.num_frames, 2, 3))
        self.target_vz = torch.gradient(self.reference_bottom,
                                       spacing=(1 / float(ref.fps),), dim=0)[0]
        self.phase_offset = torch.zeros(env.num_envs, 2, 2, device=env.device)
        self.prev_expected = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        self.prev_angular_error = torch.zeros(env.num_envs, 2, 3, device=env.device)
        self.prev_bottom = torch.zeros(env.num_envs, 2, device=env.device)
        self.geom_frame = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.geom_cache = torch.zeros(env.num_envs, device=env.device)
        self.geom_step = -1
        print('TAEBAEK_SUPPORT_V2_REFERENCE_SECONDS',
              (self.reference_support.sum(0) / float(ref.fps)).tolist(), flush=True)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, 'geom_frame'):
            ids = slice(None) if env_ids is None else env_ids
            self.geom_frame[ids] = -1
            self.prev_expected[ids] = False
            self.phase_offset[ids] = 0
            self.prev_angular_error[ids] = 0
            self.geom_cache[ids] = 0
            self.geom_step = -1

    def __call__(self, env, drift_weight=.08, rocking_weight=.05,
                 rocking_acc_weight=0., clearance_weight=.08,
                 rising_weight=.02, ramp_steps=120000):
        if self.geom_step == env.common_step_counter:
            return self.geom_cache
        # The base class handles reference discontinuities and reset grace.
        base = super().__call__(
            env, support_weight=.2, long_unload_weight=.05, liftoff_weight=.01,
            slip_weight=.1, rotation_weight=0.,
            single_support_multiplier=1.2, ramp_steps=1)
        frame = self.cmd.frame_ids
        support = self.reference_support[frame]
        continuous = (frame == self.geom_frame + 1) & (env.episode_length_buf > 1)
        position = self.cmd.robot_body_pos_w[:, self.feet]
        quaternion = self.cmd.robot_body_quat_w[:, self.feet]
        sole = position + quat_apply(quaternion, self.point.expand(env.num_envs, 2, 3))
        error = sole[..., :2] - env.scene.env_origins[:, None, :2] - self.ref_sole[frame, :, :2]
        # Fixed reference-phase entry, never reset when actual foot loses contact.
        drift, self.phase_offset = phase_drift(
            error, self.phase_offset, support, self.prev_expected, continuous)
        angular_error = (self.cmd.robot_body_ang_vel_w[:, self.feet]
                         - self.cmd.body_ang_vel_w[:, self.feet])
        smooth_valid = continuous[:, None] & self.prev_expected & support
        acceleration = torch.where(smooth_valid[..., None],
            (angular_error - self.prev_angular_error) / env.step_dt, 0.)
        bottom = sole_bottom_height(position, quaternion, self.offsets, self.radii)
        bottom -= env.scene.env_origins[:, 2, None]
        vz_error = torch.where(smooth_valid,
            (bottom - self.prev_bottom) / env.step_dt - self.target_vz[frame], 0.)
        costs = foot_costs(drift, angular_error, acceleration, bottom,
                          self.reference_bottom[frame], vz_error, support)
        weights = (drift_weight, rocking_weight, rocking_acc_weight,
                   clearance_weight, rising_weight)
        valid = (self.age > .15).float()
        ramp = support_ramp(env.common_step_counter, ramp_steps)
        single = torch.where(support.sum(-1) == 1, 1.2, 1.)
        raw = base + single * valid * sum(
            w * c.sum(-1) for w, c in zip(weights, costs.values()))
        self.geom_cache = ramp * bounded_support_cost(raw)
        self.cmd.metrics["taebaek_v2_raw_cost"] = raw
        self.cmd.metrics["taebaek_v2_bounded_cost"] = self.geom_cache
        self.cmd.metrics["taebaek_v2_ramp"] = torch.full_like(raw, ramp)
        for key, value in costs.items():
            self.cmd.metrics['taebaek_support_' + key] = value.sum(-1) * valid
        self.cmd.metrics['taebaek_support_drift_m'] = (drift.norm(dim=-1) * support).sum(-1) * valid
        self.prev_expected.copy_(support)
        self.prev_angular_error.copy_(angular_error)
        self.prev_bottom.copy_(bottom)
        self.geom_frame.copy_(frame)
        self.geom_step = env.common_step_counter
        return self.geom_cache


def env_cfg(play=False):
    cfg = add_foot_sensors(k1_rev1_mimic_env_cfg(play=play))
    cmd = cfg.commands['reference_trajectory']
    cmd.trajectory_file = str(MOTION)
    cmd.first_frame_env_fraction = .2 if not play else 0.
    cmd.start_sampling_uniform_mix = .2
    cfg.episode_length_s = 49.2 if not play else int(1e9)
    cfg.scene.num_envs = 4096 if not play else 1
    cfg.rewards['reference_pelvis_xy'] = RewardTermCfg(
        func=reference_pelvis_xy, weight=1.,
        params={'command_name': 'reference_trajectory', 'std': .5})
    cfg.rewards['taebaek_support_stability'] = RewardTermCfg(
        func=SupportStability, weight=-1., params={})
    return cfg


def agent_cfg():
    cfg = k1_rev1_mimic_ppo_runner_cfg()
    cfg.algorithm.learning_rate = 2e-5
    cfg.algorithm.schedule = 'fixed'
    cfg.algorithm.entropy_coef = .003
    cfg.max_iterations = TOTAL_UPDATES
    cfg.save_interval = 250
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
    if meta.get('support_reward_weight') != -1.:
        raise ValueError('Control ablation cannot be resumed as production')
    if num_envs is not None and meta['num_envs'] != num_envs:
        raise ValueError('Resume environment count mismatch')
    if not data.get('optimizer_state_dict', {}).get('state'):
        raise ValueError('Missing optimizer state')
    return completed


class SupportRunner(MjlabOnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard_state = {}
        self.length_window = deque(maxlen=50)
        original_log = self.logger.log
        def guarded_log(*args, **kwargs):
            if self.logger.lenbuffer:
                self.length_window.append(sum(self.logger.lenbuffer)/len(self.logger.lenbuffer))
            return original_log(*args, **kwargs)
        self.logger.log = guarded_log

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
            self.guard_state = data['infos'].get('tracking_guard', {})
            if self.guard_state.get('halt'):
                raise RuntimeError('This checkpoint failed tracking guard; review before resuming.')
            print('TAEBAEK_SUPPORT_V2_RESUME', completed, flush=True)
            return result
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SOURCE_SHA:
            raise ValueError('Warm start requires the evaluated model_17000.pt')
        result = super().load(path, dict(actor=True, critic=True, optimizer=False,
                             iteration=False, rnd=False), strict, map_location or self.device)
        self.current_learning_iteration = 0
        self.env.unwrapped.common_step_counter = 0
        if self.alg.optimizer.state:
            raise RuntimeError('Expected fresh optimizer')
        for model, key in ((self.alg.actor, 'actor_state_dict'), (self.alg.critic, 'critic_state_dict')):
            if not all(torch.equal(v.cpu(), data[key][k]) for k, v in model.state_dict().items()):
                raise RuntimeError('Warm-start weights differ from source')
        self.provenance = dict(sha256=SOURCE_SHA, source_iteration=17000,
                              revision=revision(), num_envs=self.env.unwrapped.num_envs,
                              fresh_optimizer=True, support_reward_weight=self.env.unwrapped.cfg.rewards['taebaek_support_stability'].weight)
        print('TAEBAEK_SUPPORT_V2_WARMSTART', self.provenance, flush=True)
        return result

    def save(self, path, infos=None):
        if len(self.length_window) >= 25:
            self.guard_state = guard_update(self.guard_state, self.current_learning_iteration,
                                           sum(self.length_window)/len(self.length_window))
        super().save(path, {**(infos or {}), MARKER: self.provenance,
                            'tracking_guard': self.guard_state})
        if self.guard_state:
            print('TAEBAEK_V2_TRACKING_GUARD', json.dumps(self.guard_state), flush=True)
        if self.guard_state.get('halt'):
            record = dict(self.guard_state, checkpoint=str(path), reason='Episode duration regressed on two consecutive checkpoint checks')
            marker = HERE/'tracking_guard_halt.json'
            tmp = marker.with_suffix('.tmp');tmp.write_text(json.dumps(record, indent=2));tmp.replace(marker)
            raise RuntimeError('Tracking guard stopped training after saving checkpoint. Inspect tracking_guard_halt.json.')
