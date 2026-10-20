"""Export one saved server checkpoint using the installed RSL-RL model/exporters."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch
import yaml
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution
from rsl_rl.runners import OnPolicyRunner
from types import SimpleNamespace

torch.set_num_threads(2)
run, output = map(Path, sys.argv[1:3])
output.mkdir(parents=True, exist_ok=False)
checkpoint = run / sys.argv[3]
assert checkpoint.is_file()
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
before = sha(checkpoint)
ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
assert sha(checkpoint) == before, 'Checkpoint changed while reading'
expected_iter = int(checkpoint.stem.split('_')[-1])
assert ckpt['iter'] == expected_iter
if 'taeguek07' in run.name:
    assert expected_iter == 9999
    assert ckpt['infos']['env_state']['common_step_counter'] == 10000 * 24
    assert ckpt['infos']['taeguek07_arm_smooth_v1']['revision'] == '25d007624b3c6960ba324d9ef95f855f14094a1f57ba87a54aea74b1cd9fdc14'
else:
    assert 'taebaek_arm_refined_manual' in run.name and expected_iter in (10000,14000,17000)
assert ckpt['optimizer_state_dict']['state']
assert all(torch.isfinite(v).all() for key in ('actor_state_dict', 'critic_state_dict') for v in ckpt[key].values())
(output/'checkpoint').mkdir()
shutil.copy2(checkpoint, output/'checkpoint'/checkpoint.name)
assert sha(output/'checkpoint'/checkpoint.name) == before
shutil.copytree(run/'params', output/'params')
cfg = yaml.safe_load((output/'params/sim2real.yaml').read_text())
agent = yaml.load((output/'params/agent.yaml').read_text(), Loader=yaml.FullLoader)
env = yaml.load((output/'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
motion = Path(env['commands']['reference_trajectory']['trajectory_file'])
csv = motion.with_suffix('.csv')
shutil.copy2(csv, output/'params'/csv.name)
z = np.load(motion, allow_pickle=False)
rows = np.loadtxt(csv, delimiter=',')
joints = cfg['policy_joints']
assert list(z['joint_names']) == list(z['source_joint_names']) == joints
assert z['body_names'][0] == 'pelvis'
assert rows.shape == (z['joint_pos'].shape[0], 7 + len(joints))
assert np.isfinite(rows).all()
motion_errors = {
    'joint_position': float(np.max(np.abs(rows[:, 7:] - z['joint_pos']))),
    'root_position': float(np.max(np.abs(rows[:, :3] - z['body_pos_w'][:, 0]))),
    'root_quaternion': float(np.max(np.abs(rows[:, [6,3,4,5]] - z['body_quat_w'][:, 0]))),
}
assert max(motion_errors.values()) < 1e-6, motion_errors
assert abs(float(z['fps'][0]) * cfg['step_dt'] - 1) < 1e-8
state = ckpt['actor_state_dict']
dim = state['mlp.0.weight'].shape[1]
assert dim == sum(len(t['scale']) * t.get('history_length', 1) for t in cfg['observations'].values())
obs = TensorDict({'actor': torch.zeros(1, dim)}, batch_size=[1])
kwargs = copy.deepcopy(agent['actor'])
kwargs['distribution_cfg']['class_name'] = GaussianDistribution
actor = MLPModel(obs, agent['obs_groups'], 'actor', len(joints), **kwargs).cpu().eval()
actor.load_state_dict(state, strict=True)
assert all(torch.equal(actor.state_dict()[k], v) for k,v in state.items())
exported = output/'exported'
# Use precisely the exporter methods used by the project's play.py, without
# constructing a simulation or touching the training process/GPU.
runner = OnPolicyRunner.__new__(OnPolicyRunner)
runner.alg = SimpleNamespace(get_policy=lambda: actor)
runner.export_policy_to_onnx(str(exported), 'policy.onnx')
runner.export_policy_to_jit(str(exported), 'policy.pt')
model = onnx.load(exported/'policy.onnx')
metadata = {
    'run_path': run.name,
    'checkpoint': checkpoint.name,
    'checkpoint_sha256': before,
    'metadata_source': 'saved training params/sim2real.yaml',
    'joint_names': ','.join(joints),
    'joint_stiffness': ','.join(str(cfg['joint_properties'][j]['stiffness']) for j in joints),
    'joint_damping': ','.join(str(cfg['joint_properties'][j]['damping']) for j in joints),
    'default_joint_pos': ','.join(str(cfg['joint_properties'][j]['default_position']) for j in joints),
    'command_names': 'reference_trajectory',
    'observation_names': ','.join(cfg['observations']),
    'action_scale': ','.join(map(str, cfg['actions']['joint_pos']['scale'])),
}
onnx.helper.set_model_props(model, metadata)
onnx.save_model(model, exported/'policy.onnx', save_as_external_data=False)
onnx.checker.check_model(model, full_check=True)
evaluator = ReferenceEvaluator(model)
jit = torch.jit.load(str(exported/'policy.pt')).eval()
rng = np.random.default_rng(42)
max_onnx_error = max_jit_error = 0.0
for i in range(17):
    x = np.zeros((1,dim), dtype=np.float32) if i == 0 else rng.normal(0, (0.1,1.,3.,10.)[i%4], (1,dim)).astype(np.float32)
    with torch.no_grad():
        expected = actor(TensorDict({'actor': torch.from_numpy(x)}, batch_size=[1])).numpy()
        actual_jit = jit(torch.from_numpy(x)).numpy()
    actual_onnx = evaluator.run(None, {model.graph.input[0].name: x})[0]
    assert expected.shape == actual_jit.shape == actual_onnx.shape == (1,len(joints))
    np.testing.assert_allclose(actual_onnx, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_jit, expected, rtol=2e-5, atol=2e-5)
    max_onnx_error = max(max_onnx_error, float(np.max(abs(actual_onnx-expected))))
    max_jit_error = max(max_jit_error, float(np.max(abs(actual_jit-expected))))
files = {str(p.relative_to(output)): {'bytes':p.stat().st_size, 'sha256':sha(p)}
         for p in sorted(output.rglob('*')) if p.is_file()}
manifest = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'source_host': 'local hc',
    'source_run': str(run), 'checkpoint': str(checkpoint),
    'checkpoint_iteration': int(ckpt['iter']), 'completed_updates': int(ckpt['iter']) + 1, 'environment_steps': ckpt.get('infos', {}).get('env_state', {}).get('common_step_counter'), 'checkpoint_sha256': before,
    'selection': 'Frozen intermediate checkpoint for read-only progress evaluation',
    'pt_format': 'TorchScript inference model, not a training-resume checkpoint',
    'source_motion_npz':str(motion), 'source_motion_npz_sha256':sha(motion),
    'motion_csv':str(csv), 'motion_frames':int(rows.shape[0]), 'motion_fps':float(z['fps'][0]),
    'input':{'name':model.graph.input[0].name, 'shape':[1,dim]},
    'output':{'name':model.graph.output[0].name, 'shape':[1,len(joints)]},
    'validation':{'strict_actor_load':True, 'onnx_checker':'passed', 'test_inputs':17,
                  'actor_vs_onnx_max_absolute_error':max_onnx_error,
                  'actor_vs_jit_max_absolute_error':max_jit_error, 'csv_vs_npz_max_absolute_error':motion_errors},
    'versions':{'torch':torch.__version__, 'onnx':onnx.__version__}, 'files':files,
}
(exported/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2))
