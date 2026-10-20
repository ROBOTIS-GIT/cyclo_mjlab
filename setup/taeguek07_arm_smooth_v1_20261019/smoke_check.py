"""Short training/resume verification, excluded from production checkpoint selection."""
from pathlib import Path
import datetime
import json
import os
import subprocess
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from refinement import SOURCE, SOURCE_SHA, MARKER, revision

os.chdir(ROOT)
tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
logroot = ROOT / 'logs/rsl_rl/k1_mimic'


def run(checkpoint, count, name):
    command = [sys.executable, '-u', str(HERE/'train.py'), '--checkpoint', str(checkpoint),
        '--gpu-ids', '[0]', '--env.scene.num-envs', '4096', '--agent.max-iterations', str(count),
        '--agent.run-name', name, '--agent.logger', 'tensorboard',
        '--agent.resume', 'False', '--agent.upload-model', 'False']
    with (HERE/(name+'.log')).open('w') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=180)
    matches = list(logroot.glob('*_'+name))
    assert len(matches) == 1
    return matches[0]


def check(path, completed):
    data = torch.load(path, map_location='cpu', weights_only=False)
    assert data['iter'] + 1 == completed
    assert data['infos']['env_state']['common_step_counter'] == completed * 24
    meta = data['infos'][MARKER]
    assert meta['sha256'] == SOURCE_SHA and meta['revision'] == revision()
    opt = data['optimizer_state_dict']
    assert opt['state'] and all(group['lr'] == 5e-5 for group in opt['param_groups'])
    assert all(torch.isfinite(v).all() for key in ('actor_state_dict','critic_state_dict') for v in data[key].values())
    return data


first = run(SOURCE, 3, f'taeguek07_arm_smooth_v1_smoke_{tag}')/'model_2.pt'
initial = check(first, 3)
second = run(first, 1, f'taeguek07_arm_smooth_v1_resume_smoke_{tag}')/'model_3.pt'
resumed = check(second, 4)
step1 = min(float(v['step']) for v in initial['optimizer_state_dict']['state'].values())
step2 = min(float(v['step']) for v in resumed['optimizer_state_dict']['state'].values())
assert step2 > step1 > 0
result = dict(first_checkpoint=str(first), resumed_checkpoint=str(second),
    completed_updates=[3,4], environment_steps=[72,96], optimizer_steps=[step1,step2],
    num_envs=4096, revision=revision(), source_sha256=SOURCE_SHA)
(HERE/'smoke_check.json').write_text(json.dumps(result, indent=2))
print('SMOKE_AND_RESUME_OK', json.dumps(result), flush=True)
