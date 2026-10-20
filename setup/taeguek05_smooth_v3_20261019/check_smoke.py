"""Validate a short training checkpoint and print its path for a resume test."""
from pathlib import Path
import torch
from refinement import MARKER, SOURCE_SHA

root = Path(__file__).resolve().parents[2]
paths = sorted((root/'logs/rsl_rl/k1_mimic').glob('*_taeguek05_smooth_v3_smoke/model_2.pt'))
path = paths[-1]
data = torch.load(path, map_location='cpu', weights_only=False)
assert data['infos'][MARKER]['sha256'] == SOURCE_SHA
assert data['infos']['env_state']['common_step_counter'] == 72
assert data['optimizer_state_dict']['state']
for key in ('actor_state_dict', 'critic_state_dict'):
    assert all(torch.isfinite(v).all() for v in data[key].values())
print(path)
resumes = sorted((root/'logs/rsl_rl/k1_mimic').glob('*_taeguek05_smooth_v3_resume_smoke/model_3.pt'))
if resumes:
    resumed = torch.load(resumes[-1], map_location='cpu', weights_only=False)
    assert resumed['infos']['env_state']['common_step_counter'] == 96
    assert resumed['infos'][MARKER]['sha256'] == SOURCE_SHA
    assert resumed['optimizer_state_dict']['state']
    for key in ('actor_state_dict', 'critic_state_dict'):
        assert all(torch.isfinite(v).all() for v in resumed[key].values())
    print('RESUME_CHECKPOINT_OK', resumes[-1])
