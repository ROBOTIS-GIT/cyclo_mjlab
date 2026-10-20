from pathlib import Path
import sys,json
from types import SimpleNamespace as NS
import numpy as np,torch,mujoco
sys.path[:0]=[str(Path.cwd()),str(Path('setup/taebaek_support_v1_20261020').resolve())]
from refinement import SupportStability,MOTION
from source.tasks.mimic.config.k1_rev1.base_env_cfg import TRACKED_BODY_NAMES
from mjlab.utils.lab_api.math import quat_apply
from source.assets.robots.robotis_k1.k1_rev1 import _load_k1_rev1_spec
z=np.load(MOTION);model=_load_k1_rev1_spec().compile()
ref=NS(fps=float(z['fps'][0]),num_frames=len(z['joint_pos']))
for key,attr in [('body_pos_w','body_position_w'),('body_quat_w','body_orientation_w'),('body_lin_vel_w','body_linear_velocity_w'),('body_ang_vel_w','body_angular_velocity_w')]:
 setattr(ref,attr,torch.tensor(z[key]))
cmd=NS(reference=ref,cfg=NS(body_names=list(z['body_names'])),metrics={})
env=NS(command_manager=NS(get_term=lambda _:cmd),device='cpu',num_envs=1,cfg=NS(sim=NS(mujoco=NS(timestep=.005))),sim=NS(mj_model=model))
reward=SupportStability(None,env)
rows=[]
for time in [4.8,5.0,5.02,5.2,5.4,5.6,5.8]:
 f=round(time*50);rows.append(dict(time_s=time,support_left_right=reward.reference_support[f].tolist(),bottom_left_right_m=reward.reference_bottom[f].tolist()))
result=dict(phase_samples=rows,support_seconds=(reward.reference_support.sum(0)/50).tolist())
Path('setup/taebaek_support_diagnosis_20261020/reference_labels.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
