from pathlib import Path
import os,subprocess,json,numpy as np
O=Path(__file__).resolve().parent;R=Path('/home/hc/robotis_mujoco');P=O/'bundle_9999'
B=Path('/home/hc/Documents/ChatGPT/태권도 학습 .. 로부월드/work/taeguek7_selected_vs_compose_20261019/probe')
s=O/'smoke_sequence.txt';s.write_text(f'0 5 {P}\n')
env={**os.environ,'LD_LIBRARY_PATH':f'{R}/build/localization_ws/install/ai_sapiens_interfaces/lib:/opt/ros/jazzy/lib:'+os.environ.get('LD_LIBRARY_PATH',''),'ROS_DOMAIN_ID':'198','KORYO_AUDIT_CSV':str(O/'smoke_rollout.csv'),'AUDIT_SEQUENCE':str(s),'AUDIT_RESET_SWITCH':'0'}
cmd=[str(B),'--ros-args','-p',f'model_xml_path:={R}/assets/ai_sapiens_description/mujoco/k1_rev1/k1_mjlab.xml','-p',f'policy_path:={P}','-p','policy_sim_dt:=0.005','-p','policy_decimation:=4','-p','mimic_reset_to_reference:=false','-p','headless_steps:=400']
with (O/'smoke.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
a=np.genfromtxt(O/'smoke_rollout.csv',delimiter=',',names=True)
assert len(a)==400 and a['active'].all() and (a['chapter']==5).all()
assert all(np.isfinite(a[n]).all() for n in a.dtype.names)
out={'physics_steps':400,'duration_seconds':2,'all_finite':True,'policy_active':True,'minimum_root_height_m':float(a['root_z'].min()),'quality_evaluation':'not performed'}
(O/'smoke.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
