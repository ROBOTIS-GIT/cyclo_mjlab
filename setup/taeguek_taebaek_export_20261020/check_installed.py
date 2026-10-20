from pathlib import Path
import os,subprocess,json,numpy as np,sys
O=Path(__file__).resolve().parent
R=Path('/home/hc/robotis_mujoco')
name=sys.argv[1]
P=R/'policy/Mimic/taeguek_1020_compse'/name
B=Path('/home/hc/Documents/ChatGPT/태권도 학습 .. 로부월드/work/taeguek7_selected_vs_compose_20261019/probe')
M=json.loads((P/'exported/manifest.json').read_text())
steps=int((M['motion_frames']-1)/M['motion_fps']/.005)
s=O/f'{name}_sequence.txt';s.write_text(f'0 7 {P}\n')
csv=O/f'{name}_rollout.csv'
env={**os.environ,'LD_LIBRARY_PATH':f'{R}/build/localization_ws/install/ai_sapiens_interfaces/lib:/opt/ros/jazzy/lib:'+os.environ.get('LD_LIBRARY_PATH',''),'ROS_DOMAIN_ID':'198','KORYO_AUDIT_CSV':str(csv),'AUDIT_SEQUENCE':str(s),'AUDIT_RESET_SWITCH':'0'}
cmd=[str(B),'--ros-args','-p',f'model_xml_path:={R}/assets/ai_sapiens_description/mujoco/k1_rev1/k1_mjlab.xml','-p',f'policy_path:={P}','-p','policy_sim_dt:=0.005','-p','policy_decimation:=4','-p','mimic_reset_to_reference:=false','-p',f'headless_steps:={steps}']
with (O/f'{name}_smoke.log').open('w') as f:
 subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=90)
a=np.genfromtxt(csv,delimiter=',',names=True)
assert len(a)==steps and a['active'].all()
assert all(np.isfinite(a[n]).all() for n in a.dtype.names)
out={'policy':name,'physics_steps':steps,'duration_seconds':steps*.005,'all_finite':True,'policy_active':True,'minimum_root_height_m':float(a['root_z'].min()),'root_below_035m':bool((a['root_z']<.35).any()),'scope':'single default-pose full-motion headless compatibility check; not a comparative or sim2real quality assessment'}
(O/f'{name}_smoke.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
