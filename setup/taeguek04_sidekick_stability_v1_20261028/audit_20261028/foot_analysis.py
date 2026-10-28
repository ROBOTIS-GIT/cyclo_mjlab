from pathlib import Path
import json,numpy as np,mujoco as mj
base=Path('/home/hc/robotis/cyclo_mjlab/setup/taeguek04_grf_v1_20261027/audit_20261028');out=Path('/home/hc/robotis/cyclo_mjlab/setup/taeguek04_sidekick_stability_v1_20261028/audit_20261028');r=np.load('/home/hc/Documents/ChatGPT/태권도 학습 .. 로부월드/work/taeguek45_v2_final_audit_20261019/ref_t4.npz');t=r['time'];m=mj.MjModel.from_xml_path('/home/hc/robotis_mujoco/assets/ai_sapiens_description/mujoco/k1_rev1/k1_mjlab.xml');d=mj.MjData(m);id=m.body('left_ankle_roll_link').id
result={}
for label,path in [('source',base),('middle',out),('final',out)]:
 for mode in ['reset','transition']:
  rows=np.genfromtxt(path/f'{"middle" if label=="source" else label}_{mode}.csv',delimiter=',',names=True);rows=rows[(rows['time']>=14.4)&(rows['time']<14.85)];rot=[]
  for row in rows:
   d.qpos[:]=[row[f'q{i}'] for i in range(m.nq)];mj.mj_forward(m,d);rot.append(d.xmat[id].reshape(3,3).copy())
  rot=np.array(rot);index=np.minimum(np.rint(rows['time']/.02).astype(int),len(t)-1);rr=r['rot'][index,0]
  def angles(x):return np.rad2deg(np.arctan2(x[:,2,1],x[:,2,2])),np.rad2deg(np.arctan2(-x[:,2,0],np.hypot(x[:,2,1],x[:,2,2])))
  a,b=angles(rot);ar,br=angles(rr);result[label+'_'+mode]={'roll_range_deg':float(np.ptp(a)),'pitch_range_deg':float(np.ptp(b)),'ref_roll_range_deg':float(np.ptp(ar)),'ref_pitch_range_deg':float(np.ptp(br)),'roll_error_rms_deg':float(np.sqrt(np.mean((a-ar)**2))),'pitch_error_rms_deg':float(np.sqrt(np.mean((b-br)**2)))}
  print(label,mode,result[label+'_'+mode])
(out/'foot_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
