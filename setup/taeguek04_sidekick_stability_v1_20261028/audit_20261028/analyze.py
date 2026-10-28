from pathlib import Path
import numpy as np,json
base=Path('/home/hc/Documents/ChatGPT/태권도 학습 .. 로부월드/work/taeguek45_v2_final_audit_20261019'); out=Path('/home/hc/robotis/cyclo_mjlab/setup/taeguek04_grf_v1_20261027/audit_20261028'); r=np.load(base/'ref_t4.npz'); rt=r['time']; dt=.005; bw=35.70603989*9.81

def spans(mask):
 e=np.diff(np.r_[False,mask,False].astype(int));return list(zip(np.flatnonzero(e==1),np.flatnonzero(e==-1)))
wins=[]
for f in range(2):
 for a,b in spans((r['bottom'][:,1-f]>.25)&(r['bottom'][:,f]<.012)):
  if b-a>=8:wins.append((float(rt[a]),float(rt[b-1]+.02),f))
wins.sort(); print('KICK_WINDOWS',wins,'bw',bw)
results={}
for name,p in [('grfv2_reset',Path('/home/hc/robotis/cyclo_mjlab/setup/taeguek04_grf_support_v2_20261028/playback_model_9999/reset.csv')),('grfv2_transition',Path('/home/hc/robotis/cyclo_mjlab/setup/taeguek04_grf_support_v2_20261028/playback_model_9999/transition.csv')),('side_middle_reset',Path('/home/hc/robotis/cyclo_mjlab/setup/taeguek04_sidekick_stability_v1_20261028/audit_20261028/middle_reset.csv')),('side_middle_transition',Path('/home/hc/robotis/cyclo_mjlab/setup/taeguek04_sidekick_stability_v1_20261028/audit_20261028/middle_transition.csv')),('side_final_reset',Path('/home/hc/robotis/cyclo_mjlab/setup/taeguek04_sidekick_stability_v1_20261028/audit_20261028/final_reset.csv')),('side_final_transition',Path('/home/hc/robotis/cyclo_mjlab/setup/taeguek04_sidekick_stability_v1_20261028/audit_20261028/final_transition.csv')),('v1_reset',base/'rollouts/t4_support_29999_reset.csv'),('v1_transition',base/'rollouts/t4_support_29999_transition.csv'),('v2_reset',base/'rollouts/t4_support_v2_9999_reset.csv'),('v2_transition',base/'rollouts/t4_support_v2_9999_transition.csv'),('deployed_reset',Path('/home/hc/robotis/cyclo_mjlab/setup/taeguek04_balanced_v1_20261027/sim2sim_model_14500/compare_reset.csv')),('deployed_transition',Path('/home/hc/robotis/cyclo_mjlab/setup/taeguek04_balanced_v1_20261027/sim2sim_model_14500/compare_transition.csv')),('middle_reset',out/'middle_reset.csv'),('middle_transition',out/'middle_transition.csv'),('source_reset',out/'source_reset.csv'),('source_transition',out/'source_transition.csv'),('final_reset',out/'final_reset.csv'),('final_transition',out/'final_transition.csv')]:
 d=np.genfromtxt(p,delimiter=',',names=True); t=d['time']; idx=np.minimum(np.rint(t/.02).astype(int),len(rt)-1);force=d['l_fz']+d['r_fz']; events=[]; drift=[]; excess=[]; masktot=0; ks=[]
 for f,side in enumerate(['l','r']):
  speed=np.linalg.norm(np.gradient(r['pos'][:,f,:2],.02,axis=0),axis=1); yaw=abs(np.gradient(r['yaw'][:,f],.02)); q=(speed<.1)&(yaw<1)&(r['bottom'][:,f]<.012);quiet=np.zeros(len(rt),bool)
  for a,b in spans(q):
   if b-a>=15:quiet[a+5:b-5]=True
  quiet=quiet[idx]; kick=np.zeros(len(t),bool)
  for a,b,ff in wins:
   if ff==f:kick|=(t>=a)&(t<b)
  mask=quiet|kick; masktot+=mask.sum(); maskground=quiet&(d[side+'_fz']>5); excess.extend(np.maximum(np.hypot(d[side+'_vx'],d[side+'_vy'])[maskground]-speed[idx][maskground],0).tolist())
  for a,b in spans((d[side+'_bottom']>.003)&(d[side+'_fz']<=5)):
   if b-a>=4 and any(y-x>=4 for x,y in spans(mask[a:b])):
    j=a+np.argmax(d[side+'_bottom'][a:b]);events.append({'side':side,'start':round(float(t[a]),3),'end':round(float(t[b-1]+dt),3),'height_mm':round(float(d[side+'_bottom'][j]*1000),2),'ref_mm':round(float(r['bottom'][idx[j],f]*1000),2),'kick':bool(kick[a:b].any())})
  for a,b in spans(quiet):
   if (b-a)*dt<.3:continue
   v=np.column_stack((d[side+'_x'][a:b],d[side+'_y'][a:b])); ref=r['pos'][idx[a:b],f,:2];dist=np.linalg.norm(v-v[0],axis=1);rd=np.linalg.norm(ref-ref[0],axis=1);drift.append({'side':side,'start':round(float(t[a]),2),'end':round(float(t[b-1]+dt),2),'max_cm':round(float(dist.max()*100),2),'ref_cm':round(float(rd.max()*100),2)})
 for a,b,f in wins:
  m=(t>=a)&(t<b+.4); j=np.flatnonzero(m)[np.argmax(force[m])];s=['l','r'][f];low=(d[s+'_fz']<=5)&(d[s+'_bottom']>.003)&(t>=a)&(t<b)
  ks.append({'start':a,'end':b,'support':s,'peak_N':round(float(force[j])),'peak_t':round(float(t[j]),3),'max_hop_mm':round(float(d[s+'_bottom'][low].max()*1000),2) if low.any() else 0})
 j=int(np.argmax(force)); q=np.column_stack([d[f'q{j}'] for j in r['joint_qadr']]); goal=np.column_stack([np.interp(t+.005,rt,r['joint_pos'][:,j]) for j in range(23)]); rmse=float(np.sqrt(np.mean((q-goal)**2))); arm=float(np.sqrt(np.mean((q[:,13:]-goal[:,13:])**2))); raw=np.loadtxt(out/'source/params/k1_taeguek04_collision_only_50hz_converted.csv',delimiter=','); xyref=np.column_stack([np.interp(t+.005,rt,raw[:,i]) for i in range(2)]); xy=np.column_stack([d['q0'],d['q1']]); xyerr=float(np.sqrt(np.mean(np.sum((xy-xyref)**2,axis=1)))); peak_err=[]
 for a,b,f in wins:
  mm=(t>=a)&(t<b); other=['l','r'][1-f]; targetmask=(rt>=a)&(rt<b);peak_err.append(float(abs(d[other+'_bottom'][mm].max()-r['bottom'][targetmask,1-f].max())))
 results[name]={'joint_rmse_rad':rmse,'arm_rmse_rad':arm,'pelvis_xy_rmse_m':xyerr,'kick_peak_error_mean_m':float(np.mean(peak_err)),'fall_min_z_m':float(d['root_z'].min()),'peak_N':round(float(force[j])),'peak_t':round(float(t[j]),3),'peak_BW':round(float(force[j]/bw),2),'excess3BW_Ns':round(float(np.maximum(force-3*bw,0).sum()*dt),2),'quiet_excess_mm_s':round(float(np.mean(excess)*1000),2),'events':sorted(events,key=lambda v:v['start']),'worst_drift':sorted(drift,key=lambda v:v['max_cm']-v['ref_cm'],reverse=True)[:5],'kick':ks}
 print(name,json.dumps(results[name],ensure_ascii=False))
(Path('/home/hc/robotis/cyclo_mjlab/setup/taeguek04_sidekick_stability_v1_20261028/audit_20261028')/'contact_assessment.json').write_text(json.dumps({'bodyweight_N':bw,'kick_windows':wins,'results':results},indent=2))
