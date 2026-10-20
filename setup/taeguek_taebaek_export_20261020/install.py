from pathlib import Path
import json,shutil,hashlib
O=Path(__file__).resolve().parent
T=Path('/home/hc/robotis_mujoco/policy/Mimic/taeguek_1020_compse')
assert not T.exists(),f'Destination already exists: {T}'
S=T.with_name('.taeguek_1020_compse_staging')
assert not S.exists()
S.mkdir()
info={}
for name in ('taeguek_07','taebaek'):
 for sub in ('exported','params'):
  shutil.copytree(O/name/sub,S/name/sub)
 m=json.loads((S/name/'exported/manifest.json').read_text())
 # The archived training checkpoint remains in the training workspace, not this deployment.
 m['files']={k:v for k,v in m['files'].items() if not k.startswith('checkpoint/')}
 m['installed_policy_path']=str(T/name)
 (S/name/'exported/manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 for k,v in m['files'].items():
  assert hashlib.sha256((S/name/k).read_bytes()).hexdigest()==v['sha256'],k
 info[name]={k:m[k] for k in ('source_run','checkpoint','completed_updates','checkpoint_sha256','source_motion_npz','motion_csv','motion_frames','motion_fps','validation')}
(S/'import_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
(S/'README.md').write_text('''# 2026-10-20 완료 정책\n\n- taeguek_07: 팔 보정 모션, model_9999.pt, 10,000회 완료.\n- taebaek: 접촉 보정 모션, model_49999.pt, 50,000회 완료.\n\n각 모션 폴더의 exported에는 ONNX 및 TorchScript 추론 모델, params에는 원래 학습 설정과 해당 모션의 최신 CSV가 있습니다. robotis_mujoco에서 policy_path를 원하는 모션 폴더로 지정하십시오. src 또는 기본 정책 선택 설정은 수정하지 않았습니다.\n\nCSV는 학습 NPZ와 프레임 수, 관절 순서, 관절 각도, 루트 위치/회전을 비교하여 검증했습니다. env.yaml의 trajectory_file은 원래 학습 경로를 보존하며, sim2sim은 params의 CSV를 사용합니다. exported/policy.pt는 추론용 TorchScript이며 학습 재개 체크포인트가 아닙니다.\n''')
S.rename(T)
print(json.dumps({'installed':str(T),'policies':info},indent=2))
