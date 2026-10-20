from pathlib import Path
import json,hashlib,shutil,datetime,os
from zoneinfo import ZoneInfo
O=Path(__file__).resolve().parent;B=O/'bundle_8250'
root=Path('/home/hc/robotis_mujoco/policy/Mimic/taeguek_compose_1019');target=root/'taeguek_07'
now=datetime.datetime.now(ZoneInfo('Asia/Seoul'));tag=now.strftime('%Y%m%d_%H%M%S')
backup=root/'backup'/f'taeguek_07_before_arm_v1_{tag}'
stage=root/f'.taeguek_07_install_{tag}'
def hashes(folder):
 return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob('*')) if p.is_file()}
manifest=json.loads((B/'exported/manifest.json').read_text());smoke=json.loads((O/'smoke.json').read_text())
assert manifest['checkpoint_iteration']==8250 and manifest['completed_updates']==8251
assert smoke['all_finite'] and smoke['policy_active'] and smoke['minimum_root_height_m']>.35
original=hashes(target);backup5=hashes(root/'backup/taeguek_05');chapter3=hashes(root/'taeguek_03')
assert not stage.exists() and not backup.exists()
stage.mkdir()
for name in ('exported','params'):shutil.copytree(B/name,stage/name)
expected=hashes(stage)
for name,h in expected.items():assert hashlib.sha256((B/name).read_bytes()).hexdigest()==h
meta=O/'compose_metadata_before';meta.mkdir()
for name in ('README.md','import_manifest.json'):shutil.copy2(root/name,meta/name)
os.rename(target,backup)
try:os.rename(stage,target)
except BaseException:
 os.rename(backup,target)
 raise
assert hashes(target)==expected and hashes(backup)==original
assert hashes(root/'backup/taeguek_05')==backup5 and hashes(root/'taeguek_03')==chapter3
im=json.loads((root/'import_manifest.json').read_text());old=next(x for x in im['included'] if x['chapter']==7)
im.setdefault('replacements',[]).append({'chapter':7,'replaced_at':now.isoformat(),'previous':old.copy(),'backup':str(backup.relative_to(root))})
old.update(folder='taeguek_07',source_host='local hc',checkpoint=manifest['checkpoint'],iteration=8250,completed_updates=8251,
 checkpoint_sha256=manifest['checkpoint_sha256'],csv=manifest['motion_csv'].split('/')[-1],frames=manifest['motion_frames'],fps=manifest['motion_fps'],
 sha256_verified=True,sim2sim_load_check_steps=9275,status='intermediate',quality_status='ArmSmoothV1 intermediate 8250: full-motion health check passed in one default-pose condition; previous comparative evaluation was model_7750, not this model.')
for entry in im['included']:
 if entry['chapter']==5 and not (root/'taeguek_05').exists() and (root/'backup/taeguek_05').is_dir():
  entry['folder']='backup/taeguek_05';entry['status']='user_moved_to_backup'
im['updated_at']=now.isoformat()
readme=f'''# 태극 보완 학습 정책 · 2026-10-19

- `taeguek_03`: TargetedV3 최종 model_9999. 기존 파일 유지.
- `taeguek_07`: 로컬 ArmSmoothV1 최신 저장본 **model_8250.pt**, 8,251 / 10,000 업데이트 완료 시점. 학습 중간본이며 학습은 계속 진행 중이다.
- `backup/taeguek_05`: 사용자가 이동한 SmoothV3 최종본. 이번 갱신에서 변경하지 않았다.
- `backup/{backup.name}`: 이번 교체 직전의 7장 TargetedV3 정책.
- 태극 4·8장은 포함하지 않는다.

각 활성 장의 `exported`에는 ONNX/TorchScript와 출처 manifest, `params`에는
학습 설정 3종과 해당 학습의 50Hz 모션 CSV가 있다.

7장은 원본 actor와 ONNX/TorchScript 출력 비교, CSV–NPZ 일치, 현재 sim2sim 실행 코드로
기본 자세에서 9,275 물리 스텝(46.375초)의 전체 모션을 검증했다.
정책 활성/유한값을 유지했으며 루트 높이 35cm 미만 낙상은 없었다.
이 검증은 단일 조건의 동작 확인이며, 최신 8250의 세부 개선율 평가는 아니다.
앞선 비교 보고서는 7750 중간본에 대한 결과이므로 그 개선율을 8250에 그대로 적용하지 않는다.

갱신 시각: {now.isoformat()}
출처와 검증 정보: `import_manifest.json`, `taeguek_07/exported/manifest.json`.
'''
for name,data in [('import_manifest.json',json.dumps(im,indent=2)+'\n'),('README.md',readme)]:
 temp=root/(name+'.new');temp.write_text(data);os.replace(temp,root/name)
record=dict(installed_at=now.isoformat(),target=str(target),backup=str(backup),checkpoint=manifest['checkpoint'],checkpoint_sha256=manifest['checkpoint_sha256'],completed_updates=8251,
 installed_files=expected,previous_policy_files=original,unchanged_user_backup5=True,unchanged_chapter3=True,validation=manifest['validation'],sim2sim_health=smoke)
(O/'installation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['installed_at','target','backup','completed_updates','unchanged_user_backup5','unchanged_chapter3']},indent=2))
