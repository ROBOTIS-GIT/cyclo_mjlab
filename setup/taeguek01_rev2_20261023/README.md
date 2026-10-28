# K1 Rev.2 태극 1장 신규 학습

Desktop `/home/hc/robotis/cyclo_mjlab`, branch `feature-umr-taeguendo_1_test`.
`third_party/ai_sapiens` 서브모듈도 같은 이름의 로컬 브랜치로 분기했다.
본학습은 사용자가 직접 실행한다. 자동 재학습을 시작하지 않는다.

## 컨테이너 내부 실행

```bash
cd /workspace/cyclo_mjlab
python -u scripts/reinforcement_learning/train.py Cyclo-Mimic-K1-Rev2-Taeguek01 \
  --gpu-ids '[0]' \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 30000 \
  --agent.logger tensorboard \
  --agent.resume False \
  --agent.upload-model False
```

GPU0, 4096환경, 30000 PPO 업데이트를 처음부터 학습한다. 기존 Rev.1 정책을 불러오지 않는다.
태극 1장 하나만 실행하며 다른 품새로 넘어가는 커리큘럼은 없다. tmux 없이 현재 터미널에서 실행한다.
출력: `logs/rsl_rl/k1_rev2_taeguek01/<실행시간>_rev2_taeguek01_scratch/`.
학습시작 시 `params/sim2real.yaml`, `env.yaml`, `agent.yaml`이 생성되고 500업데이트마다 체크포인트가 저장된다.
중단은 Ctrl+C. 재개할 때는 같은 태스크 명령에 `--checkpoint /workspace/cyclo_mjlab/logs/rsl_rl/k1_rev2_taeguek01/<run>/model_N.pt`를 추가한다.
일반 train.py의 `--agent.max-iterations`는 해당 실행에서 추가로 수행할 횟수이므로 재개 시 남은 횟수를 지정한다.

## 모델과 모션

- 출처: `/home/hc/Desktop/ai_sapiens/ai_sapiens_description`의 `urdf/k1_rev2`, `meshes/k1_rev2`, `mujoco/k1_rev2`.
- 이 세 폴더를 `third_party/ai_sapiens/ai_sapiens_description`에 복사했다. 기존 Rev.1 자산은 유지된다.
- 학습 엔진은 URDF와 일치하는 Rev.2 `mujoco/k1_rev2/k1.xml`을 사용한다. URDF 대비 링크별 질량, CoM, 전체 관성 텐서, joint transform/axis/limit를 수치 검증했다.
- Rev.2 총 질량37.3424451kg, 링크25개, 제어관절23개, collision32개. 원본 URDF에서 collision 없는9개 링크에 임의 형상을 추가하지 않았다.
- 모터 종류·토크 한계·armature 및 발 충돌구9개×2는 기존과 동일하므로 PD/action scale/접촉 설정은 검증된 Rev.1 설정을 공유한다.
- source: `source/assets/motions/K1_rev1/taeguek_01/k1_taeguek01_collision_only_50hz_converted.csv`.
- Rev.2용: `source/assets/motions/K1_rev2/taeguek_01/k1_rev2_taeguek01_50hz.{npz,csv}`.
- 1582프레임,31.62초, 정책50Hz/물리200Hz. 에피소드 제한33.62초로 전체 모션을 수용한다.
- 원본 관절각·루트·타이밍을 유지하고 Rev.2로 몸체 FK/선속도/각속도를 다시 계산했다. 사람 모션을 새로 UMR 리타게팅한 결과는 아니다.
- Rev.2 무릎의12.5mm 좌우 위치 변경과 발목의 반대 오프셋은 발 위치에서 상쇄된다. 발바닥 최소높이0.205/0.276mm, 매 프레임 낮은 발의 최대높이10.784mm, 관절 한계 초과없음.
- 원본 모션에 self-contact 1.57mm 한 표본이 남아 있다. 자동으로 관절각이나 collision을 변경하지 않았다.
- 보상은 기존 태극1장 추종/경로/기본접촉 설정을 유지했다. 6장 V9 전환 전용 보상을 이식하지 않았다.

## 검증 및 재생

`prepare.py`는 위 URDF/MJCF 일치·관절한계·발 접촉높이를 검증하고 Rev.2 NPZ와 CSV를 재생성한다.
`inputs.json`은 출처와 자산·모션 해시, FK round-trip 오차를 기록한다.
`validate.py`는4096환경에서2 PPO업데이트, 유한가중치·optimizer 진행·저장·JIT/ONNX export·64추론 스텝을 확인한다.
검증 결과는 `validation.json`, 로그는 `validation.log`. smoke run은 본학습과 다른 폴더 `k1_rev2_taeguek01_smoke`에 저장한다.

```bash
python scripts/reinforcement_learning/play.py Cyclo-Mimic-K1-Rev2-Taeguek01 \
  --checkpoint /workspace/cyclo_mjlab/logs/rsl_rl/k1_rev2_taeguek01/<run>/model_N.pt
```

Rev.2 정책은 Rev.2 모델에서 재생해야 한다. 기존 robotis_mujoco Rev.1 배포 설정으로 복사하지 않는다.
이번 변경은 로컬 작업 상태이며, 원격에서 사용하려면 상위 저장소의 새 파일뿐 아니라 ai_sapiens 서브모듈의 rev2 파일도 함께 전달해야 한다.
