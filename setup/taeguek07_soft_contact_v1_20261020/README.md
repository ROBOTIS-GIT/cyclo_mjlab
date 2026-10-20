# 태극 7장 잔발·착지 충격 보완 V1

사용자가 실물에서 사용했다고 확인한 `taeguek_1020_compse/taeguek_07`의 팔 보정 완료 정책에서 이어가는 별도 실험입니다. 본 학습은 자동 시작하지 않습니다.

## 실행

현재 태백 학습이 GPU 0을 사용 중이면 아래 실행은 GPU 점유를 감지해 거절합니다. 다른 학습을 자동 종료하지 않습니다. GPU가 비었을 때 실행하십시오.

로컬 컨테이너 안:

```bash
cd /workspace/cyclo_mjlab
python -u setup/taeguek07_soft_contact_v1_20261020/launch.py 0 --num-envs 4096
```

Desktop 호스트:

```bash
docker exec -it --user 1000:1000 -w /workspace/cyclo_mjlab cyclo_mjlab \
  python -u setup/taeguek07_soft_contact_v1_20261020/launch.py 0 --num-envs 4096
```

준비 상태 확인은 같은 명령에 `--dry-run`을 붙입니다. 중단 후 같은 명령을 실행하면 이 실험의 마지막 호환 체크포인트에서 이어서 총10000회까지만 진행합니다. smoke 실행은 선택하지 않습니다. 환경 수와 보상 revision이 다른 체크포인트는 거절합니다.

## 원본과 설정

- source: `2026-10-19_11-17-37_taeguek07_arm_smooth_v1_finetune_20261019_111729/model_9999.pt`
- SHA256: `2228c9f5e3f6e3bd8849a46894066347700f208e6a0c9426016ad88d72cfc7cc`
- 해당 checkpoint, 팔 보정 CSV/NPZ, 기존 entry bank와 보상 코드를 이 폴더에 고정 복사했습니다.
- motion: `k1_taeguek07_arm_corrected_v1_50hz_converted`, 2320frames,50Hz,46.38초.
- actor/critic만 warm start, 새 optimizer 및 보완 iteration0. 재개는 optimizer/counter까지 복원합니다.
- num_envs4096, 추가10000 PPO updates, lr5e-5 fixed, entropy0.001,250회마다저장.
- 기존 Mimic/XY, 지지발 접촉·발 높이·flight-height, 팔 자세·속도·가속도 안정화 보상 유지.
- 기존 팔 안정화와 지지 보상은 성숙한 강도로 유지. 새 항목만20%→100%로1000update동안강화.
- 기존 전체 모션 길이, 시작 상태/집중 구간 샘플링을 유지합니다. physics/PD/friction/관측/행동/CSV는 변경하지 않습니다.

## 새 보상

### support_micro_motion (weight=-1)

기준 지지 phase에 한정하고 reset150ms유예. 기존 접촉 패널티를 이 항목에 중복 추가하지 않습니다.

- 접촉점 초과 미끄럼: 추가0.30 (기존0.15와합계최대0.45).
- 지지 구간 누적 이탈:0.35. 기준 이동을 뺀 발바닥 중심 수평 이동;8mm 허용/30mm척도.
- 반복 기울어짐:0.25. 기준 대비XY각속도 오차;0.2rad/s허용.
- 각속도 오차의 변화:0.04. 5rad/s²허용/20rad/s²척도.
- 원래 높이/상승/비행 보상은 그대로 두고 이 항목에는 중복하지 않습니다.
- 실제 접촉 이탈로 누적 위치 기준을 초기화하지 않습니다. 들림으로 보상을 회피하지 못합니다.

### soft_ground_contact (weight=-1)

명목 모델 질량35.706kg, 중량350.276N으로 정규화합니다. 아래는 학습용 soft threshold이며 하드 제한이나 실제 하드웨어 안전 한계가 아닙니다.

- peak_weight=0.5: 한 발2.2배체중(약771N), 양발합3배체중(약1051N)초과의robust cost.
- loading_rate_weight=0.08: 물리 substep의 급격한 힘 증가를 penalize.80배체중/s부터시작,120배체중/s척도. 발차기를 위한 하중 감소는 이 항목에서 penalize하지 않습니다.
- approach_weight=0.12: 실제 발바닥이 지면4cm이내, 기준이6cm이내일 때만, 낮은 접촉점의 하강속도0.6m/s초과를 penalize. 공중의 빠른 발차기는 제외합니다.
- reset150ms유예. 실제 착지에는 적용하여 더 부드러운 접촉을 유도합니다. 모든 점프/착지를 면제하지는 않습니다.

기존 original sim2sim 감사에서는 양발합최고접촉력 약1880–2281N이 관찰됐습니다.27.945–27.950초,29.525초 등의 순간 peak가 포함됩니다. 이는 MuJoCo 추정치이며 실물 force 측정치가 아닙니다.

## 검증 및 성공 조건

- test_rewards.py 8개: 정적체중허용, peak단조증가, 하중감소허용, dt정규화, 공중발차기/느린착지허용, 기준동작허용, 기존보상유지, 원본해시.
- 128env3updates smoke 및 이어서2updates resume 검증. 결과 validation.json.
- full training이나 실제 개선 입증을 대신하지 않습니다.
- 2500/5000/10000회에서 original sim2sim 전체+연속 진입 조건으로 비교: 접촉력peak/짧은구간impulse, 발 미끄럼/반복기울기, 팔추종, 발차기높이, 완주.
- 실물에서는 동일동작속도/PD/바닥조건에서 충격 측정값(가능하면접촉력/IMU/전류)을 함께 비교해야 합니다. 시뮬레이션 패널티 감소가 실물 충격 감소를 보장하지 않습니다.

기존 태백 보완 학습, 기존7장정책, robotis_mujoco/src는 수정하지 않았습니다.
