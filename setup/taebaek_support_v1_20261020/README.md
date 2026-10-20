# 태백 지지발 안정화 V1 — 실행 준비

기존 팔 보정 태백의 잔발 평가 결과를 반영한 별도 fine-tuning 실험입니다.
본 학습은 사용자가 명령어로 실행합니다.

## 실행

Desktop 호스트 터미널:

```bash
docker exec -it --user 1000:1000 -w /workspace/cyclo_mjlab cyclo_mjlab \
  python -u setup/taebaek_support_v1_20261020/launch.py 0 --num-envs 4096
```

이미 로컬 컨테이너 안이라면:

```bash
cd /workspace/cyclo_mjlab
python -u setup/taebaek_support_v1_20261020/launch.py 0 --num-envs 4096
```

준비 상태만 확인:

```bash
python -u setup/taebaek_support_v1_20261020/launch.py 0 --num-envs 4096 --dry-run
```

GPU가 다른 학습/뷰어에 사용 중이면 실행을 거절합니다. 기존 프로세스를 자동 종료하지 않습니다.
동일 명령어로 재실행하면 이 실험의 호환 체크포인트만 골라 남은 업데이트를 이어갑니다.
smoke 검증 실행은 재개 후보에서 제외합니다. 잠금은 중복 실행을 방지합니다.
Ctrl+C로 중단 후 다시 실행하면 마지막 저장 이후 작업만 반복됩니다.

## 입력과 학습 설정

- 평가를 완료한 `model_17000.pt`를 `source/`에 고정 복사했습니다.
- 원래 실행: `2026-10-20_02-38-55_taebaek_arm_refined_manual`.
- source SHA256: `6653bc0311b6de47fcec59232af125d781a991efb4b44eed86e09608112d3564`.
- 팔 보정 CSV/NPZ도 `motion/`에 고정 복사했습니다. 2360프레임, 50Hz, 47.18초.
- actor/critic만 이어받고 optimizer와 보완 학습 iteration은 새로 시작합니다.
- 환경 4096개, 추가 10000 PPO 업데이트, GPU 0.
- 학습률 5e-5 fixed, entropy 0.001, 250회마다 체크포인트 저장.
- 기존 Mimic, pelvis XY, joint_acc, torque, action_rate, collision 보상은 보존합니다.
- episode limit 30초→49.2초. 20%는 첫 프레임에서 시작하고 나머지는 적응형 시작 샘플링을 사용합니다.
- 로그: `logs/rsl_rl/k1_mimic/*_taebaek_support_v1_finetune_*`.
- 새 보상과 의존 코드 revision, 원본 해시, num_envs, 완료 업데이트/환경 스텝을 검증해 재개합니다.

## 추가 보상

`taebaek_support_stability`, 외부 weight=-1.
기준 모션에서 예상 지지 구간을 계산합니다. 낮은 발이라도 빠르게 이동하는 보행 구간은 제외하고,
반대 발이 높이 올라간 발차기에서는 지지발을 포함합니다. 비행/차는 발에는 지지 패널티가 없습니다.

| 항목 | 내부 가중치 | 의미 |
|---|---:|---|
| 접촉 상실 | 0.9 | 예상 지지 구간의 실제 접촉 유지 |
| 지속적인 비접촉 | 0.3 | 길게 떠 있는 상태 억제 |
| 반복 liftoff | 0.08 | 물리 서브스텝 접촉 이탈 사건; 시간 간격 보정 |
| 접촉점 미끄럼 | 0.45 | 기준 접촉점 속도+0.05m/s 초과 |
| 지지 구간 누적 이탈 | 0.5 | 실제 발바닥 중심 이동−기준 이동, 허용 8mm/척도30mm |
| 앞뒤·좌우 흔들림 | 0.25 | 기준 대비 signed XY 각속도 오차, 허용0.2rad/s |
| 반복 각속도 변화 | 0.06 | 각속도 오차 변화율, 허용5rad/s²/척도20rad/s² |
| 지지발 추가 높이 | 0.8 | 기준 높이+3mm 초과/척도8mm |
| 불필요한 상승 | 0.2 | 기준 수직속도+0.05m/s 초과/척도0.25m/s |

새 패널티는 20%에서 시작해 24000 환경 스텝(1000 PPO 업데이트)에 100%까지 증가합니다.
한 발 지지는 1.5배 가중, reset 후150ms는 유예합니다.
누적 이탈 기준은 **기준 지지 구간 진입 때만** 새로 잡습니다. 실제 접촉 이탈로는 초기화하지 않아
재디딤으로 누적 이탈 비용을 지우지 못합니다.
기하/흔들림 비용은 실제 접촉 여부와 독립적으로 적용해 발을 띄워 회피하지 못하게 합니다.
큰 실패에도 비용이 계속 증가하는 smooth robust cost를 사용합니다.
기준 이동/각속도를 빼므로 정상 피벗·움직임을 정지시키는 보상이 아닙니다.

## 검증과 평가 범위

- test_rewards.py: 기준 동작/비행 허용, 큰 이탈 구별, 높이 허용, 흔들림, phase/reset,
  기존 보상 보존 검증.
- smoke.log: 128 envs에서 고정 원본 가중치를 strict 로드하고 3 PPO 업데이트.
- smoke_resume.log: 그 체크포인트에서 optimizer/counter를 복원해 2회 추가 업데이트.
- 본 학습과 별도이며 개선된 정책 성능을 입증하는 테스트는 아닙니다.
- 2500/5000/10000회 시점에서 original sim2sim 전체 모션을 비교해야 합니다.
  접촉 이탈 수뿐 아니라 발바닥 수평 이동·raw roll/pitch·팔 추종·발차기 높이를 함께 평가합니다.
- 33.5초 역방향 회전 보상은 이번 발 안정화 범위에 포함하지 않았습니다.

기존 실행 설정, robotis_mujoco 정책 및 src는 변경하지 않았습니다.
