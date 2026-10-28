# 태극 5장 SoftContact V1 — ArmTrackV4의 착지·접촉 보완

현재 Backup에 배치된 왼팔 보정 ArmTrackV4 완료본에서 추가 학습한다. 목표는 35.08초/18.01초의 높은 수직 GRF, 35.32–35.49초 재이륙, 참조상 지지 중 양발 하중 손실을 줄이는 것이다. 실제 개선 여부는 완료 후 전체 재생으로 판정한다. 이 설정 준비만으로 품질 개선을 선언하지 않는다.

## 실행

Desktop 또는 Verda의 `cyclo_mjlab` 컨테이너 안에서:

```bash
cd /workspace/cyclo_mjlab
python -u setup/taeguek05_soft_contact_v1_20261021/launch.py 0 --num-envs 4096 --iter 10000
```

같은 명령으로 재개하면 이 실험의 저장본을 찾아 optimizer/완료 카운터를 복원하고 총 10,000회에서 남은 횟수만 학습한다. `--iter 20000`으로 총 목표를 늘릴 수 있다. 원본 정책의 10,000회와 별도로 이번 보완 횟수를 센다. 처음에는 actor/critic을 정확히 로드하고 새 optimizer로 시작한다. smoke 체크포인트는 본 학습 재개 대상에서 제외된다. 환경 수·revision·입력 해시·optimizer·step counter가 다르면 중단한다.

계획 확인만:

```bash
python -u setup/taeguek05_soft_contact_v1_20261021/launch.py 0 --num-envs 4096 --iter 10000 --dry-run
```

GPU 0이 다른 학습에 사용 중이면 실행기가 시작을 거절한다. Desktop에서는 태극 8장 SupportHoldV2와 충돌하지 않도록 완료 후 실행한다. Verda의 GPU는 별도 장비이므로 Desktop 학습과 독립적으로 사용할 수 있다. 다른 학습을 자동 종료하거나 컨테이너를 재시작하지 않는다. 캐시와 lock은 `/tmp`의 UID별 경로를 사용한다.

## Verda에서 직접 시작하기

로컬 PC에서:

```bash
chmod 400 ~/Downloads/verda_server_key
ssh -i ~/Downloads/verda_server_key root@95.133.254.117
```

서버 호스트에서 tmux 세션을 만들고 컨테이너에 들어간다:

```bash
tmux new-session -A -s taeguek05-soft-contact
docker exec -it -u cyclo -w /workspace/cyclo_mjlab cyclo_mjlab bash
```

컨테이너에서 실행한다:

```bash
python -u setup/taeguek05_soft_contact_v1_20261021/launch.py 0 --num-envs 4096 --iter 10000
```

`Ctrl+B`를 누른 뒤 `D`로 tmux에서 빠져나오면 SSH 종료 후에도 학습이 유지된다. 다시 접속하면 `tmux attach -t taeguek05-soft-contact`로 확인한다. 이미 학습 중인 세션에 붙은 경우 실행 명령을 중복 입력하지 않는다.

호스트 저장 경로는 `/opt/cyclo/cyclo_mjlab`, 컨테이너 경로는 `/workspace/cyclo_mjlab`이다. 결과는 `logs/rsl_rl/k1_mimic/*_taeguek05_soft_contact_v1_finetune_*/`에 저장된다.

## 고정한 출발점

- `source/model_9999.pt`: 왼팔 보정 ArmTrackV4 10,000회 완료본. SHA256 `2b73bd44508ee6f7734d93057fafa7d77d5ca8b7182213f719f314ada5988c75`.
- `motion/k1_taeguek05_left_arm_refined_50hz_converted.npz/csv`: 기존 왼팔 보정 참조의 고정 복사본. 50Hz, 2017프레임, 40.32초.
- 양팔 정지 자세 V3의 미완료 노트북 체크포인트는 사용하지 않는다.
- 정책 50Hz, 물리 200Hz, PD·관측·행동·접촉 물성·기존 랜덤화 유지. 기존 sim2real.yaml과 새 환경에서 생성한 설정 전체의 일치를 검사한다.
- 기존 SmoothV3/SupportV2의 지지·발차기·시간적 평활화 보상과 ArmTrackV4의 손·팔꿈치·팔 자세·정지 보상을 유지한다. 이미 학습된 보상은 시작부터 최종 강도이며 새 비용만 점진 적용한다.

## 보완 목표

| 항목 | 설계 |
|---|---|
| 순간 GRF | 물리 substep마다 발별 2.2BW·양발 합 3BW 초과를 검출. 계수 0.40 |
| 하중 증가율 | 양의 힘 증가만 사용. 80BW/s 초과를 120BW/s로 정규화, 계수 0.04. 발을 들기 위한 하중 감소는 비용 없음 |
| 지면 접근 | 실제 발바닥이 6cm 이내에서 아래로 0.6m/s보다 빠르게 접근할 때, 계수 0.12. 참조상 200ms 내 착지 예정 또는 이미 낮은 접촉 참조인 경우만 적용 |
| 착지 후 안정화 | 참조상 4cm 이상의 들림 뒤 착지한 후 최대 450ms, 낮은 접촉이 유지되고 저점 이동속도 <0.6m/s일 때 적용. 접촉 부재 0.20, 이탈 이벤트 0.01/policy_dt, 추가 들림 0.05 |
| 양발 하중 손실 | 보수적인 참조 지지 구간에 한발이라도 있어야 하는데 양발 합 하중이 0.2BW 미만이면 계수 0.20. 정상 좌우 체중 이동과 참조상 공중 구간은 제외 |

BW는 XML 로봇 질량×중력이다. 이 문턱은 보상 초기 설정이며 실물 안전 한계가 아니다. 새 비용은 `2*log1p(raw/2)`로 완화하되 큰 충격을 동일 값으로 하드 포화하지 않는다. 가중치는 검증할 초기값이며 최적값이라는 주장이 아니다.

참조 landing label은 시각 하드코딩 없이 전체 모션에서 자동 추출한다. 실제 발을 빠르게 움직이거나 접촉을 끊는 것으로 reference label을 끌 수 없다. 35.08초처럼 참조발이 아직 높은 조기 착지에도 다음 착지가 200ms 이내이면 지면 접근 비용이 작동한다. 빠른 낮은 이동은 별도의 발 고정 비용을 추가하지 않고, 지면을 향한 과도한 하강과 큰 충격만 다룬다. 다음 의도된 이륙에서는 착지 후 고정 목적을 해제한다.

reset/참조 프레임 점프 뒤 150ms는 새 항목을 제외한다. 접촉 상태는 새 substep만 시간순으로 소비하며 20N/5N과 10ms debounce를 쓴다. 접촉 부재는 높이가 작아도 계산하므로 발이 2mm밖에 안 들린 무하중 상황도 놓치지 않는다.

## 학습 계획·검증

- 기본 10,000회, 4096환경, learning rate `2e-5` 고정, entropy 0.001, 에피소드 42.34초.
- 새 비용만 5,000 업데이트에 걸쳐 10→100%로 강화한다. 기존 팔·발차기 비용은 약화하지 않는다.
- 500회 저장은 복구와 큰 학습 붕괴 감지용이다. 500회 sim2sim으로 성공/실패를 판단하는 자동평가가 아니다.
- 500회 이후 팔·손·관절·episode length 기준을 잡고 1000회 이후 큰 악화가 두 저장 시점 연속 발생하면 체크포인트 저장 후 멈춘다. 이 guard는 품질 합격 판정이나 미세한 악화 감지 수단이 아니다.
- `test_rewards.py`, `test_launch.py`: substep 피크, 하중 감소 예외, BW 정규화, 큰 비용 구분, 정상 스윙·체중 이동 예외, 접촉 누적·reset, resume/총 목표 계수 검증.
- `validate.py`: CPU에서 원본 actor/critic 정확한 로드, 짧은 PPO 학습·새 환경 재개, optimizer 연속성, 완전 ramp 상태의 보상 유한성, sim2real 설정 일치, 본 학습과 smoke 분리 검사. GPU의 4096환경 학습 성능·메모리는 이번 CPU 검사 범위가 아니다.
- `coverage.json`: 실제 18초·35초 충격 구간, 35초 재이륙, 30.8초 지지 구간에 목적이 적용되는지 기록.

## 완료 후 평가

기준 평가: `/home/hc/Documents/ChatGPT/태권도 학습 .. 로부월드/work/taeguek05_armtrack_v4_contact_audit_20261021/REPORT.md`.

동일한 50Hz·original 발바닥·두 시작 조건에서 전체 40.32초를 비교한다. 35.08초 1.40kN·18.01초 1.30–1.37kN의 피크, 3BW 초과 시간·힘 적분, 착지 후 오른발 재이륙 6–10mm/저하중 155–175ms, 30.83초 양발 무하중 50ms를 함께 본다. 다른 구간으로 충격을 옮겼는지 확인하고 팔·손 추종, 발차기 높이, 경로와 낙상이 악화하지 않아야 개선본으로 선택한다.

4.95초의 낮은 참조 이동처럼 원본 자체가 미끄러지듯 이동하는 구간은 보상만으로 고정하지 않는다. 학습 후에도 남는 구간은 그때 참조 착지 시각·하강 속도·지지 주석을 별도 검토한다. 현재 배치된 정책과 참조 데이터 원본은 그대로 보존했다.

검증 완료: 동작/재개/CLI 테스트 17개 통과. CPU 8환경에서 3회 학습 후 새 환경에서 1회 재개, optimizer step 60→80 확인. 완전 ramp에서 128 policy step의 유한성 검사와 원본 actor/critic 정확한 로드, sim2real 설정 전체 일치 확인. `coverage.json`의 네 문제 구간 적용 비율은 모두 100%다. `validation_cpu.json`에 최종 revision과 실행 기록을 보관했다. smoke는 품질 개선을 입증하지 않는다.

Verda 검증 완료 (2026-10-21): RTX PRO 6000 Blackwell에서 4096환경으로 3회 PPO 업데이트와 새 환경에서 1회 재개, 완전 ramp 128 policy step 검증을 통과했다. 테스트 17개와 train.py CLI 및 launch dry-run도 통과했다. 체크포인트와 모션 해시는 원본과 일치하고 reward revision은 CPU 검증과 동일하다. 본 학습은 시작하지 않았으며 GPU 검증 로그는 `validation_verda_gpu.log`, 결과는 `validation_cuda_0.json`에 보관했다.
