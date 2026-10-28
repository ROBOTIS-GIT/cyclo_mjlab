# 태극 6장 PrecisionSupportV4

2026-10-21 두 번째 자동 점검 후, V3에서 남은 지지발 기울기 흔들림과 몸통 경로 오차를 보완한다.

출발점: V3 `model_6250.pt`(6,251 업데이트 완료), SHA256 `6ceeb2490e93fc70b84d8e42b38b587426a24ba225551784d1dfb5b0c2b6eb8b`. source/에 동결했다. 최초 BalancedV1 대비 초반 접촉 끊김 합계는 210→140ms로 줄었지만, 흔들림과 경로 오차 때문에 합격하지 못한 모델이다.

## 변경

- V2/V3의 팔·손·지지·발차기·반동 및 포화가 제거된 GRF 비용은 처음부터 최종 강도로 유지한다.
- 새 기울기 비용: 중력 방향 벡터를 이용해 yaw 회전에 불변. 0.010 벡터 오차(작은 각도에서 약 0.57°) 이후, .0523 스케일/계수 .08의 robust 비용. 기울기 잔차 변화율 .15/s 이후 .60 스케일/계수 .06 비용도 적용한다.
- 보조 기울기 목표는 50fps의 5탭 binomial 필터로 평활화한다. 동일 발의 지지 mask 안 프레임만 섞으며, 공중 프레임을 끌어오지 않는다. 전체 원본 모션과 기존 추종 보상 목표는 그대로다.
- 새 경로 복구: pelvis world XY 오차가 2.5cm를 넘으면 10cm 스케일 robust 비용, weight -.15. 기존 std .5m의 pelvis XY 보상을 유지하며 작은 경로 오차에 대한 구분을 보완한다.
- 새 항목만 2,000 업데이트 동안 강화한다. 학습률 3e-6 고정, entropy .0005, 4096환경, 추가 10,000회 목표, 250회마다 저장.
- 50Hz 정책/200Hz 물리, 원본 CSV/NPZ, 관측, PD, action scale, 외란, 종료 조건 유지.

실행/재개 (컨테이너 안):

```bash
python -u setup/taeguek06_precision_support_v4_20261021/launch.py 0 --num-envs 4096 --iter 10000
```

자동 실행은 tmux 없이 docker exec -d로 실행하고 production.log에 기록한다. 이미 실행 중이면 중복 실행하지 않는다. launch_state.json과 GPU/프로세스를 확인한다. --iter는 이번 V4의 누적 목표이며 원본 V3의 완료 횟수는 포함하지 않는다.

## 검증과 평가

16개 단위 동작 테스트가 통과했다. 4096환경 GPU 학습 3회+새 환경에서 1회 재개, 최종 강도 128스텝, actor/critic 복원·관측/PD 일치·보상 유한성·reset/cache 검사는 validation_cuda_0.json을 확인한다. 본 학습 시작 확인은 launch_state.json에 따로 기록한다. 짧은 검증은 실제 잔발 개선 증명이 아니다.

상태: /home/hc/Documents/ChatGPT/태권도 학습 .. 로부월드/work/taeguek06_early_support_20261021/automation_state.json
보고서: 같은 폴더 checks/REPORT_20261021_2300.md

최초 BalancedV1 비교 기준과 합격 조건은 그대로 유지한다. 1021_compose 배포는 별도 요청이 있을 때 한다.
