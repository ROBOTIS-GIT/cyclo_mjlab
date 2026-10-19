# 태극 2~5장 순차 학습

`scripts/reinforcement_learning/train_taeguek_02_to_05.sh`는 기본 Mimic 태스크를
2 → 3 → 4 → 5장 순서로 실행합니다. 각 장은 별도 정책을 처음부터 학습합니다.
이전 장의 가중치를 이어받거나 네 모션을 하나로 연결하는 학습은 아닙니다.
한 작업이 실패하면 즉시 중단하며 남은 장은 실행하지 않습니다.

기본값: GPU 0, 환경 4,096개, 장마다 PPO 20,000 iterations,
TensorBoard 로깅, 기존 태스크의 500 iterations 체크포인트 저장 간격.
처음부터 네 모션 파일과 XY 보상 설정을 모두 검사합니다.

클라우드 서버에 접속한 다음:

```bash
cd /home/ubuntu/cyclo_mjlab
./docker/container.sh enter
tmux new -s taeguek_02_05
bash scripts/reinforcement_learning/train_taeguek_02_to_05.sh
```

`Ctrl+B`를 누르고 손을 뗀 뒤 `D`를 누르면 학습을 계속 실행한 채 tmux에서
나옵니다. 다시 컨테이너에 진입한 뒤 `tmux attach -t taeguek_02_05`로 돌아옵니다.
SSH 연결 종료에는 견디지만, 인스턴스나 컨테이너를 중지하면 학습 프로세스는
중단됩니다. 저장한 체크포인트를 이용한 재개는 별도 학습 명령으로 해야 합니다.

실행량 변경 예시 (컨테이너 안):

```bash
NUM_ENVS=2048 MAX_ITERATIONS=50000 GPU_ID=1 \
  bash scripts/reinforcement_learning/train_taeguek_02_to_05.sh
```

학습 없이 사전 검사와 명령 출력:

```bash
bash scripts/reinforcement_learning/train_taeguek_02_to_05.sh --dry-run
```

장별 결과는 `logs/rsl_rl/k1_mimic/<timestamp>_taeguekNN_xy_<RUN_TAG>/`에,
전체 실행 로그는 `logs/sequences/taeguek_02_to_05_<RUN_TAG>.log`에 저장합니다.
서버에서 두 경로 모두 `/home/ubuntu/cyclo_mjlab/logs/` 아래의 EBS에 있습니다.
`RUN_TAG`를 지정하지 않으면 실행 시각과 프로세스 ID로 자동 생성합니다.

## XY 추종 보상

네 태스크 모두 `reference_pelvis_xy`가 활성화됩니다.

- 가중치: `1.0`
- 거리 척도: `std=0.5m`
- 점수: `exp(-((x-x_ref)^2 + (y-y_ref)^2) / 0.5^2)`
- 해당 프레임의 기준 골반 경로에 대한 월드 XY 위치 오차를 사용합니다.
- 실제 로봇에 매 스텝 정렬한 경로를 쓰지 않으므로 누적 XY 이탈을 반영합니다.
- Z, yaw, 전체 무게중심을 직접 평가하는 항은 아닙니다.

학습 출력의 `Episode_Reward/reference_pelvis_xy`와 저장된
`params/env.yaml`에서 적용 여부를 확인할 수 있습니다.

## 실행 검증 (2026-10-16)

서버에서 이 스크립트로 32개 환경·장별 2 iterations를 실행해 2~5장 모두
정상 종료했습니다. 네 정책의 `model_1.pt`와 설정 파일이 생성됐고,
총 8회 PPO iteration 출력 모두에 `Episode_Reward/reference_pelvis_xy`가
기록됐습니다. 보상 함수/좌표계/태스크 등록에 대한 기존 테스트 3개도 통과했습니다.
별도 모의 실행에서 3장 실패 시 4·5장을 시작하지 않고 실패 코드를 반환하는 것을
확인했습니다. 기본 4,096개 환경·20,000 iterations 본 학습은 시작하지 않았습니다.
