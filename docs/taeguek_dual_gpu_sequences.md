# GPU 2대로 태극 2~8장 학습

두 GPU에서 별도 프로세스로 학습할 수 있습니다.

| 물리 GPU | 순서 | 스크립트 |
|---|---|---|
| 0 | 2 → 3 → 4 → 5 | `scripts/reinforcement_learning/train_taeguek_02_to_05.sh` |
| 1 | 6 → 7 → 8 | `scripts/reinforcement_learning/train_taeguek_06_to_08.sh` |

각 스크립트는 해당 GPU 한 장을 사용합니다. 각 품새는 별도 정책으로 처음부터
학습하며, 한쪽 작업이 끝나거나 실패해도 다른 GPU의 작업은 독립적으로 계속됩니다.
기본값은 품새당 환경 4,096개·20,000 iterations입니다.

서버에서 컨테이너에 진입합니다.

```bash
cd /home/ubuntu/cyclo_mjlab
./docker/container.sh enter
```

컨테이너 안에서 GPU 1의 6~8장 학습을 시작합니다.

```bash
tmux new -s taeguek_06_08
bash scripts/reinforcement_learning/train_taeguek_06_to_08.sh
```

`Ctrl+B`, `D`로 tmux에서 나옵니다. GPU 0의 2~5장 학습도 아직 시작하지
않았다면 다음 명령으로 별도 세션에서 시작합니다.

```bash
tmux new -s taeguek_02_05
bash scripts/reinforcement_learning/train_taeguek_02_to_05.sh
```

서버나 컨테이너를 중지하면 학습도 중단됩니다. tmux는 SSH 연결 종료 중에도
프로세스를 유지하지만 재부팅 후 학습을 자동 재개하지는 않습니다.

실행량을 바꾸려면 해당 스크립트 실행 앞에 환경변수를 지정합니다.

```bash
NUM_ENVS=2048 MAX_ITERATIONS=50000 \
  bash scripts/reinforcement_learning/train_taeguek_06_to_08.sh
```

`GPU_ID`를 생략하면 2~5장 스크립트는 GPU 0, 6~8장 스크립트는 GPU 1입니다.
현재 구성에서 두 프로세스는 별도 품새 이름의 결과 폴더와 별도 순차 실행 로그를
사용합니다. 결과는 서버의 EBS 경로 `/home/ubuntu/cyclo_mjlab/logs/`에 저장합니다.

6~8장에도 `reference_pelvis_xy` 보상이 적용됩니다. 가중치 `1.0`,
`std=0.5m`이며 기준 모션의 월드 좌표 골반 XY 경로를 추종합니다.

`nvidia-smi`로 실제 GPU 사용량을 확인합니다. GPU 1만 노출한 학습 프로세스
내부에서는 CUDA 장치 번호가 다시 0부터 시작하므로 로그의 `cuda:0`은 정상입니다.

## 검증 (2026-10-16)

GPU 1에서 32개 환경·품새당 2 iterations로 6 → 7 → 8장 실행을 완료했습니다.
세 품새의 체크포인트와 XY 보상 설정을 확인했으며 6회 업데이트 모두에 XY 보상이
기록됐습니다. 실행 중 `nvidia-smi`에서 학습 프로세스가 물리 GPU 1만 사용하는
것도 확인했습니다. GPU 0의 2~5장 스크립트는 앞서 동일 규모로 검증했습니다.
두 GPU의 4,096환경 본 학습을 동시에 실행하는 부하 테스트는 아직 하지 않았습니다.
