# Taegeuk 5 ArmTrackV4

Compose SmoothV3의 팔 추종 부족과 유지 자세 떨림을 함께 개선하는 독립 실험이다.
기준 모션, 관측·행동, PD, 외란, 종료 조건은 유지한다. 기존 5장·7장 실험은 수정하지 않는다.
학습 완료나 실물 개선을 보장하는 설정이 아니며 아래 비교 지표로 체크포인트를 선택해야 한다.

## 보상 변경

기존 SmoothV3의 팔 자세 보상(+0.35, std=0.45 rad), 기준 대비 속도·가속도 오차,
몸통 안정성, 발 지지 보상을 유지한다. 이미 학습한 발/몸통/팔 SmoothV3 보상은
첫 학습 스텝부터 최종 강도로 적용하고, 새 보상만 1,000 업데이트 동안 10%→100%로 올린다.

추가 비용은 `sqrt(1+x²)-1`을 사용한다. 큰 오차에서 상한을 잘라버리지 않아
크게 벗어난 팔도 자세를 회복하는 방향을 구분한다. PPO가 보상 함수를 직접 미분하는 것은 아니다.

| 항목 | 비용 계수 | 정규화 |
|---|---:|---|
| 팔 관절 자세 오차 | 0.60 | 0.5 rad; 어깨 가중치 1.5, 다른 팔 관절 1.0을 평균 1로 정규화 |
| 몸통 기준 팔꿈치 위치 | 0.25 | 0.06 m |
| 몸통 기준 손 위치 | 0.50 | 0.08 m |
| 몸통 기준 손 방향 | 0.20 | 0.5 rad |
| 유지 자세의 기준 대비 속도 오차 | 0.18 | 0.6 rad/s |
| 유지 자세의 기준 대비 가속도 오차 | 0.04 | 20 rad/s² |

각 항목은 해당 관절/신체의 평균이며 합계에 리워드 가중치 -1을 적용한다.
위 숫자는 서로 다른 정규화 척도를 가지므로 숫자만으로 영향력을 비교하지 않는다.
손·팔꿈치의 상대 자세는 실제/기준 각각의 몸통 좌표계에서 계산하여 전역 위치·회전에 독립적이다.
제한된 관절의 자세 회복 비용에는 각도를 ±π로 접지 않는다.

유지 자세 게이트는 기준 관절 속도의 ±100 ms 최대값으로 결정한다.
`1/(1+(speed/0.35)^4)`에 `1/(1+(pose_error/0.35)^4)`를 곱한다.
따라서 빠른 지르기 전후에는 추가 억제가 약해지고, 목표 자세에서 멀어지면
복귀 움직임을 허용한다. 자세·손 추종 비용은 계속 적용되어 잘못된 자세를 유지해도 유리하지 않다.
리셋/프레임 점프 뒤 0.15초 동안 이 추가 속도·가속도 비용은 적용하지 않는다.

## 시작 체크포인트와 학습량

- Compose SmoothV3의 최종 `model_9999.pt`에서 actor/critic을 정확히 불러온다.
- 최초 실행은 새 optimizer로 추가 10,000 업데이트, LR=5e-5 고정, entropy=0.001.
- 노트북 기본 1,024 환경, 24 policy steps/update, 250 업데이트마다 저장.
- 4,096 환경 학습보다 업데이트당 샘플 수가 작다. 같은 10,000 업데이트가 같은 데이터량을 뜻하지 않는다.
- 중단 후 같은 명령어는 V4 finetune만 골라 optimizer/업데이트 수를 복원하고 남은 횟수만 수행한다.
- smoke 결과는 제외한다. 보상 코드/원본 체크포인트/환경 수가 다른 재개는 거부한다.
- 최종 `model_9999.pt`는 새 V4 학습 10,000회 완료를 뜻한다.

체크포인트는 Git에 넣지 않는다. 다음 위치로 별도 전송해야 한다:

```
setup/taeguek05_arm_track_v4_20261019/source/model_9999.pt
SHA256: 00678994ac8fff241f521e9251520f2b562507bc1e9a0a10d25755e6bc4011a3
```

기준 NPZ는 Git에 이미 포함되어 있으며 실행기가 해시를 검증한다:
`source/assets/motions/K1_rev1/taeguek_05/k1_taeguek05_collision_only_50hz_converted.npz`
(`b5c0c1ece88adc2e78ab0abf239a583442cf04b0a00553c89110ec4d4649d744`).

## 노트북 전용 체크아웃

노트북의 기존 `/home/robotis-insu/cyclo_mjlab`에는 미커밋 작업이 있어 그곳에서 바로 pull하지 않는다.
기존 저장소의 `work` 안에 별도 Git 저장소를 만든다. 아래 clone은 최초 한 번만 실행한다.

```bash
git clone --recurse-submodules --single-branch --branch codex/taeguek05-arm-track-v4 \
  https://github.com/ROBOTIS-GIT/cyclo_mjlab.git \
  ~/cyclo_mjlab/work/taeguek05_arm_v4_20261019
```

이후 코드 갱신은 이 별도 체크아웃에서만 진행한다:

```bash
cd ~/cyclo_mjlab/work/taeguek05_arm_v4_20261019
git pull --ff-only
git submodule update --init --recursive
```

체크포인트를 처음 가져오는 경우 노트북 호스트에서 실행한다(데스크톱 IP가 바뀌면 변경):

```bash
cd ~/cyclo_mjlab/work/taeguek05_arm_v4_20261019
mkdir -p setup/taeguek05_arm_track_v4_20261019/source
scp hc@192.168.60.143:/home/hc/robotis/cyclo_mjlab/setup/taeguek05_arm_track_v4_20261019/source/model_9999.pt \
  setup/taeguek05_arm_track_v4_20261019/source/model_9999.pt
```

## 학습 실행

2026-10-19 검증 당시 기존 `cyclo_mjlab` 컨테이너는 NVML Unknown Error로
CUDA를 사용할 수 없었다. 기존 컨테이너를 재시작하지 않고 같은 이미지로
`cyclo_mjlab_taeguek05_v4` 전용 컨테이너를 만들었으며 RTX 5070 인식을 확인했다.
이 컨테이너에서는 별도 체크아웃이 `/workspace/cyclo_mjlab`에 바로 마운트된다.

다른 노트북에서 처음 준비하는 경우에만 호스트에서 생성한다:

```bash
docker run -d --name cyclo_mjlab_taeguek05_v4 --gpus all --shm-size=2g \
  -v "$HOME/cyclo_mjlab/work/taeguek05_arm_v4_20261019:/workspace/cyclo_mjlab" \
  -w /workspace/cyclo_mjlab cyclo-mjlab:latest sleep infinity
```

준비된 노트북 호스트 터미널에서 사용자 권한으로 들어간다:

```bash
docker exec -it --user "$(id -u):$(id -g)" \
  -w /workspace/cyclo_mjlab cyclo_mjlab_taeguek05_v4 bash
```

컨테이너 내부에서:

```bash
python -u setup/taeguek05_arm_track_v4_20261019/launch.py 0 --num-envs 1024 --dry-run
python -u setup/taeguek05_arm_track_v4_20261019/launch.py 0 --num-envs 1024
```

현재 GPU에 다른 작업이 있으면 기본적으로 시작하지 않는다. 기존 viewer를 종료한 후 실행한다.
가벼운 viewer와 동시 실행을 명시적으로 원하면 `--allow-gpu-sharing`을 추가할 수 있다.
이 스크립트는 다른 프로세스를 종료하지 않는다.
노트북이 절전/재부팅되면 학습도 중단되므로 장시간 실행 시 절전을 해제한다.
중단되면 같은 명령어로 재개한다. 체크포인트가 생성된 뒤에는 `--num-envs`를 바꾸지 않는다.

로그는 이 별도 체크아웃의 `logs/rsl_rl/k1_mimic/*_taeguek05_arm_track_v4_finetune_*`에 저장된다.
데스크톱과 별도 경로이므로 기존 5장/7장 로그와 섞이지 않는다.

## 검증과 학습 후 판정

```bash
python setup/taeguek05_arm_track_v4_20261019/test_rewards.py
python setup/taeguek05_arm_track_v4_20261019/validate.py --device cpu --num-envs 16
```

`validate.py`는 smoke 3회 학습 후 새 환경에서 1회 재개하고, 최대 강도 보상의 유한성과
리셋 처리를 점검한다. 본 학습을 시작하지 않으며 개선 효과를 측정하는 평가도 아니다.
GPU smoke는 `--device cuda:0 --num-envs 1024`를 사용한다.

학습 후 기존 Compose와 동일 기준 모션·sim2sim 시작 조건에서 다음을 비교한다:

1. 전체 팔 관절/손·팔꿈치 추종 오차, 특히 18.3–19.3초 오른팔.
2. 기준 동작을 뺀 팔 진동과 유지 자세 떨림, 특히 35초대 손목.
3. 몸통 기울기, 발 들림·지지력, 발차기 높이, 이동 경로와 낙상 여부.

총 리워드 상승이나 episode length만으로 개선을 판단하지 않는다. 기준 모션의 급격한
변화와 구동 한계에 의한 오차는 보상만으로 해결되지 않을 수 있다.
