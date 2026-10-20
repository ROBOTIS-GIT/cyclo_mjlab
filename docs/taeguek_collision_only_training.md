# 태극 1~8장 충돌 보정 모션 학습 연결

2026-10-15 기준, 기본 학습·재생 태스크는 각각의 **50 Hz 충돌 보정 NPZ**를 사용합니다.

| 장 | 학습 태스크 | 모션 디렉터리 |
|---|---|---|
| 1 | `Cyclo-Mimic-K1-Rev1-Taeguek01` | `source/assets/motions/K1_rev1/taeguek_01` |
| 2 | `Cyclo-Mimic-K1-Rev1-Taeguek02` | `source/assets/motions/K1_rev1/taeguek_02` |
| 3 | `Cyclo-Mimic-K1-Rev1-Taeguek03` | `source/assets/motions/K1_rev1/taeguek_03` |
| 4 | `Cyclo-Mimic-K1-Rev1-Taeguek04` | `source/assets/motions/K1_rev1/taeguek_04` |
| 5 | `Cyclo-Mimic-K1-Rev1-Taeguek05` | `source/assets/motions/K1_rev1/taeguek_05` |
| 6 | `Cyclo-Mimic-K1-Rev1-Taeguek06` | `source/assets/motions/K1_rev1/taeguek_06` |
| 7 | `Cyclo-Mimic-K1-Rev1-Taeguek07` | `source/assets/motions/K1_rev1/taeguek_07` |
| 8 | `Cyclo-Mimic-K1-Rev1-Taeguek08` | `source/assets/motions/K1_rev1/taeguek_08` |

각 폴더에는 다음 파일이 있습니다. `NN`은 01~08입니다.

- 60 Hz 원본: `k1_taeguekNN_collision_only.csv` — 시간, root XYZ/WXYZ, 관절 이름 헤더 포함.
- 최종 50 Hz 입력: `k1_taeguekNN_collision_only_50hz.csv` — 원래 시간축으로 리샘플링한 뒤 새로 생긴 충돌 구간만 재보정한 파일.
- 학습 입력: `k1_taeguekNN_collision_only_50hz_converted.npz` — 23관절 위치·속도와 25개 몸체의 위치·방향·속도.
- 변환 CSV: `k1_taeguekNN_collision_only_50hz_converted.csv` — 헤더·시간 열 없음, root XYZ/XYZW와 23관절.

1·2장의 기존 `-Curriculum` 태스크와 커리큘럼/평가 스크립트의 모션 식별 정보도 새 NPZ를 가리킵니다. 3~8장은 기본 Mimic 태스크를 새로 등록했습니다. 기존 모션 파일은 보존했습니다.

## 실행 시간과 속도

입력 **60 Hz → 출력 50 Hz**, `--smooth-window 1`로 원래 시간축에서 리샘플링했습니다. Root 위치·관절각은 선형 보간, root quaternion은 구면 선형 보간을 사용합니다. 추가 속도·가속도 제한, 자세 평활화, 시간 늘리기는 적용하지 않았습니다. 샘플 수가 줄어 짧은 속도 피크는 달라질 수 있습니다.

리샘플링만 적용하면 손 교차 구간 등의 프레임 사이에서 자기충돌이 재발했습니다. 따라서 50 Hz의 충돌 구간을 국소적으로 다시 보정했습니다. 이 보정에는 속도·가속도 제한이나 시간 방향 평활화를 사용하지 않았으며, root XY와 방향은 유지했습니다. 바닥 관통을 막는 최소 root Z 보정도 포함합니다. 충돌 구간의 관절 속도는 국소적으로 달라질 수 있습니다.

NPZ 속도는 최종 50 Hz 자세의 중앙차분(양 끝은 단방향 차분)으로 다시 계산하고, 몸체 위치·방향·속도도 학습 모델로 재계산했습니다. 마지막 시점이 20ms 격자에 맞지 않으면 그 이전 샘플까지 사용하므로 이번 파일은 끝부분만 최대 약 0.0167초 짧습니다. 전체 시간축을 늘리거나 압축하지 않았습니다.

현재 reference command는 정책 스텝마다 한 프레임씩 진행합니다. 기존 기본 설정인 **정책 50 Hz**, decimation 4, **물리 200 Hz**(`timestep=0.005`)를 사용합니다. 이전에 추가했던 60/240 Hz 설정은 제거했습니다. 실제 모터 PD 1kHz는 별도의 하위 제어 주기이며 이번 변경으로 조정하지 않았습니다.

기존 60 Hz `k1_taeguekNN_collision_only_converted.npz`와 변환 CSV는 그대로 보존했습니다. 기본 학습 연결은 이름에 `_50hz_`가 있는 새 NPZ입니다.

## 컨테이너에서 처음부터 학습하는 예시

### Pelvis XY 경로 추종 보상 (2026-10-16)

태극 1~8장 기본 태스크와 기존 1·2장 `-Curriculum` 태스크에 별도
`reference_pelvis_xy` 항을 추가했다. `weight=1.0`, `std=0.5 m`이며,
점수는 `exp(-||pelvis_world_xy - reference_world_xy||² / 0.5²)`이다.
기존 보상과 관측·행동 크기, 외란 설정은 유지한다. 고려·Dance 태스크에는 적용하지 않는다.

기준은 해당 프레임의 `body_pos_w`이며 환경 원점이 이미 포함되어 있다.
실제 로봇 anchor에 매 스텝 정렬되는 `aligned_body_position_w`는 사용하지 않는다.
초기 pose noise는 참조 경로를 배치하는 변환이 아니라 복원해야 하는 초기 외란이다.
따라서 이 보상만을 위해 참조 경로에 초기 노이즈를 다시 더하거나 회전시키지 않는다.
추후 에피소드별 경로 배치 augmentation을 추가한다면 전체 참조/기존 보상과
같은 고정 변환을 사용하도록 함께 수정해야 한다.

제자리 고정이나 전체 CoM 보상이 아니며, 기준에 이동이 있으면 그 경로를 따라간다.
Z/yaw는 이 항에서 직접 평가하지 않는다. 비정상 XY 값은 0점 처리한다.
보상 매니저가 환경의 시간 간격을 처리하므로 함수 자체에는 dt를 곱하지 않는다.
로그에서 `reference_pelvis_xy` 항을 확인할 수 있다.

실행 중인 학습에는 자동 반영되지 않는다. 다음 새 학습 프로세스부터 적용되며,
기존 명령을 그대로 사용할 수 있다. CPU 검증은 다음과 같다.

```bash
python -m unittest discover -s tests -p test_pelvis_xy_rewards.py -v
```

아래 예시는 1장입니다. 태스크의 `01`과 run name을 원하는 장에 맞게 바꾸면 됩니다.

```bash
cd /workspace/cyclo_mjlab
MPLCONFIGDIR=/tmp/cyclo-matplotlib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u scripts/reinforcement_learning/train.py \
  Cyclo-Mimic-K1-Rev1-Taeguek01 \
  --env.scene.num-envs 4096 \
  --agent.run-name taeguek01_collision_only_50hz \
  --agent.resume False \
  --agent.upload-model False
```

변환을 다시 실행하려면 동일 컨테이너에서 다음 명령을 사용합니다.

```bash
python scripts/tools/motion/umr_csv_to_npz.py \
  --input_file source/assets/motions/K1_rev1/taeguek_01/k1_taeguek01_collision_only_50hz.csv \
  --input_fps 50 --output_fps 50 --smooth-window 1
```

## 검증 범위

8개 NPZ 모두 실제 컨테이너의 ReferenceTrajectory CPU 로더로 읽었습니다. 기본 8개 및 기존 커리큘럼 2개 태스크의 학습·재생 경로와 정책 주기/FPS 일치를 검사했습니다. 최종 50 Hz CSV와 관절각·시간·중앙차분 속도를 비교했고, 실제 로봇 모델로 몸체 순기구학을 재계산해 일치를 확인했습니다. 관절각 차이는 float32 저장 정밀도 수준입니다.

추가로 학습 컨테이너의 MuJoCo 3.5.0에서 최종 NPZ 프레임 사이를 10등분한 2ms 간격으로 총 159,438개 자세를 검사했습니다. 설정된 자기충돌은 검출되지 않았습니다. 리샘플링 이후 추가 보정은 5·7장에서 최대 약 13.1°였으며, 나머지 장은 약 1.0~5.3°였습니다.

이는 변환·연결 및 기구학적 충돌 검증이며 학습 실행이나 동역학 추종 검증은 아닙니다. 속도 리밋 초과 가능성은 원본 요청대로 남아 있습니다. 실행 중인 프로세스의 이미 로드된 reference는 바뀌지 않으며, 다음 학습·재생 실행부터 새 설정을 사용합니다.
