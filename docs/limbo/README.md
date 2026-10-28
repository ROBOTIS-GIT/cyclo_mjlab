# Limbo — feature-limbo

현재 로컬 `robotis/cyclo_mjlab`에 Verda의 림보 모션과 최신 Robust V6 학습 코드를 이관했다. 이 브랜치는 로컬 기존 커밋 `fd15eac`에서 분기했다. 기존 태스크와 공통 학습기는 유지하고 림보 파일 및 태스크 등록만 추가한다.

## 실행

로컬 PC에서 기존 컨테이너에 진입:

```bash
docker exec -it -w /workspace/cyclo_mjlab cyclo_mjlab bash
```

컨테이너에서:

```bash
bash scripts/reinforcement_learning/train_limbo_robust.sh
```

기본값은 **4096개 환경, 20000 iterations**다. 실행 중인 다른 림보 학습이 있으면 중복 시작하지 않는다. 본학습은 자동 시작하지 않았다.

- 태스크: `Cyclo-Mimic-K1-Rev1-Limbo-1p2m-RobustObs124`
- 전용 학습 진입점: `scripts/reinforcement_learning/train_limbo.py`
- 시작 정책: `source/assets/checkpoints/limbo/robust_warm_start.pt` (V5 검증본 16200)
- 출력: `logs/rsl_rl/k1_limbo_1p2_robust_obs124/<run>/`

기존 `scripts/reinforcement_learning/train.py`와 다른 태스크의 실행 방식은 변경하지 않았다. 새 진입점은 림보의 fresh-optimizer warm start를 지원한다.

## 포함한 모션

`source/assets/motions/K1_rev1/`:

| 폴더 | 내용 |
|---|---|
| `limbo_1p2_obs124` | 현재 V6용 120cm, 50Hz, 476프레임, 2초 준비 포함 CSV/NPZ |
| `limbo_1p2_prepare_v1` | 준비 시작 자세 65개 reset bank |
| `limbo_1p2_contact_v2` | 120cm 참조 모션과 변환 기록 |
| `limbo_1p3_contact_v4` | 130cm 참조 모션과 변환 기록 |
| `limbo_1p1_contact_v1` | 110cm 참조 모션과 변환 기록 |
| `limbo_1p0_contact_v1` | 100cm 참조 모션과 변환 기록 |
| `limbo_contact_v2`, `limbo_bar_v3` | 앞선 림보 모션과 변환 기록 |
| `limbo_csv_sources` | Desktop/limbo의 80·90·100·110·120·130cm CSV 원본 |

현재 등록한 학습 태스크는 **120cm V6 하나**다. 다른 높이는 모션 자료로 보관하며 V6 학습 태스크로 새로 연결한 것은 아니다. 과거 127/추가 관측 기반 준비 태스크는 등록하지 않았다.

학습 재현에 필요한 교사 정책·교사 replay·시작 정책 3개는 `source/assets/checkpoints/limbo/`에 포함했다. Verda의 학습 로그, 전체 체크포인트 이력, 다른 모션/태권도 코드, `.codex_runs` 실행 폴더는 이관하지 않았다.

## 모델과 호환성

Verda가 사용한 K1은 pelvis collider와 hip/knee/torso 충돌 형상이 현재 로컬 기본 모델과 다르다. 학습 조건 보존을 위해 림보 전용 모델을 `source/assets/robots/limbo_k1_rev1/`에 고정했다. 림보만 이 모델을 사용하며 기본 로봇 설정과 `third_party/ai_sapiens`는 변경하지 않았다.

25개 mesh는 로컬 서브모듈과 Verda의 바이트 해시가 일치하여 재사용한다. 원본 XML·mesh 해시는 모델 폴더의 `source_manifest.json`에 기록했다. XML의 상대 mesh 경로만 로컬 저장소 구조에 맞췄다. 모델의 질량·관성·기구학·충돌·토크 한도 등 18개 물리 배열이 Verda 원본과 동일함을 검사했다.

기존 actor/critic **124채널**을 유지한다:

`motion_command(46), motion_anchor_ori_b(6), base_ang_vel(3), joint_pos_rel(23), joint_vel_rel(23), last_action(23)`

현재 컨테이너의 MJLab 1.2.0, MuJoCo 3.5.0, rsl-rl-lib 5.0.1에서 검증했다. 새 설치나 전역 라이브러리 변경은 하지 않았다.

## 강건성 학습 구성

BeyondMimic 여섯 추종 보상과 림보의 발 접촉·미끄러짐 보상을 유지했다.

- 실제 런타임의 1초 목표각·PD 게인 혼합을 매 5ms 재현하고, 추론은 20ms로 유지한다. `last_action`은 원래 정책 출력이다.
- 준비 시작 75%, 본동작 프레임 재생 25%다. 초기에는 즉시 제어/짧은 전환을 충분히 섞고, 마지막에는 1초 전환 70%, 0.5초 15%, 즉시 제어 15%가 된다.
- 6000 iterations 동안 시작 자세·속도·초기 외란·마찰의 난도를 높인다. FK로 발 구체가 지면을 깊게 관통하지 않도록 만든 3단계 시작 자세 풀을 사용한다.
- 최종 시작 속도 범위는 베이스 xy ±0.30/0.25m/s, 관절 ±0.35rad/s다. 첫 0.2~1.5초에 속도 외란을 한 번 가한다.
- 마찰 0.5~1.1까지 확장하고, COM ±5mm와 encoder bias ±0.003rad, 원래 actor 관측 노이즈 폭의 50%를 적용한다.
- 준비 구간에서는 교사 anchor를 제외한다. 본동작의 약한 anchor는 유지한다. 고정 탐색 std 0.08, learning rate 1e-5다.

보상 초기 튜닝값:

| 항목 | V5 | V6 |
|---|---:|---:|
| 완주 weight | 500 | 100 |
| 준비 성공/실패 weight | ±50 | ±100 |
| 준비 방향 비용 | -0.25 | -0.5 |
| 준비 안정성 비용 | -0.1 | -0.15 |
| 초기 큰 action 비용 | 없음 | -0.15 |
| 초기 목표각 급변 비용 | 없음 | -0.10 |
| 초기 토크 포화 비용 | 없음 | -0.5 |

값은 본학습을 시작하기 위한 초기값이다. 아직 이 구성으로 강건성 향상을 달성했다고 주장하지 않는다.

## 평가와 확인

1000 iterations마다 기존 65개 자세, 동일 자세의 1초 전환, 별도 seed의 중간·최대 외란 프로필을 각각 평가한다. 초기 출력 크기와 목표각 급변도 기록한다. 명목 성능이 크게 떨어지는 평가가 두 번 연속이면 중단하고 기존 후보를 보존한다. `best_robust.pt`는 관측된 후보이며 `best_robust.json`의 `deployment_gate_passed`를 함께 확인해야 한다.

기능 확인:

```bash
python scripts/reinforcement_learning/verify_limbo_robust.py
python scripts/reinforcement_learning/verify_limbo_robust_guard.py
```

검증 출력은 Git에서 제외되는 `logs/limbo_verification/`에 저장한다. 로컬 이관 확인 결과는 이 폴더의 `verification.json`에 기록한다. 짧은 PPO 검사는 코드 연결 확인이며 강건한 정책의 학습 완료를 뜻하지 않는다.

`limbo_preserve_ppo.py`는 rsl-rl-lib 5.0.1의 PPO update를 수정한 코드다. 해당 BSD 라이선스는 `LICENSE.rsl_rl`, 로봇 모델 라이선스는 모델 폴더의 `LICENSE.ai_sapiens`에 보존했다.
