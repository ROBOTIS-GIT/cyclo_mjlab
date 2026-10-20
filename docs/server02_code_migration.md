# server02 코드 반영 (2026-10-19)

서버의 학습 코드를 기존 `feature-umr_taeguendo` 브랜치에 통합했다.

- 태극 1~8장 50 Hz 모션 및 pelvis XY 추종 보상.
- 태극 2~8장 SupportV1 잔발 억제 보상과 GPU별 순차 학습 스크립트.
- 태극 4·5장 SupportV2 및 3·4·7·8장 TargetedV3 추가 학습 코드.
- 기존 고려 SupportV3 코드와 공통 지지발 보상.
- 서버에서 사용한 패키지 제약 및 Dockerfile.

## 실행 환경

로컬의 기본 Dockerfile과 재시작 정책은 유지한다. 서버 환경을 다시
빌드하려면 호스트 셸에서 다음 변수를 설정하고 기존 컨테이너 스크립트를 사용한다.
코드 반영 자체는 실행 중인 로컬 컨테이너를 재생성하거나 학습을 시작하지 않는다.

```bash
export CYCLO_MJLAB_DOCKERFILE=docker/Dockerfile.server02
export CYCLO_MJLAB_RESTART=unless-stopped
./docker/container.sh build
./docker/container.sh start
```

코드와 결과물을 EBS 위에 보관하려면 저장소, Docker 데이터 및 containerd
데이터 디렉터리가 모두 EBS 파일시스템에 있는지 확인해야 한다.
Dockerfile만으로 저장 장치가 EBS로 지정되지는 않는다.

## 추가 학습 코드와 체크포인트

`setup/taeguek3478_refinement_20261019/`의 `refinement.py`,
`train_manual.py`, `manual_sequence.py`와 입력 entry bank, 검증 manifest를
서버와 동일하게 보존했다. `manual_sequence.py`는 GPU 0에서 3→4장,
GPU 1에서 7→8장을 선택하며, TargetedV3 체크포인트의 optimizer와
완료 iteration을 복원한다. 원본 warm-start 체크포인트도 SHA256 검증에 필요하다.

이 디렉터리 및 `setup/taeguek45_support_v2_20261018/`의 스크립트는
해당 실험의 재현용이다. manifest와 과거 검증 JSON은 서버 실행 기록이며,
새 실행을 검증했다는 뜻은 아니다. 서버 호스트 경로를 사용하는
`launch_host.py`, `freeze_sources.py`, `capture_startup.py` 등은 로컬에서
그대로 실행하지 않는다. 컨테이너 내부 기준 경로는 `/workspace/cyclo_mjlab`이다.

체크포인트, 학습 로그, policy export는 Git에 넣지 않는다. 전체 원본은
로컬 `/home/hc/Desktop/server02_backup_20261019/cyclo_mjlab/`에 있으며,
무결성 및 재개 검증 결과는 같은 백업의 `verification/`에 있다.
실제 재개 전에는 필요한 `logs/rsl_rl/k1_mimic/` 실행 디렉터리를
현재 프로젝트에서 접근할 수 있게 복원하고 `manual_sequence.py 0 --dry-run`
및 `manual_sequence.py 1 --dry-run`으로 선택 결과를 확인해야 한다.
launch 검증은 원본 warm-start 및 smoke 체크포인트도 요구한다.

서버 종료 시 저장된 TargetedV3 상태는 3·7장 10,000회 완료,
4장 3,001회 및 8장 2,001회 완료이다. 이번 코드 반영으로 학습을 재개하지 않았다.
