# 태백 팔 진동 보정 모션 Desktop 학습

웹뷰어 8823의 오른쪽 보정본을 사용합니다.

- Task: Cyclo-Mimic-K1-Rev1-Taebaek-XY
- CSV/NPZ: source/assets/motions/K1_rev1/taebaek/k1_taebaek_arm_refined_50hz_converted.*
- 2360프레임, 50Hz, 47.18초. 물리 200Hz, 정책 50Hz.
- 기존 Mimic + pelvis XY 보상. 직전 태백 학습과 같은 4096 environments, 50000 updates.
- 처음부터 학습(resume=False), TensorBoard, upload-model=False.
- 컨테이너: cyclo_taebaek_arm_refined_20261020. 실행 명령은 launch_command.json.
- 기존 상시 cyclo_mjlab 컨테이너에서는 CUDA 인식 실패가 있어, GPU 연결을 새로 만든 별도 컨테이너로 실행했습니다. 기존 컨테이너는 유지합니다.
- 기존 setup/taebaek_local_20261018/train_taebaek.py도 새 모션을 가리킵니다. 변경 전 파일은 train_taebaek_before.py입니다.
- 과거 after_t7 실행 스크립트와 완료 정책·원본 모션은 보존합니다.

로그:
```bash
docker logs -f --tail 60 cyclo_taebaek_arm_refined_20261020
```

연결 검증: validation.json — ReferenceTrajectory 로더, 로봇 관절/바디 이름, 유한값, CSV/NPZ 일치, train/play 모션 경로 확인.

현재 자동 실행 학습은 사용자 요청으로 중단했습니다. GPU 수동 컨테이너 cyclo_taebaek_manual_20261020은 대기 중이며 학습 프로세스는 없습니다. docker exec -it cyclo_taebaek_manual_20261020 bash 로 접속 후 bash setup/taebaek_arm_refined_20261020/manual_training_command.sh 를 실행하면 처음부터 학습합니다.
