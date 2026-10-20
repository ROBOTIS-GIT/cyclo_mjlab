# 노트북 실행

대상: robotis-insu@192.168.60.194, RTX 5070 Laptop GPU 8GB.
기존 컨테이너 cyclo_mjlab_taeguek05_v4를 재사용합니다. 컨테이너 이름에 05가 있지만 아래 명령은 태극 7장 SoftContactV1을 실행합니다.

노트북 호스트 터미널:

```bash
bash /home/robotis-insu/cyclo_mjlab/work/taeguek05_arm_v4_20261019/setup/taeguek07_soft_contact_v1_20261020/run_laptop.sh
```

준비 상태만 확인하려면 끝에 `--dry-run`을 붙입니다.
런처는 GPU 점유 시 종료하며 다른 학습을 중단하지 않습니다.
중단 후 같은 명령으로 실행하면 이 실험의 호환 체크포인트에서 이어서 총 10,000회까지 진행합니다. smoke 실행은 제외합니다.

- 환경 수: 4096, GPU 0, 추가 PPO 업데이트: 10000.
- 시작 정책: taeguek_1020_compse/taeguek_07에 해당하는 팔 보정 완료 model_9999.pt.
- 기존 팔/동작 추종 유지 + 지지발 미세 움직임/착지 충격 보상 보완.
- CSV와 NPZ는 이 setup의 motion/에 함께 고정 복사되어 있습니다.
- run_laptop.sh는 컨테이너 시작, 사용자 UID, 쓰기 가능한 캐시 경로를 지정합니다.
- 새 장기 학습은 사용자가 위 명령을 실행할 때 시작합니다.
- 노트북 원래 브랜치와 태극 5장 파일은 유지합니다. 공용 motion_curriculum의 7장 기본 경로는 이전 값이지만, 이 작업의 env_cfg가 고정된 팔 보정 모션 경로로 교체합니다.
- 로그: /home/robotis-insu/cyclo_mjlab/work/taeguek05_arm_v4_20261019/logs/rsl_rl/k1_mimic/*_taeguek07_soft_contact_v1_finetune_*
- 검증 결과: laptop_validation.json. validation.json은 Desktop의 사전 검증 결과입니다.

전송 시 디스크 여유는 약 2.9GB였습니다. 스크립트는 시작 시 1GiB 미만이면 중단합니다. 실행 중 여유 공간을 예약하거나 파일을 자동 삭제하지는 않습니다. 학습 중에는 전원을 연결하고 절전/최대절전 진입을 피하십시오.
