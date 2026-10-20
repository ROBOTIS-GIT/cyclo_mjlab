# 태극 7장: 사용자가 선택한 이전 보정 V1 연결

2026-10-19. 비교 뷰어 8814의 **가운데 모션**을 선택했다. 오른쪽 V2는 연결하지 않았다.

- 학습 NPZ: `source/assets/motions/K1_rev1/taeguek_07/k1_taeguek07_arm_corrected_v1_50hz_converted.npz`
- 동반 CSV: 같은 경로의 `.csv`. 숫자 30열, root xyz + quaternion xyzw + 23 joints.
- 50Hz, 2,320 프레임, 마지막 시각 46.38초.
- SHA256: `877390d22ab136a8fc99726e240c09e73efdd2d2b9ff4d4524a6c1c90c0346df`

`source/tasks/mimic/motion_curriculum.py`의 7장 경로를 변경했다. 일반 Taeguek07 및 SupportV1의 train/play 설정이 이 파일을 읽는다. 현재 전용 학습 설정인 `setup/taeguek07_arm_smooth_v1_20261019/refinement.py`도 동일 경로와 해시를 명시했다. 다른 장의 경로, 보상, 물리·제어 주기는 변경하지 않았다.

컨테이너 안에서 각 train/play 설정의 경로를 확인하고, 실제 K1 XML의 관절·바디 이름을 사용해 ReferenceTrajectory CPU 로딩을 검증했다. NPZ와 동반 CSV 관절 값도 일치한다. 결과는 `validation.json`, 파일 출처는 `connection.json`에 기록했다.

기존 기준 모션 파일은 남겨 두었다. 수정 전 설정 파일은 `before/`에 백업했다. 학습이나 배포는 실행하지 않았다.

ArmSmoothV1의 기존 revision 검사는 유지했다. 모션 경로/해시가 바뀌어 revision도 달라졌으므로 이전 모션의 ArmSmoothV1 체크포인트를 동일 실행의 연속 재개로 취급하지 않는다. 기존 `launch.py`는 일치하는 새 revision 체크포인트가 없으면 원래 지정된 complete 7 초기 체크포인트에서 새 optimizer로 미세조정을 시작하는 동작을 유지한다. 이 작업에서는 해당 launcher를 실행하지 않았다.
