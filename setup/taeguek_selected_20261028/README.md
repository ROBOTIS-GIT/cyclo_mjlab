# 태극 1~8장 선택본 (2026-10-28)

사용자 요청: 마지막이라는 이유보다 평가 결과를 우선한다. 사용자에게 중요했던 지지발 잔발을 우선하되 상충하는 지표와 다른 로봇 Rev.는 구분한다. 이전 자료는 삭제하지 않고 작업공간의 `work/taeguek_cleanup_20261028/archive`에 상대 경로 그대로 보관한다. 로그 정리 시 선택본과 함께 최종 체크포인트는 완료 기록과 재개 안전성을 위해 보존한다. 기존 공통 태스크/센서/보상 코드는 다른 모션도 참조하므로 유지한다.

| 장 | 보존 후보와 파일 | 근거/한계 |
|---|---|---|
| 1 | `setup/taeguek01_rev2_20261023`; `setup/taeguek01_rev2_20261023/deployment_model_8000/checkpoint/model_8000.pt` | Rev2 기준 마지막 사용본. Rev1의 결과와 서로 다른 로봇이므로 `k1_mimic/2026-10-16_04-05-56_taeguek01_xy_from_scratch`의 51500/59999도 보존; 최선의 상호 비교는 미확정. |
| 2 | `setup/taeguek_selected_20261028/taeguek02/checkpoint/model_29999.pt` | SupportV1 최종. 이전 대비 지지 홉과 추종 개선. |
| 3 | `setup/taeguek_selected_20261028/taeguek03/checkpoint/model_29999.pt` | SupportV1 최종. 지지 안정성을 유지하며 추종 개선. |
| 4 | `setup/taeguek04_sidekick_stability_v1_20261028`; `setup/taeguek04_sidekick_stability_v1_20261028/audit_20261028/middle/checkpoint/model_5000.pt` | 옆차기 홉·기울기 개선. 골반 XY 경로 오차 증가. 최종 9999는 완료 기록으로 별도 유지. |
| 5 | `setup/taeguek05_soft_contact_v1_20261021/completed_run/model_9999.pt` | 정지 지지발 및 팔 추종 개선; 30.8초 접촉 이탈·일부 충격은 악화. |
| 6 | `setup/taeguek06_precision_support_v4_20261021`; `setup/taeguek06_precision_support_v4_20261021/source/model_6250.pt` | V9 최종보다 발차기 지지발·GRF 유리. |
| 7 | `setup/taeguek07_landing_v4_20261022`; `setup/compose_1023_t7/bundle/checkpoint/model_9999.pt` | V3 대비 일부 팔/경로와 왼발 움직임 개선; 전체 충격적분은 악화. V3 비교 번들 `setup/compose_1022_selection/t7_9999` 보존. |
| 8 | `setup/taeguek08_contact_balance_v7_20261026`; `setup/taeguek_selected_20261028/taeguek08/checkpoint/model_7000.pt` | 평가 후보 중 정지발 평균 속도 최저. GRF/XY는 절충이며 모든 지표 종합 최우수 아님. 19999는 완료 기록으로 유지. |

`source/assets/motions/K1_rev1/taeguek_01`~`taeguek_08`의 50Hz 기준 입력/NPZ/CSV 및 7장 팔 보정 50Hz 모션은 유지한다. 각 선택 실험 자체의 `motion`도 그대로다. 원본·다른 샘플링 모션은 아카이브에 남는다. 1장 Rev2 모션은 `source/assets/motions/K1_rev2`에 유지한다.

복원 절차: `work/taeguek_cleanup_20261028/restore.py`를 참고한다. 이미 같은 경로에 파일이 있으면 덮어쓰지 않는다. 파일 이동은 Git 커밋이나 원격 변경이 아니며 로컬 작업 트리에 나타난다.

Rev2 학습용 MJCF와 메시의 저장소 내 사본은 `source/assets/robots/k1_rev2`에 있다. 별도 `ai_sapiens` 서브모듈의 로컬 미추적 파일은 변경하지 않는다. 학습 로그와 이전 실험 아카이브는 로컬에 보존되며 Git에는 포함하지 않는다.
