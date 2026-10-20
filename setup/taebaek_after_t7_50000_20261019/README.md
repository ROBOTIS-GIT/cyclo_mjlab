# 태극 7장 완료 후 태백 50,000회 자동 시작

사용자 요청: 현재 실행 중인 태극 7장이 정상 완료된 후 태백을 처음부터 50,000회 학습. 30분마다 이 대화의 heartbeat가 확인.

- 자동화 ID: `7-50000` (Codex 앱이 실행 중이어야 점검 가능)
- 감시 대상: `2026-10-19_11-17-37_taeguek07_arm_smooth_v1_finetune_20261019_111729`
- 완료 조건: `model_9999.pt`, 내부 iter 9999 / env step 240000 / 현재 리워드 revision 일치, 유한한 actor/critic, 저장 안정화 및 프로세스 종료.
- GPU 0의 다른 compute 작업이 있으면 대기. 다른 작업을 종료하지 않음.
- 태백: 접촉 보정 50 Hz / 2360 frames / 47.18초, 4096 environments.
- max_iterations=50000, resume=False, 체크포인트 미지정. 기존 태백에서 이어 학습하지 않음.
- 고정 컨테이너: `cyclo_taebaek_after_t7_50000_20261019`
- run_name: `taebaek_contact_corrected_scratch50000_after_t7_20261019`
- `launch_state.json`과 컨테이너 이름으로 중복 방지. 실패해도 무조건 재시작하지 않음.
- 실제 PPO 업데이트 진행을 확인한 후 한 번 알림, 자동화 일시정지.

확인만:
```bash
python3 /home/hc/robotis/cyclo_mjlab/setup/taebaek_after_t7_50000_20261019/start_after_t7.py
```
완료 조건 충족 시에만 자동 시작:
```bash
python3 /home/hc/robotis/cyclo_mjlab/setup/taebaek_after_t7_50000_20261019/start_after_t7.py --start-if-ready
```
시작 후 로그:
```bash
docker logs --tail 100 -f cyclo_taebaek_after_t7_50000_20261019
```

검증: 실제 실행 이미지 cyclo-mjlab:latest에서 --check 통과(CUDA RTX5090, 모션 전체 유한값, 50Hz, 23관절, 50000회, resume=False). 시작 게이트는 현재 7장 진행 중임을 감지하여 학습을 시작하지 않았음. 이전 완료된 7장 체크포인트를 현재 리워드 revision과 혼동하지 않고 거부함. 태백 PPO 학습은 7장 완료 후 시작하므로 아직 실행 검증하지 않음.
