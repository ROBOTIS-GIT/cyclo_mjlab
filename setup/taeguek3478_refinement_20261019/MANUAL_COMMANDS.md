# 사용자가 실행하는 2 GPU 보완 학습

새 학습을 자동으로 시작하거나 기존 학습을 중단하지 않았다. 아래 명령은 사용자가 서버 터미널에서 실행한다.

## 이미 컨테이너 안인 경우

프롬프트가`cyclo@...:/workspace/cyclo_mjlab$`이면 아래 명령을 사용한다. 이 컨테이너에는`docker`명령이 없으므로`docker exec`를 다시 붙이면 안 된다.

```bash
cd /workspace/cyclo_mjlab
tmux new-session -s taeguek34-v3 'python -u setup/taeguek3478_refinement_20261019/manual_sequence.py 0; exec bash'
```

Ctrl+B를 누른 뒤 D로 빠져나온 다음 GPU1을 시작한다.

```bash
tmux new-session -s taeguek78-v3 'python -u setup/taeguek3478_refinement_20261019/manual_sequence.py 1; exec bash'
```

화면에 바로 붙도록`-d`를 사용하지 않는다. 학습이 실패하거나 완료돼도 세션을 남겨 출력을 확인할 수 있도록 마지막에`exec bash`를 붙였다. `cyclo`사용자가 검증 결과를 저장할 수 있도록`launch_validation.json`소유권도 조정했다.

## 호스트에서 접속

```bash
ssh -i ~/Desktop/server02_rtx6000pro ubuntu@52.91.61.88
```

## 기존 두 순차 작업 중단

```bash
cd /home/ubuntu/cyclo_mjlab
sudo python3 setup/taeguek3478_refinement_20261019/manual_stop.py --apply
```

등록한 기존 순차 실행기 둘과 그 자식 프로세스만 종료한다. PID 시작 시각과 명령 해시를 확인하며, 같은 PID가 다른 프로세스로 재사용됐으면 건드리지 않는다. 먼저 실행기를 멈춰 다음 장이 자동 시작되지 않도록 한다. 체크포인트 파일은 삭제하지 않는다. `--apply` 없이 실행하면 조회만 한다.

## GPU별 학습 시작

```bash
tmux new-session -d -s taeguek34-v3 'docker exec -it -w /workspace/cyclo_mjlab cyclo_mjlab python -u setup/taeguek3478_refinement_20261019/manual_sequence.py 0'
tmux new-session -d -s taeguek78-v3 'docker exec -it -w /workspace/cyclo_mjlab cyclo_mjlab python -u setup/taeguek3478_refinement_20261019/manual_sequence.py 1'
```

- GPU0:3→4장, GPU1:7→8장.
- 3장은 최신 정상 V3 체크포인트에서 액터·크리틱·옵티마이저·환경 카운터를 복원하고 남은 횟수만 수행한다. 총10000업데이트를 목표로 하며 저장 간격500회 사이의 미저장 진행분은 재학습된다.
- 4·7·8장은 기존에 선정한 체크포인트에서 각10000회 보완한다. 이후 수동 작업을 다시 시작하면 해당 장의 V3 저장본도 자동으로 이어받는다.
- 기존7장SupportV1의19000회 저장본도 보존한다. 새7장 보완의 시작점은 앞서 sim2sim에서 검증한16500회 선택본이다.
- 해당 GPU를 다른 프로세스가 사용 중이면 실행을 거부한다. 같은 GPU의 수동 실행기는 파일 잠금으로 중복 실행을 막는다.
- 사용자 Ctrl+C 또는 학습/export 실패 시 다음 장은 시작하지 않는다.
- 완료 후보는 같은 setup폴더의`exports/t{장}_targeted_v3_9999_manual_*` 아래에 저장하며, `exported/`, `params/` 및 일치하는CSV를 포함한다. 현재 설치된sim2sim정책은 자동 교체하지 않는다.

## 확인

```bash
tmux attach -t taeguek34-v3
# Ctrl+B를 누른 뒤 D: 학습을 유지하며 빠져나오기
tmux attach -t taeguek78-v3
```

GPU 상태:

```bash
nvidia-smi
```

장별 로그는`manual_t{장}_*.log`, 순차 상태는`manual_gpu{GPU}_*.status.json`이다. 두 tmux세션은 SSH 연결을 끊어도 유지되지만 서버/컨테이너 재시작 후 자동 재개하는 서비스는 아니다.
