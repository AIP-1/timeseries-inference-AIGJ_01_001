#!/usr/bin/env bash
# 과제 4~7 실행과 과제 8 서비스 시작을 한 번에. 보드에서 저장소 폴더에서 실행:
#   nohup scripts/run_all.sh > data/run_all.log 2>&1 &
# 수집기(collector)·influxdb 는 건드리지 않는다. .env 내용은 출력하지 않는다.
set -u
cd "$(dirname "$0")/.."
step() { echo; echo "===== $(date '+%F %T') $*"; }
run() { step "$*"; "$@"; echo "exit=$?"; }

step "과제 3 학습이 돌고 있으면 끝날 때까지 기다림"
while pgrep -f "analysis.task3" > /dev/null; do sleep 60; done

run docker compose run --rm lab python -m analysis.task3_select
run docker compose run --rm lab python -m analysis.task4
run docker compose run --rm lab python -m analysis.task5
run docker compose run --rm lab python -m analysis.task6
run docker compose run --rm lab python -m analysis.task7
run python3 scripts/trt_bench.py data/reports/task7
run docker compose run --rm lab python -m analysis.task7 --report

step "과제 8 서비스 시작 (mqtt 브로커 + infer)"
docker compose up -d --no-deps mqtt infer
sleep 180
run docker compose ps mqtt infer
run curl -s localhost:8091/status
run curl -s localhost:8091/spaces
run docker compose exec -T mqtt mosquitto_sub -p 1884 -t 'aigj/#' -v -C 2 -W 60
step "끝"
