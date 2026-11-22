#!/bin/bash
# 用法：scripts/famou_run_worker.sh 实验ID 评测入口 [并发数]
# 本地评测进程断网会退出；实验还没结束（不是 COMPLETED/FAILED/CANCELLED）就等 30 秒后重启，最多 200 次。
cd "$(dirname "$0")/.."
EXP="$1"; EVAL="$2"
for i in $(seq 1 200); do
  echo "== 第 $i 次启动评测进程 $(date '+%F %T')"
  uv run --extra famou python scripts/famou_worker.py --experiment-id "$EXP" --evaluator-path "$EVAL" --max-concurrent "${3:-1}"
  st=$(uv run --extra famou python scripts/famou_ctl.py experiment status "$EXP" --json 2>/dev/null | tail -1 \
       | python3 -c 'import sys, json; print(json.load(sys.stdin)["data"]["status"])' 2>/dev/null)
  echo "== 评测进程退出，云端状态：${st:-未知} $(date '+%F %T')"
  case "$st" in COMPLETED|FAILED|CANCELLED) break ;; esac
  sleep 30
done
