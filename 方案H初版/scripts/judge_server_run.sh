#!/usr/bin/env bash
# 服务器上的完整流程（64 核 / 128 GB）。用法：
#   nohup bash scripts/judge_server_run.sh > /root/work/run.out 2>&1 &
# 进度：/root/work/run.log（每个阶段的开始、结束、用时）；各阶段输出：/root/work/logs/<阶段>.log
# 中断后重跑会跳过已完成的场景（judge_pipeline 每个场景单独存盘）。
#
# 训练数据两种配比 × 网络三种大小 = 6 个模型；验证集上选阈值后，每种配比挑最好的大小进基准。
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
J="$P scripts/judge_pipeline.py"
M=obsplan/models
W=/root/work
mkdir -p $W/logs $M

log() { echo "[$(date '+%F %T')] $*" | tee -a $W/run.log; }
stage() {
  local name=$1; shift
  local t0=$(date +%s)
  log "开始 $name"
  "$@" >> $W/logs/$name.log 2>&1
  local rc=$?
  log "结束 $name rc=$rc 用时 $(( ($(date +%s) - t0) / 60 )) 分钟"
  return $rc
}

TRAIN="--dataset v2,v3 --split train --n 60,60"
VAL="--dataset v2,v3 --split validation --n 12,12"
TEST="--dataset v2,v3 --split test"

# 1. 场景（生成输入 + 算可见窗口）
stage scenes_train $J scenes $TRAIN --jobs 16 --threads 4
stage scenes_val   $J scenes $VAL   --jobs 16 --threads 4
stage scenes_test  $J scenes $TEST  --jobs 16 --threads 4

# 冒烟测试（另一份副本里跑）通过后才进入老师阶段
log "等冒烟测试"
while [ ! -f $W/SMOKE_OK ]; do
  if [ -f $W/SMOKE_FAIL ]; then log "冒烟测试失败，停止"; exit 1; fi
  sleep 30
done

# 2. 老师：每个训练场景 20 分钟精简长跑；3. 导出特征与标签
stage teacher $J teacher $TRAIN --minutes 20 --jobs 8 --threads 8
stage export  $J export  $TRAIN --jobs 8 --threads 8

# 4. 训练 6 个模型（并行，每个 8 个 BLAS 线程）
MODELS=""
for data in v3 mix; do
  if [ $data = v3 ]; then DS="--dataset v3 --split train --n 60"; else DS="$TRAIN"; fi
  for size in S M L; do
    case $size in S) H=32,32;; M) H=64,64;; L) H=128,128;; esac
    f=$M/judge_${data}_${size}.npz
    MODELS="$MODELS,$f"
    OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      stage train_${data}_${size} $J train $DS --hidden $H --model $f &
  done
done
wait
MODELS=${MODELS#,}

# 5. 阈值（验证集，6 个模型）→ 每个模型选自己的阈值
stage tune $J tune $VAL --model $MODELS --jobs 8 --threads 8
stage pick $J pick $VAL --model $MODELS

# 每种配比挑验证集上最好的大小
BEST=$($P - <<EOF
import json, numpy as np
best = {}
for f in "$MODELS".split(","):
    m = np.load(f, allow_pickle=False)
    meta = json.loads(str(m["meta"]))
    rel = meta.get("tune", {}).get("best_rel_vs_famou", -1e9)
    data = f.split("judge_")[1].split("_")[0]
    if data not in best or rel > best[data][0]:
        best[data] = (rel, f)
print(",".join(v[1] for v in best.values()))
EOF
)
log "进基准的模型：$BEST"

# 6. 基准（测试集 126 个场景，含 Skyfield 校验）；7. 赛题；8. 汇总
stage bench   $J bench $TEST --model $BEST --jobs 8 --threads 8 --skyfield
stage contest $J contest --model $BEST --threads 32 --skyfield
stage summary $J summary
log "全部完成"
touch $W/ALL_DONE
