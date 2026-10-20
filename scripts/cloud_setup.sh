#!/bin/bash
# Claude Code 云端会话启动时安装方案H初版的依赖；在本机直接退出，不改动本地环境。

if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR/方案H初版" || exit 0
LOG=/tmp/cloud_setup.log

if uv sync --frozen > "$LOG" 2>&1; then
  echo "云端环境：方案H初版 依赖已安装（uv sync）"
else
  echo "云端环境：方案H初版 依赖安装失败，日志见 $LOG"
fi

# 云端默认网络访问不了 ssd.jpl.nasa.gov，改从 PyPI 的 skyfield-data 包取 DE421 星历（与 JPL 原文件逐字节相同）
if [ ! -f cache/de421.bsp ]; then
  mkdir -p cache /tmp/skyfield-data
  python3 -m pip download skyfield-data==7.0.0 --no-deps -q -d /tmp/skyfield-data >> "$LOG" 2>&1 &&
    python3 - >> "$LOG" 2>&1 <<'EOF'
import glob, hashlib, os, zipfile
data = zipfile.ZipFile(glob.glob("/tmp/skyfield-data/skyfield_data-7.0.0-*.whl")[0]).read("skyfield_data/data/de421.bsp")
assert hashlib.sha256(data).hexdigest() == "a20a7139da04cbc462454634918e9a9ca69127044e2cc9d4f9c16e238d2deedc", "de421.bsp 校验和不符"
open("cache/de421.bsp.part", "wb").write(data)
os.replace("cache/de421.bsp.part", "cache/de421.bsp")
EOF
fi
if [ -f cache/de421.bsp ]; then
  echo "云端环境：DE421 星历已就绪（方案H初版/cache/de421.bsp）"
else
  echo "云端环境：DE421 星历获取失败，Skyfield 校验会尝试联网下载；日志见 $LOG"
fi
exit 0
