# 方案 H：命令行规划 + 网页界面。要用仓库根目录 data/ 里的赛题数据，所以在仓库根目录构建：
#   docker build -f 方案H初版/Dockerfile -t obsplan .
#   docker run --rm -p 127.0.0.1:8501:8501 obsplan            # 网页界面：浏览器打开 http://localhost:8501（只对本机开放）
#   docker run --rm -v "$PWD/out:/out" obsplan python -m obsplan plan --mode fast --validate --out /out/plan
# 国内网络慢时构建加：--build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#                     --build-arg UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
# 构建上下文只含 Dockerfile.dockerignore 放行的文件（赛题数据和方案 H 的代码），本地凭据不会进入镜像。
FROM python:3.12.14-slim-bookworm

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG UV_DEFAULT_INDEX=https://pypi.org/simple
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" uv==0.12.10

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app/方案H初版
# 依赖单独一层，按 uv.lock 锁定的版本安装（换镜像源时 uv 会改写锁文件里的地址，版本不变）
COPY 方案H初版/pyproject.toml 方案H初版/uv.lock ./
RUN uv sync --extra app --no-dev --no-cache

COPY data/数据.zip data/优先级.zip /app/data/
COPY 方案H初版/.streamlit ./.streamlit
COPY 方案H初版/results ./results
COPY 方案H初版/scripts/check_app.py ./scripts/
COPY 方案H初版/app ./app
COPY 方案H初版/obsplan ./obsplan

# 预先算好默认规则下的可见窗口、取好 DE421 星历，再用 1 天的小场景把 numba 函数编译进缓存，
# 容器里第一次规划就不用再等这几步（星历下载失败不影响构建，第一次校验时会再试）
RUN python -m obsplan windows \
 && (python -c "from obsplan.config import Config; from obsplan.ephemeris import ensure_de421; ensure_de421(Config().cache_dir)" \
     || echo "DE421 星历暂未下载，第一次校验时会再试") \
 && python -m obsplan plan --days 1 --out /tmp/warmup > /dev/null \
 && rm -rf /tmp/warmup cache/*_1d_*

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
