# v3.0 规则对照与前后端交付

依据：用户提供的 `final/赛题四.md`，文件标题“学生拍摄方案 - 输入格式规范”，v3.0，更新日期 2026-11-23。此文档优先于旧 README 的 2026-10-21 六天口径。本页记录实现及假设；验收数值另见 `v3-acceptance.md`。

## 规则逐项核对

| 条款 | 当前实现 | 证据与边界 |
| --- | --- | --- |
| UTF-8 标准 JSON 数组 | `submission.records/json_bytes/parse_json`；仅四个字段 | 拒绝注释、NaN、重复键、字符串/浮点/布尔 NORAD 和额外字段；UTF-8 不带 BOM |
| 真实五位整数 NORAD | 原始目录编号直接转换为整数；提交规划只选择 10000–99999 | 10000 个目标中 9217 个满足；675 个不足五位，108 个 Alpha-5 解码后超过五位。排除数量写入每份摘要，不补零、不截断、不重编号。应向组织方确认这些真实目标的提交规则 |
| site_1–site_10 | 原始站点下标 + 1 | 原始站址经纬度与 v3 表按两位小数一致，计算保留原始精度与海拔 |
| 站点分组、组内升序 | 按数值站点与开始时间排序 | site_10 排在 site_9 后；严格重新解析检查顺序 |
| 2026-07-31 至 2026-08-06 | Config 默认 7 天；API/CLI 提交模式限制日期 | 规划界面结束为不含端点的次日 00:00；观测的最后允许端点为 08-06 23:59:59Z。先裁剪网格窗口，再按秒精修；不会先产生越界记录再静默截断 |
| 同站间隔 ≥30 秒 | 按开始时间排序，相邻记录的前次结束至下次开始 | 29.999999 秒错误；30 秒通过。多项违规只计一个硬错误记录 |
| 有效时长 ≥120 秒 | 提交区间与单个预报窗口的最大交集 | 不相加分离窗口；时长是端点差，不能把 121 个采样点计作 121 秒 |
| 时长分与优先级 | 120 秒 0.6；到 240 秒线性升至 1；超过 240 秒封顶 | 高/中/低基础分 5/3/2；不在单条任务中四舍五入，报告保留浮点精度 |
| 同星冷却 ≥48 小时 | 同卫星所有可解析记录的相邻前次结束至下次开始 | 未计分记录仍成为下一条的比较锚点。A 合法、B 在冷却内、C 与 B 间隔不足，则 B/C 都不计分；后续相邻间隔达到 48h 后恢复计分 |
| 偏差 ≤1 秒 | 正常计分，但有效交集仍需 ≥120 秒 | 1 秒包含在正常范围 |
| 1<偏差≤5 秒 | 软错误，单条得分乘 0.5 | 5 秒包含在软错误范围 |
| 偏差>5 秒且无交集 | 硬错误 | 任何不足 120 秒的交集也由时长规则判硬错误 |
| 偏差>5 秒且交集≥120 秒 | `unresolved`，最终分数 `null` | v3 没有定义此情况，不能自行补充官方得分 |
| 整体惩罚 | `max(0,1-(hard+soft)/covered)` | 硬/软互斥、每任务只算一次；零覆盖按系数 0 处理，空计划得分 0 且不标记可提交 |
| 逐任务报告 | 原始记录、匹配窗口、有效时长、偏差、状态、全部原因、得分 | 汇总有覆盖、硬/软错误、待确认数量、基础分、惩罚系数、最终本地分 |

## 仍需确认的解释

1. **预报来源**：数据压缩包里只有 `轨道.txt`、`站址.txt` 和三份优先级文件，未发现官方预报窗口。提交模式用 SGP4 生成候选，独立 Skyfield/DE421 按秒校验。评分所用的本地窗口由另一轮 Skyfield 计算得到，限定在提交区间及两侧各 6 秒，不是完整过境预报，也不是官方窗口。报告明确 `official:false`，并保存窗口文件及 SHA-256。可用 CLI `--forecast` 传入实际预报窗口重新评分。
2. **偏差定义**：采用提交区间超出所选窗口左右边界的最大秒数。完全包含在长预报窗口内的短观测偏差为 0。若用两个区间端点的绝对差，会惩罚合法的窗口内选段；此解释仍待官方确认。
3. **冷却锚点**：v3 写“相邻间隔”，没有指明开始/结束。继续采用仓库旧 README 对答疑 #30/#32 的记录“上次结束到下次开始”，这是较保守的锚点；此次未获得原答疑文件的新确认。错误记录不跳过。恢复计分采用相邻间隔达到 48h 后恢复，而非一次违规永久污染之后全部记录。
4. **覆盖计数**：按有正常或软错误计分记录的已知卫星去重；文档示例能证明未知卫星不在分母，不能完全确定只有硬错误的已知卫星是否计入。报告固定公开此策略，不声称是官方最终成绩。
5. **精度**：本地几何采样步长为 1 秒，使用连续可见采样边；不宣称亚秒几何边界精度。文档 207 秒软错误的 1.33 只是两位小数展示，实现保留 1.335，不提前截断。

## 算法边界

H/E/B 复用既有 `run_planning` 与 Schedule。API 固定快速模式和确定性种子 2026，保留按秒精修；输入筛选发生在算法运行前。提交模式关闭全问题的按秒补覆盖和弹射链，以免它们绕过五位编号/选站/末秒窗口筛选。原实验代码和 Streamlit 仍供研究使用，其 CSV 与内部 `combined` 指标不能替代 v3 JSON 与评分报告。

七天内一颗卫星可能合法观测四次。现有 H 的价格子问题至多选择三次，仍可作为构造启发式使用，但其价格目标不是整个七天问题的上界。运行摘要对此使用 `price_objective` 和 `price_subproblem_observation_limit:3`，不展示旧的最优差距。旧上界证明的测试明确用历史六天场景，另测试七天时拒绝输出该上界。本次不声称为新评分优化到了最优。

## HTTP 接口

同源 `/api`；会话 Cookie `obsplan_session` 为 HttpOnly、SameSite=Strict、Path=/api。不同会话不能列出、读取或校验彼此的运行。Nginx 保留 Host；开发 Vite 代理必须 `changeOrigin:false`。上线 HTTPS 后设置 `OBSPLAN_SECURE_COOKIE=1`。这是浏览器会话隔离，不是多用户账户系统；清除 Cookie 后不能从新会话找回原运行，需要管理员从持久目录恢复。

| 接口 | 行为 |
| --- | --- |
| `GET /api/health` | API 可达状态、worker 心跳及队列数量 |
| `GET /api/config` | v3 范围、H/E/B、默认参数与评分解释；建立会话 |
| `POST /api/runs` | JSON `{"date":"2026-07-31","days":7,"stations":[1,2,3,4,5,6,7,8,9,10],"algo":"H"}`；必填 `Idempotency-Key`，新建 202，重复同参数 200，重复不同参数 409，错误输入 422，队列满 429 |
| `GET /api/runs` | 当前会话最近 100 次运行 |
| `GET /api/runs/{id}` | `queued/running/ready/failed`、阶段、进度、错误、结果版本、独立校验状态 |
| `GET /api/runs/{id}/result` | 本次配置、实际任务、摘要、带固定 revision 的下载链接 |
| `POST /api/runs/{id}/validate` | 对当前完整计划独立校验；重复请求不重复排队，失败允许重试 |
| `GET /api/runs/{id}/files/{kind}?revision=N` | `submission/csv/summary/validation/forecast`，每版不可变；尚未就绪返回 409 |

前端保存 H/E/B、日期和站点后直接提交。页面关闭不终止计算；刷新后从本地保存的运行号与服务端会话恢复。网络失败保留幂等提交键，重试不重复创建。结果页支持历史记录、日期和双站切换，任务明细遍历整份计划；提交 JSON、CSV、摘要及报告属于同一运行、同一版本。顶部上传仍为本地文本预览，本轮没有提供替换服务端轨道目录的接口。

## 一致性与失败恢复

SQLite WAL 管理运行及计算队列，短事务领取任务、限制最多 8 项排队/运行作业；一个独立 worker 持 POSIX 文件锁，不允许同存储目录重复 worker。每 5 秒心跳，30 秒超时展示离线。重启时将中断作业标为失败，已排队作业继续，历史已发布计划不受影响。规划失败新建任务，校验失败在原运行上重试。

每版结果先写临时目录、fsync 文件、原子重命名，最后数据库发布版本号。网页只读取已发布版本。崩溃发生在重命名与数据库提交之间时，未公开目录在重试时移动为 `.orphan-*` 保存，再生成新版。下载链接包含 revision，避免校验更新时摘要与计划来自不同版本。已有 `Run.save` 的原子写入和损坏检测均保留。

窗口缓存使用输入数据与全套配置的指纹分区；独立校验重新核对数据指纹，目录改变会拒绝使用新数据校验旧结果。存储/缓存/数据均在服务端设置，HTTP 请求不能传入任意文件路径。

## 本机启动和验证

Python 3.12；依赖由 `uv.lock` 固定。

```sh
uv sync --frozen --extra api --extra app
uv run --no-sync uvicorn obsplan.api:app --host 127.0.0.1 --port 8000 --workers 1
# 另一个终端，同目录：
uv run --no-sync python -m obsplan.worker
# frontend 目录：
npm ci
npm run dev -- --host 127.0.0.1 --port 4174 --strictPort
```

```sh
uv run --no-sync pytest -q
uv run --no-sync python scripts/check_delivery_api.py --algos H --days 7 --stations 1 2 3 4 5 6 7 8 9 10
uv run --no-sync python scripts/check_delivery_api.py --algos E B
uv run --no-sync python -m obsplan plan --mode fast --validate --out outputs/plan-v3
uv run --no-sync python -m obsplan validate outputs/plan-v3/submission.json --out outputs/recheck.json
# 已获得预报文件时；同样是四字段 JSON 数组：
uv run --no-sync python -m obsplan validate outputs/plan-v3/submission.json --forecast forecast.json --out outputs/forecast-recheck.json
```

## Ubuntu 服务与回滚

本次目标主机为 Ubuntu 22.04，2 vCPU、1736 MiB RAM、无 swap。保留已有 Nginx 及其余服务。API 监听 127.0.0.1:8000，1 个进程；独立 worker 串行计算，窗口批量 40 颗、1 个进程，太阳角批量 2000 个时刻，Numba/BLAS 单线程，服务内存上限 1250 MiB。首次 Python/Numba 编译、星历与窗口准备另计用时，不用本机历史秒数承诺服务器性能。

- 代码：`/opt/observation-planner/releases/<release>/`，`current` 指向活动版本。
- 前端：沿用 `/var/www/observation-planner/releases/<release>/` 和 `current`。
- 持久队列/结果：`/var/lib/observation-planner/`；缓存和星历：`/var/cache/observation-planner/`。
- systemd 模板：`deploy/observation-planner-{api,worker}.service`；以专用 `obsplan` 用户运行。
- Nginx API 段见 `deploy/nginx-api.conf`，替换旧 `/api/` 的 404 占位；每次替换前备份完整配置并执行 `nginx -t`。
- 部署时先建立新目录并 `uv sync --frozen --no-dev --extra api`，用新目录运行输入解析/导入检查，再切换 symlink、启动服务；最后检查健康和一次真实 HTTP 流程。
- 回滚先停 worker 与 API，恢复上一版后端/前端 symlink 和 Nginx 配置，执行 `nginx -t` 后 reload，再按上一版能力启动服务。保留持久结果目录，不清空队列或覆盖历史文件。首个后端版本之前没有 API，回滚即恢复静态前端并停用两个新服务。

日志：`journalctl -u observation-planner-api -u observation-planner-worker`。不要在多个版本中同时运行同一持久目录的 worker。公网 80 端口的网络连通性与主机内 Nginx/API 验收分开报告。
