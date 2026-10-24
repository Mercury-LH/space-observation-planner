"""空间目标观测规划 · 产品界面

运行（在 方案H初版/ 下）：
    uv sync --extra app
    uv run streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from time import perf_counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import numba
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

SCHEME_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCHEME_DIR))

from obsplan.app_backend import (ALGOS, CLOUD_THRESHOLD, Run, csv_bytes, explain_target, list_runs,  # noqa: E402
                                 orbit_class, pass_profile, replay_weather, run_effort, run_planning, simulate_days,
                                 station_nights,
                                 subset_problem, summary_json, task_rows, validate_run, weather_stats,
                                 status_csv_bytes, status_rows)
from obsplan.target_status import STATUS, describe, summarize  # noqa: E402
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402
from obsplan.effort import EFFORT_ORDER, get_effort  # noqa: E402

RUNS = SCHEME_DIR / "outputs" / "app_runs"
BLUE, ORANGE, AQUA, NAVY = "#2a78d6", "#eb6834", "#1baf7a", "#14213D"
PRI_COLORS = {"高": ORANGE, "中": BLUE, "低": AQUA}

st.set_page_config(page_title="空间目标观测规划", page_icon="🛰️", layout="wide")


@st.cache_resource(show_spinner="正在读取数据并计算可见窗口（首次约 30 秒）……")
def base(cfg_json: str):
    cfg = Config(**json.loads(cfg_json))
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    return cfg, ds, pb, orbit_class(ds)


def cfg_key(cfg: Config) -> str:
    return cfg.to_json()


def problem_for(run: Run):
    cfg = Config(**run.cfg)
    cfg_, ds, pb, orbit = base(cfg_key(cfg))
    if run.subset:
        pb = subset_problem(pb, run.subset["target_frac"], run.subset["stations"], run.subset.get("seed", 0))
    return cfg_, ds, pb, orbit


def effort_reference(selected):
    """Display the published reference cohort, never a predicted current score."""
    profiles = [get_effort(level) for level in EFFORT_ORDER]
    st.caption("六档是离散的计算预设。历史同条件前沿只有四个确认点；high、xhigh 仍待统一校准。"
               "向右增加搜索投入，不保证每次运行质量都严格提高。")
    st.dataframe(pd.DataFrame([
        {"档位": p.key, "方案": p.title,
         "证据": "同条件确认" if p.evidence_status == "controlled" else "探索结果，待校准",
         "参考求解秒": round(p.reference_solver_s, 2) if p.reference_solver_s is not None else None,
         "参考覆盖": p.reference_covered,
         "参考质量 Q": p.reference_quality}
        for p in profiles
    ]), hide_index=True, use_container_width=True)
    st.caption("表中 high、xhigh 的质量与覆盖来自探索记录，耗时留空；不把它们加入下方同条件前沿。")
    verified = [p for p in profiles if p.reference_solver_s is not None]
    fig = go.Figure(go.Scatter(
        x=[p.reference_solver_s for p in verified], y=[p.reference_quality for p in verified],
        text=[p.key for p in verified], mode="markers+text", textposition="top center",
        marker={"size": [15 if p.key == selected else 10 for p in verified],
                "color": [ORANGE if p.key == selected else BLUE for p in verified]},
        customdata=[[p.reference_covered, p.reference_total_s] for p in verified],
        hovertemplate="%{text}<br>求解 %{x:.2f} 秒<br>Q %{y:.6f}<br>覆盖 %{customdata[0]}"
                      "<br>含校验端到端 %{customdata[1]:.2f} 秒<extra></extra>",
    ))
    fig.update_layout(title="已测候选的质量—时间前沿（四个确认点）", height=360,
                      xaxis={"title": "完整求解秒（对数刻度）", "type": "log"},
                      yaxis_title="内部质量 Q", margin=dict(l=25, r=35, t=55, b=35))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(profiles[0].reference_context + "历史参考包含完整来源生成与精修；Q 是内部指标，不是官方成绩。")


# ------------------------------------------------------------------ 侧栏：选择结果 / 新建规划
st.sidebar.title("🛰️ 观测规划")
runs = list_runs(RUNS)
labels = {}
for p in runs:
    try:
        r = Run.load(p)
        m = r.metrics
        name = r.notes.get("label", f"{r.run_id[:15]} · {r.algo}")
        labels[p] = f"{name} · 覆盖 {m['covered']} · 得分 {m['score']:.0f}" + (" · 演示" if r.subset else "")
    except Exception:  # 损坏的结果文件不影响界面
        continue
choice = None
if labels:
    default = st.session_state.get("selected")
    keys = list(labels)
    idx = next((i for i, p in enumerate(keys) if p.stem == default), 0)
    choice = st.sidebar.selectbox("查看结果", keys, index=idx, format_func=lambda p: labels[p])

with st.sidebar.expander("新建规划", expanded=not labels):
    st.caption("规则默认取自答疑文档，规划期默认为企业老师 2026-10-21 确认的 8 月 1–7 日（UTC），可按需修改。")
    planning_style = st.radio("规划方式", ["六档自动规划", "自定义算法"], key="planning_style")
    automatic = planning_style == "六档自动规划"
    effort = None
    if automatic:
        effort = st.select_slider("计算档位", options=EFFORT_ORDER, value="instant", key="effort_level",
                                  help="拖动只选择预设；点击运行才开始求解。instant 是最快档，不表示零等待。")
        st.caption("instant · medium · high · xhigh · max · ultra")
        profile = get_effort(effort)
        st.markdown(f"**{profile.title}**")
        st.caption(profile.description)
        if profile.evidence_status == "controlled":
            st.caption(f"历史全量参考：求解 {profile.reference_solver_s:.1f} 秒；"
                       f"含校验 {profile.reference_total_s:.1f} 秒。")
        else:
            st.info("已有探索结果，尚待同条件校准，不展示耗时承诺。")
        st.caption("六档均使用已采纳的本地伐谋规则，并自动做独立逐秒校验。")
    d0 = st.date_input("规划开始（UTC 0 点）", value=date(2026, 8, 1))
    days = st.number_input("规划天数", 1.0, 7.0, 6.0, 1.0)
    c1, c2 = st.columns(2)
    elev_min = c1.number_input("最低仰角 °", 0.0, 30.0, 10.0, 1.0)
    elev_max = c2.number_input("最高仰角 °", 60.0, 90.0, 85.0, 1.0)
    sun_max = st.number_input("站点太阳高度角上限 °", -18.0, 0.0, -6.0, 1.0)
    use_sunlit = st.checkbox("目标须被太阳照亮", value=True)
    moon = st.slider("月光规避角 °（0 = 不考虑，加分项）", 0, 45, 0, 5)
    if automatic:
        w_cov, algo, fast, iters, weather, evolved, refine_on = 0.5, "portfolio", False, 300, False, True, True
        st.caption("自动档位统一使用覆盖与观测分各 50% 的内部质量口径。更改规则、日期或规模后，历史参考不再直接适用。")
    else:
        w_cov = st.slider("覆盖数权重（其余给得分）", 0.0, 1.0, 0.5, 0.05)
        algo = st.radio("算法", ["H", "E", "B"], format_func=lambda k: ALGOS[k])
        mode = st.radio("模式", ["快速", "高质量"], horizontal=False)
        fast = mode == "快速"
        iters = 150 if fast else 300
        weather = st.checkbox("按历史天气做期望价值规划（加分项，仅 H）", value=False, disabled=algo != "H")
        evolved = st.checkbox("用伐谋演化的放置规则（仅 H，推荐）", value=True, disabled=algo != "H" or weather)
        refine_on = st.checkbox("按秒精修（推荐）", value=True)
    st.markdown("**演示范围**（缩小规模）")
    frac = st.slider("目标比例", 0.05, 1.0, 1.0, 0.05)
    stations = st.multiselect("站点", list(range(1, 11)), default=list(range(1, 11)))
    if not stations:
        st.warning("至少选择一个站点。")
    go_btn = st.button("运行规划", type="primary", use_container_width=True, disabled=not stations)

if go_btn:
    ui_started = perf_counter()
    cfg = Config(start_utc=f"{d0:%Y-%m-%d}T00:00:00", days=float(days), elev_min=elev_min, elev_max=elev_max,
                 sun_max=sun_max, use_sunlit=use_sunlit, w_cov=w_cov, w_score=1 - w_cov, moon_sep_deg=float(moon))
    cfg_, ds, pb, orbit = base(cfg_key(cfg))
    subset = {}
    if frac < 1 or len(stations) < 10:
        subset = {"target_frac": float(frac), "stations": [s - 1 for s in stations], "seed": 0}
        pb = subset_problem(pb, subset["target_frac"], subset["stations"], 0)
    upstream_preprocessing_s = perf_counter() - ui_started
    bar = st.sidebar.progress(0.0, text="开始规划……")
    callback = lambda f, t: bar.progress(min(1.0, max(0.0, f)), text=t)
    try:
        if automatic:
            run = run_effort(pb, effort, progress=callback, full_problem=not subset,
                             workers=min(8, numba.config.NUMBA_NUM_THREADS))
        else:
            run = run_planning(pb, algo, iters=iters, progress=callback,
                               weather=weather and algo == "H", fast=fast, refine_seconds=refine_on,
                               full_problem=not subset, evolved_rule=evolved)
    except (ValueError, RuntimeError) as exc:
        st.sidebar.error(str(exc))
        st.stop()
    run.subset = subset
    run.notes.update(upstream_preprocessing_time_s=upstream_preprocessing_s,
                     ui_total_time_s=perf_counter() - ui_started,
                     ui_total_time_scope="data/windows/subset preparation through solve"
                     + (" and initial validation" if automatic else "") + "; excludes save and render")
    bar.progress(1.0, text=f"完成：求解 {run.runtime_s:.2f} 秒")
    run.save(RUNS)
    st.session_state["selected"] = run.run_id
    st.rerun()

if choice is None:
    st.title("空间目标观测规划")
    st.info("在左侧拖动计算档位，点击「运行规划」后开始求解。结果和档位会一起保存。instant 是最快档，仍需计算与校验；首次运行还需准备可见窗口。")
    effort_reference(effort)
    st.stop()

run = Run.load(choice)
cfg, ds, pb, orbit = problem_for(run)
m = run.metrics
rows = task_rows(pb, run, orbit)
df = pd.DataFrame(rows)

st.title("空间目标观测规划")
if run.notes.get("effort_level"):
    st.caption(f"本次计算档位：{run.notes['effort_level']} · 求解 {run.runtime_s:.2f} 秒 · "
               f"独立校验 {run.notes.get('validation_time_s', 0):.2f} 秒")
    if "ui_total_time_s" in run.notes:
        st.caption(f"数据准备至校验结束共 {run.notes['ui_total_time_s']:.2f} 秒（不含保存与页面绘制）。")
with st.expander("六档说明与历史实测前沿"):
    effort_reference(effort)
scope = "全量" if not run.subset else f"演示：{run.subset['target_frac']:.0%} 目标 × {len(run.subset['stations'])} 站"
extra = (f" · 月光规避 {cfg.moon_sep_deg:g}°" if cfg.moon_sep_deg > 0 else "") + \
    (" · 按历史天气期望规划" if run.notes.get("weather_aware") else "") + \
    (" · 按秒精修" if run.notes.get("refine", {}).get("refined") else "")
st.caption(f"{ALGOS.get(run.algo, run.algo)} · {cfg.start_utc[:10]} 起 {cfg.days:g} 天（UTC）· {scope} · "
           f"仰角 {cfg.elev_min:g}°–{cfg.elev_max:g}° · 太阳角 < {cfg.sun_max:g}° · 目标受照 {'是' if cfg.use_sunlit else '否'}{extra}")

tabs = st.tabs(["概览", "站点地图", "甘特图", "时间价格", "任务解释", "天气与月光", "校验与导出", "结果对比", "规则说明"])

# ------------------------------------------------------------------ 概览
with tabs[0]:
    k = st.columns(6)
    v = run.validation
    tiles = [
        ("覆盖目标", f"{m['covered']}", f"可观测 {(pb.max_obs > 0).sum()} / 全部 {ds.n_targets}"),
        ("累计得分", f"{m['score']:.0f}", "含 48 小时后的重复观测"),
        ("综合指标", f"{m['combined']:.4f}", f"覆盖权重 {cfg.w_cov:g}（内部口径）"),
        ("观测任务", f"{m['tasks']}", f"其中重复观测 {m['repeat_tasks']}"),
        ("求解用时", f"{run.runtime_s:.2f} 秒", "不含窗口预处理与独立校验"),
        ("Skyfield 校验", "未校验" if v is None else ("未完成" if v.get("geometry_checked") is False else
          ("全部合法" if v["legal"] else "有违规")),
         "到「校验与导出」运行" if v is None else f"违规 {sum(v['violations'].values())} 项"),
    ]
    for col, (label, value, note) in zip(k, tiles):
        col.metric(label, value)
        col.caption(note)
    c1, c2, c3 = st.columns(3)
    pri = pd.DataFrame({"优先级": ["高", "中", "低"],
                        "已覆盖": [m["covered_高"], m["covered_中"], m["covered_低"]],
                        "未覆盖": [m["total_高"] - m["covered_高"], m["total_中"] - m["covered_中"], m["total_低"] - m["covered_低"]]})
    fig = px.bar(pri, x="优先级", y=["已覆盖", "未覆盖"], color_discrete_sequence=[BLUE, "#d9dce3"],
                 title="各优先级覆盖情况")
    fig.update_layout(legend_title_text="", yaxis_title="目标数", height=360)
    c1.plotly_chart(fig, use_container_width=True)
    if len(df):
        oc = df.groupby(["轨道", "类型"]).size().reset_index(name="任务数")
        fig = px.bar(oc, x="轨道", y="任务数", color="类型", color_discrete_sequence=[BLUE, ORANGE],
                     title="按轨道类型的观测任务")
        fig.update_layout(legend_title_text="", height=360)
        c2.plotly_chart(fig, use_container_width=True)
    dark_h = [sum(b - a for a, b in runs_) * cfg.cell_s / 3600 for runs_ in pb.win.dark]
    util = pd.DataFrame({"站点": [f"{i + 1}号" for i in range(pb.n_stations)], "黑夜可用": dark_h,
                         "已安排": m["station_busy_h"]})
    fig = px.bar(util, x="站点", y=["黑夜可用", "已安排"], barmode="group", color_discrete_sequence=["#d9dce3", NAVY],
                 title="各站时间利用（小时）")
    fig.update_layout(legend_title_text="", yaxis_title="小时", height=360)
    c3.plotly_chart(fig, use_container_width=True)

    st.subheader("目标状态分析")
    st.caption("按观测任务状态理解（答疑 #26）：每颗目标观测了几次（只计满足 48 小时间隔的观测，与计分口径相同）、"
               "总观测时长、首次与最近一次观测时间，以及没有覆盖或没有观测满的原因。")
    srows = status_rows(pb, run)
    ssum = summarize(srows)
    st.write(describe(ssum))
    cnt = pd.DataFrame(ssum["by_priority"]).T
    cnt = cnt.loc[:, (cnt != 0).any(axis=0)]
    cnt.index = [f"{p}优先级" for p in cnt.index]
    st.dataframe(cnt, use_container_width=True)
    pick = st.selectbox("查看明细", ["未覆盖与未观测满"] + list(STATUS.values()), key="status_pick")
    sdf = pd.DataFrame(srows)
    if pick == "未覆盖与未观测满":
        sdf = sdf[(sdf["status"] == "UNCOVERED") | ((sdf["n_obs"] > 0) & (sdf["n_obs"] < sdf["max_obs"])
                                                    & (sdf["priority"] == "高"))]
        st.caption("未覆盖的目标，以及还能再观测、但没有观测满的高优先级目标。")
    else:
        sdf = sdf[sdf["status_cn"] == pick]
    show = sdf.rename(columns={"target_id": "目标编号", "target_name": "名称", "priority": "优先级", "orbit_class": "轨道",
                               "status_cn": "状态", "n_obs": "观测次数", "max_obs": "理论最多", "observed_seconds": "观测秒数",
                               "task_score": "得分", "first_obs_utc": "首次观测", "last_obs_utc": "最近观测",
                               "stations": "站点", "n_windows": "窗口数", "note": "说明"})
    st.dataframe(show.drop(columns=["status", "window_seconds", "n_stations_visible"]).head(500),
                 use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ 站点地图
with tabs[1]:
    cnt = df.groupby("站点").size() if len(df) else pd.Series(dtype=int)
    st_df = pd.DataFrame({"站点": [f"{i + 1}号站" for i in range(ds.n_stations)], "经度": ds.st_lon, "纬度": ds.st_lat,
                          "海拔(m)": ds.st_alt_m, "任务数": [int(cnt.get(i + 1, 0)) for i in range(ds.n_stations)],
                          "已安排(小时)": m["station_busy_h"]})
    fig = px.scatter_geo(st_df, lon="经度", lat="纬度", size="任务数", text="站点", size_max=22,
                         hover_data={"海拔(m)": True, "任务数": True, "已安排(小时)": True, "经度": ":.2f", "纬度": ":.2f"},
                         projection="natural earth", color_discrete_sequence=[ORANGE])
    fig.update_traces(textposition="bottom center")
    fig.update_geos(showcountries=True, countrycolor="#d9dce3", landcolor="#f1f0ea", oceancolor="#eef2f8",
                    showocean=True)
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(st_df, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ 甘特图
with tabs[2]:
    nights = station_nights(pb)
    c1, c2, c3 = st.columns([1, 2, 1])
    s_sel = c1.selectbox("站点", list(range(1, pb.n_stations + 1)), format_func=lambda s: f"{s}号站")
    ns = [(n, a, b) for (s, n, a, b) in nights if s == s_sel - 1]
    if not ns:
        st.info("该站在规划期内没有黑夜时段。")
    else:
        n_sel = c2.selectbox("夜晚", ns, format_func=lambda x: f"第 {x[0] + 1} 夜：{cfg.start_dt + timedelta(seconds=x[1] * cfg.cell_s):%m-%d %H:%M} – "
                                                          f"{cfg.start_dt + timedelta(seconds=x[2] * cfg.cell_s):%m-%d %H:%M} UTC")
        color_by = c3.radio("颜色", ["优先级", "轨道"], horizontal=True)
        a = cfg.start_dt + timedelta(seconds=n_sel[1] * cfg.cell_s)
        b = cfg.start_dt + timedelta(seconds=n_sel[2] * cfg.cell_s)
        part = df[(df["站点"] == s_sel) & (df["开始"] >= a) & (df["结束"] <= b)] if len(df) else df
        occupied_s = part["时长(秒)"].sum() + 30 * len(part) if len(part) else 0
        st.caption(f"这一夜安排了 {len(part)} 个任务，占用约 {occupied_s / 60:.0f} 分钟 / "
                   f"黑夜 {(b - a).total_seconds() / 60:.0f} 分钟")
        if len(part):
            fig = px.timeline(part, x_start="开始", x_end="结束", y="轨道", color=color_by,
                              color_discrete_map=PRI_COLORS if color_by == "优先级" else {"低轨": BLUE, "中高轨": ORANGE, "地球同步": AQUA},
                              hover_data=["目标编号", "名称", "优先级", "时长(秒)", "得分", "类型"])
            fig.update_yaxes(categoryorder="array", categoryarray=["地球同步", "中高轨", "低轨"])
            fig.update_layout(height=380, xaxis_title="UTC", yaxis_title="", legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
            show = part.sort_values("开始").copy()
            for c in ("开始", "结束"):
                show[c] = show[c].dt.strftime("%m-%d %H:%M:%S")
            st.dataframe(show, hide_index=True, use_container_width=True, height=300)

# ------------------------------------------------------------------ 时间价格
with tabs[3]:
    if run.mu is None:
        st.info("这份结果未保存时间价格矩阵。可在自定义算法中选择 H 查看价格分析。")
    else:
        mu = run.mu.astype(float)
        day = st.slider("查看第几天", 1, int(cfg.days), 2)
        bins = 20
        a0, b0 = (day - 1) * int(86400 / cfg.cell_s), min(day * int(86400 / cfg.cell_s), mu.shape[1])
        blk = mu[:, a0:b0]
        blk = blk[:, : blk.shape[1] // bins * bins].reshape(mu.shape[0], -1, bins).mean(axis=2)
        blk = blk / max(mu.max(), 1e-12)
        xs = [cfg.start_dt + timedelta(seconds=(a0 + i * bins) * cfg.cell_s) for i in range(blk.shape[1])]
        fig = go.Figure(go.Heatmap(z=blk, x=xs, y=[f"{i + 1}号站" for i in range(mu.shape[0])],
                                   colorscale=[[0, "#fcfcfb"], [0.2, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]],
                                   colorbar=dict(title="相对价格"), hovertemplate="%{y} %{x|%H:%M} UTC<br>价格 %{z:.2f}<extra></extra>"))
        fig.update_layout(height=480, yaxis=dict(autorange="reversed"), xaxis_title="UTC")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("价格越高，说明这一时段被越多目标争抢。以默认的 8 月规划期为例：高纬度站夜短，整夜都接近最贵；"
                    "低纬度站午夜前后更贵。规划时每个任务优先放进它窗口内最便宜的时段。")

# ------------------------------------------------------------------ 任务解释
with tabs[4]:
    c1, c2 = st.columns([1, 2])
    covered = set(run.target.tolist())
    examples = {"自己输入编号": None}
    hi_cov = [t for t in np.nonzero(ds.priority == 5.0)[0] if t in covered]
    hi_cov.sort(key=lambda t: orbit[t] != "低轨")          # 低轨目标的过境曲线更能说明约束
    unc = [t for t in np.nonzero(pb.max_obs > 0)[0] if t not in covered]
    if hi_cov:
        examples["示例：一颗已覆盖的高优目标"] = int(hi_cov[0])
    if unc:
        examples["示例：一颗没排上的目标"] = int(unc[0])
    ex = c1.radio("选择目标", list(examples))
    if examples[ex] is None:
        num = c2.text_input("目标编号（NORAD，如 50574 或 100045）", value=str(ds.satnum[hi_cov[0]]) if hi_cov else "")
        hit = np.nonzero(ds.satnum == int(num))[0] if num.strip().isdigit() else []
        t = int(hit[0]) if len(hit) else None
        if t is None and num:
            c2.warning("没有这个编号。")
    else:
        t = examples[ex]
    if t is not None:
        e = explain_target(pb, run, t)
        st.subheader(e["name"])
        for line in e["lines"]:
            st.markdown(f"- {line}")
        if e["windows"]:
            wdf = pd.DataFrame(e["windows"]).sort_values("开始").drop(columns=["最低价格"], errors="ignore")
            for c in ("开始", "结束"):
                wdf[c] = pd.to_datetime(wdf[c]).dt.strftime("%m-%d %H:%M:%S")
            st.dataframe(wdf, hide_index=True, use_container_width=True,
                         column_config={"相对价格": st.column_config.ProgressColumn("相对价格", min_value=0, max_value=1),
                                        "占用率": st.column_config.ProgressColumn("占用率", min_value=0, max_value=1)})
        mine = sorted(np.nonzero(run.target == t)[0].tolist(), key=lambda i: run.start_s[i])
        if mine:
            st.markdown("**过境详情：逐秒复核这次观测满足的条件**")
            pick = st.selectbox("选择一次观测", mine, key="pass_pick",
                                format_func=lambda i: f"{run.station[i] + 1}号站 · "
                                                      f"{cfg.start_dt + timedelta(seconds=float(run.start_s[i])):%m-%d %H:%M:%S} UTC 起 · "
                                                      f"{int(run.dur_s[i])} 秒")
            pf = pass_profile(pb, run, int(pick))
            ts = [cfg.start_dt + timedelta(seconds=float(x)) for x in pf["sec"]]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.1,
                                subplot_titles=("目标仰角（°）", "站点太阳高度角（°）"))
            fig.add_trace(go.Scatter(x=ts, y=pf["el"], mode="lines", line=dict(color=BLUE, width=2), name="目标仰角",
                                     hovertemplate="%{x|%H:%M:%S}<br>仰角 %{y:.2f}°<extra></extra>"), 1, 1)
            for y in (cfg.elev_min, cfg.elev_max):
                fig.add_hline(y=y, line_dash="dash", line_color="#8A93A3", line_width=1, row=1, col=1)
            fig.add_trace(go.Scatter(x=ts, y=pf["sun_el"], mode="lines", line=dict(color=ORANGE, width=2), name="太阳高度角",
                                     hovertemplate="%{x|%H:%M:%S}<br>太阳高度角 %{y:.2f}°<extra></extra>"), 2, 1)
            fig.add_hline(y=cfg.sun_max, line_dash="dash", line_color="#8A93A3", line_width=1, row=2, col=1)
            t0 = cfg.start_dt + timedelta(seconds=pf["start"])
            t1 = cfg.start_dt + timedelta(seconds=pf["end"])
            fig.add_vrect(x0=t0, x1=t1, fillcolor=ORANGE, opacity=0.15, line_width=0, row="all", col=1)
            dark = np.diff(np.concatenate([[0], (~pf["lit"]).astype(int), [0]]))
            for a_, b_ in zip(np.nonzero(dark == 1)[0], np.nonzero(dark == -1)[0]):
                fig.add_vrect(x0=ts[a_], x1=ts[b_ - 1], fillcolor=NAVY, opacity=0.12, line_width=0, row="all", col=1)
            fig.update_layout(height=460, showlegend=False, margin=dict(t=40, b=20, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            sm = pf["summary"]
            st.caption(f"观测 {t0:%H:%M:%S}–{t1:%H:%M:%S} UTC（{sm['dur']} 秒）：仰角 {sm['el_min']:.1f}°–{sm['el_max']:.1f}°"
                       f"（要求 {cfg.elev_min:g}°–{cfg.elev_max:g}°），站点太阳高度角最高 {sm['sun_max']:.1f}°"
                       f"（要求低于 {cfg.sun_max:g}°），目标{'全程受照' if sm['lit_all'] else '有时段处于地影'}。"
                       "橙色带为观测时段，深色带为目标处于地球阴影；虚线为规则界限。前后各显示 5 分钟。")

# ------------------------------------------------------------------ 天气与月光（加分项）
with tabs[5]:
    st.markdown("赛题基础部分不考虑天气；这里作为加分项：用 Open-Meteo 历史再分析数据取各站**过去 10 年同期**的逐小时总云量，"
                f"云量 ≤ {CLOUD_THRESHOLD:g}% 视为可观测。首次使用需要联网（只发送四舍五入到 0.5° 的站点坐标），之后读缓存。")
    if st.button("读取历史云量并回放本计划", type="primary"):
        st.session_state["wx_run"] = run.run_id
    if st.session_state.get("wx_run") == run.run_id:
        with st.spinner("读取历史云量……"):
            cloud, sc = weather_stats(pb)
            rep = replay_weather(pb, run)
        c1, c2 = st.columns(2)
        wdf = pd.DataFrame({"站点": [f"{i + 1}号" for i in range(len(sc))], "晴空概率": sc,
                            "平均云量%": np.nanmean(cloud, axis=(0, 2))})
        fig = px.bar(wdf, x="站点", y="晴空概率", color_discrete_sequence=[BLUE], title=f"各站 {cfg.start_dt:%m 月 %d 日}起 {cfg.days:g} 天同期的晴空概率（全天统计，过去 10 年）",
                     hover_data={"平均云量%": ":.0f", "晴空概率": ":.2f"})
        fig.update_layout(yaxis_tickformat=".0%", height=380)
        c1.plotly_chart(fig, use_container_width=True)
        years = list(range(2016, 2016 + len(rep["coverage_by_year"])))
        ydf = pd.DataFrame({"年份": [str(y) for y in years], "覆盖目标": rep["coverage_by_year"]})
        fig = px.bar(ydf, x="年份", y="覆盖目标", color_discrete_sequence=[ORANGE], title="把本计划放到过去每一年的真实天气里")
        fig.add_hline(y=m["covered"], line_dash="dash", line_color="#8A93A3", annotation_text=f"计划覆盖 {m['covered']}")
        fig.update_layout(height=380)
        c2.plotly_chart(fig, use_container_width=True)
        k = st.columns(3)
        k[0].metric("期望实际覆盖", f"{rep['coverage_mean']:.0f}")
        k[0].caption(f"计划 {m['covered']}，约 {rep['coverage_mean'] / max(m['covered'], 1):.0%}")
        k[1].metric("期望实际得分", f"{rep['score_mean']:.0f}")
        k[1].caption(f"计划 {m['score']:.0f}")
        k[2].metric("月光规避", f"{cfg.moon_sep_deg:g}°" if cfg.moon_sep_deg > 0 else "未考虑")
        k[2].caption("可在左侧新建规划时设置")
        st.info("结论：只靠历史统计做期望价值规划，改进空间有限（见 README 的样本外实验）；要绕开云，需要按实际执行结果动态重规划（见下方）。")

    st.divider()
    st.subheader("动态重规划：每天按执行结果重排")
    st.markdown("模拟真实运行：每天 0 点（UTC）根据前一天的**实际执行结果**更新状态——哪些目标已成功覆盖、还剩几次可观测、"
                "48 小时间隔从哪次成功观测算起——然后重新规划剩余时段，只执行当天的任务。观测期间每个小时云量都 "
                f"≤ {CLOUD_THRESHOLD:g}% 才算成功；失败的目标第二天可以再排。")
    dyn_path = SCHEME_DIR / "results" / "dynamic.json"
    DYN_LABELS = {"static": "静态<br>标准计划", "static_clim": "静态<br>+气候统计", "dynamic_clim": "每天重排<br>+气候统计",
                  "dynamic_forecast": "每天重排<br>+真实预报", "dynamic_oracle": "每天重排<br>+完美预报",
                  "hindsight": "事后最优<br>天气全知"}
    if dyn_path.exists():
        dj = json.loads(dyn_path.read_text(encoding="utf-8"))
        rows = []
        for yr, v in dj["years"].items():
            for key, p_ in v["policies"].items():
                rows.append({"策略": DYN_LABELS.get(key, p_["name"]), "年份": f"{yr} 年天气", "综合": p_["combined"], "成功覆盖": int(p_["covered"]),
                             "得分": round(p_["score"]), "类型": "参照（现实中拿不到）" if key in ("dynamic_oracle", "hindsight") else "可实现"})
        ddf = pd.DataFrame(rows)
        fig = px.bar(ddf, x="策略", y="综合", color="年份", barmode="group", pattern_shape="类型",
                     color_discrete_sequence=[BLUE, ORANGE], hover_data={"成功覆盖": True, "得分": True, "类型": False},
                     title="全量实验：真实天气下实现的综合指标（2024、2025 年）")
        fig.update_layout(height=420, xaxis_title="", yaxis_tickformat=".2f", legend_title_text="", xaxis_tickangle=0,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, use_container_width=True)
        gains, fc_gain, oracle_gain = [], [], []
        for v in dj["years"].values():
            p_ = v["policies"]
            if {"static", "dynamic_clim"} <= set(p_):
                gains.append(p_["dynamic_clim"]["combined"] / p_["static"]["combined"] - 1)
            if {"dynamic_clim", "dynamic_forecast"} <= set(p_):
                fc_gain.append(p_["dynamic_forecast"]["combined"] / p_["dynamic_clim"]["combined"] - 1)
            if {"dynamic_clim", "dynamic_oracle"} <= set(p_):
                oracle_gain.append(p_["dynamic_oracle"]["combined"] / p_["dynamic_clim"]["combined"] - 1)
        if gains:
            rng = lambda xs: f"{min(xs):+.1%}" if len(xs) == 1 or abs(max(xs) - min(xs)) < 5e-4 else f"{min(xs):+.1%} 至 {max(xs):+.1%}"  # noqa: E731
            st.caption(f"全量 10000 目标的实验结果（`scripts/dynamic_experiment.py`，规划期同期的真实天气）：每天重排比一次排完"
                       f" {rng(gains)}" + (f"；再用真实存档预报 {rng(fc_gain)}" if fc_gain else "")
                       + (f"；完美预报（参照）{rng(oracle_gain)}" if oracle_gain else "") + "。")
    c = st.columns([1, 1, 2])
    year = c[0].selectbox("用哪一年的真实天气", list(range(2016, 2026)), index=8, key="dyn_year")
    c[1].markdown("&nbsp;")
    go_sim = c[1].button("逐日模拟", type="primary", key="dyn_go")
    c[2].caption("按当前计划的场景（目标、站点、规划期、规则），把“一次排完、照单执行”和“每天按执行结果重排”各模拟一遍。演示模式几秒；全量约 3 分钟。"
                 "气候统计用其余 9 年，不含被检验的这一年。首次使用需要联网读取历史云量。")
    if go_sim:
        bar = st.progress(0.0, text="准备……")
        res = simulate_days(pb, int(year), progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
        bar.empty()
        st.session_state["dyn_res"] = (run.run_id, int(year), res)
    got = st.session_state.get("dyn_res")
    if got and got[0] == run.run_id:
        res = got[2]
        a, b = res["static"], res["dynamic_clim"]
        k = st.columns(3)
        k[0].metric("成功覆盖（每天重排）", f"{b['covered']:.0f}", f"{b['covered'] - a['covered']:+.0f} 对比一次排完")
        k[1].metric("实际得分（每天重排）", f"{b['score']:.0f}", f"{b['score'] - a['score']:+.0f}")
        k[2].metric("综合指标提升", f"{b['combined'] / max(a['combined'], 1e-9) - 1:+.1%}")
        days = [f"第 {d + 1} 天" for d in range(res["days"])]
        ldf = pd.concat([pd.DataFrame({"天": days, "累计成功覆盖": x["daily"]["covered"], "当天尝试": x["daily"]["attempts"],
                                       "当天失败": x["daily"]["failed"], "策略": x["name"]}) for x in (a, b)])
        fig = px.line(ldf, x="天", y="累计成功覆盖", color="策略", markers=True, color_discrete_sequence=["#8A93A3", BLUE],
                      hover_data={"当天尝试": True, "当天失败": True},
                      title=f"{got[1]} 年真实天气下逐日累计成功覆盖的目标数")
        fig.update_layout(height=380, xaxis_title="", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ 校验与导出
with tabs[6]:
    if run.validation is None:
        st.write("用 Skyfield（评审参考标准）对每个任务逐秒重算仰角、站点太阳角和目标受照，并复算得分。")
        if st.button("开始校验（全量约 2 分钟）", type="primary"):
            with st.spinner("逐秒校验中……"):
                run = validate_run(pb, run)
                run.save(RUNS)
            st.rerun()
    else:
        v = run.validation
        if v.get("geometry_checked") is False:
            st.error("独立几何校验未完成或结构检查失败，这份计划不能作为合法结果。")
        elif v["legal"]:
            st.success(f"全部 {m['tasks']} 个任务通过 Skyfield 逐秒校验。")
        else:
            st.error(f"发现违规：{v['violations']}")
        mg = v["margins"]
        if len(mg) == 3 and all(value is not None and np.isfinite(value) for value in mg):
            st.write(f"最小余量：仰角下限 {mg[0]:.3f}°，仰角上限 {mg[1]:.3f}°，太阳角 {mg[2]:.3f}°")
        else:
            st.caption("未取得完整的几何余量。")
        if not v["legal"] and st.button("重试独立校验（保留原计划）", key="retry_validation"):
            retry_started = perf_counter()
            try:
                with st.spinner("使用原计划重新逐秒校验……"):
                    run = validate_run(pb, run)
                    run.notes.pop("validation_error", None)
                    run.notes["revalidation_time_s"] = perf_counter() - retry_started
                    run.save(RUNS)
            except Exception as exc:
                st.error(f"校验仍未完成：{type(exc).__name__}。原计划已保留，可稍后重试。")
            else:
                st.rerun()
    c1, c2, c3 = st.columns(3)
    c1.download_button("下载观测计划（CSV）", csv_bytes(pb, run), file_name=f"plan_{run.run_id}.csv", mime="text/csv",
                       use_container_width=True)
    c2.download_button("下载目标状态表（CSV）", status_csv_bytes(status_rows(pb, run)),
                       file_name=f"target_status_{run.run_id}.csv", mime="text/csv", use_container_width=True)
    c3.download_button("下载结果摘要（JSON）", summary_json(run), file_name=f"summary_{run.run_id}.json",
                       mime="application/json", use_container_width=True)

# ------------------------------------------------------------------ 结果对比
with tabs[7]:
    recs = []
    for p in runs:
        try:
            r = Run.load(p)
        except Exception:
            continue
        rc = r.cfg
        recs.append({
            "名称": r.notes.get("label", r.run_id), "算法": r.algo, "模式": r.notes.get("mode", ""),
            "范围": "全量" if not r.subset else f"演示 {r.subset['target_frac']:.0%}×{len(r.subset['stations'])}站",
            "规划期": rc["start_utc"][:10], "月光规避": f"{rc.get('moon_sep_deg', 0):g}°" if rc.get("moon_sep_deg") else "—",
            "按天气规划": "是" if r.notes.get("weather_aware") else "—",
            "覆盖": r.metrics["covered"], "得分": round(r.metrics["score"], 1), "综合": round(r.metrics["combined"], 4),
            "用时(秒)": round(r.runtime_s), "校验": "—" if r.validation is None else ("合法" if r.validation["legal"] else "有违规"),
        })
    if not recs:
        st.info("还没有保存的结果。")
    else:
        cdf = pd.DataFrame(recs)
        st.dataframe(cdf, hide_index=True, use_container_width=True)
        full = cdf[cdf["范围"] == "全量"]
        if len(full) > 1:
            c1, c2 = st.columns(2)
            fig = px.bar(full, x="名称", y="覆盖", color_discrete_sequence=[BLUE], title="覆盖目标数（全量）")
            fig.update_layout(height=360, xaxis_title="")
            c1.plotly_chart(fig, use_container_width=True)
            fig = px.bar(full, x="名称", y="得分", color_discrete_sequence=[ORANGE], title="累计得分（全量）")
            fig.update_layout(height=360, xaxis_title="")
            c2.plotly_chart(fig, use_container_width=True)
        st.caption("综合 = w × 覆盖/10000 + (1 − w) × 得分/30062（内部比较口径），不同覆盖权重的运行之间不宜直接比较综合值。")

# ------------------------------------------------------------------ 规则说明
with tabs[8]:
    st.markdown(f"""
| 项目 | 本次设置 | 依据 |
|---|---|---|
| 规划期 | {cfg.start_utc[:10]} 起 {cfg.days:g} 天（UTC） | 企业老师 2026-10-21 确认为 8 月 1–7 日；时间制见答疑 #42 |
| 仰角 | {cfg.elev_min:g}°–{cfg.elev_max:g}°（规划时各留 {cfg.elev_margin:g}° 余量） | 站址文件、答疑 #26 |
| 站点太阳高度角 | < {cfg.sun_max:g}°（留 {cfg.sun_margin:g}° 余量） | 答疑 #26 |
| 目标受照 | {'须被太阳照亮（地影边界留 %g km 余量）' % cfg.sunlit_margin_km if cfg.use_sunlit else '不考虑'} | 答疑 #26、#36、#44 |
| 观测时长与计分 | 120–240 秒，60%→100% 线性 | 答疑 #30 |
| 切换 | {cfg.switch_s} 秒；每站同一时刻一个目标 | 说明.docx |
| 重复观测 | 上次结束 {cfg.repeat_gap_h:g} 小时后可再计分 | 答疑 #30、#32 |
| 月光规避（加分项） | {'月亮在地平线以上时，目标与月亮角距 ≥ %g°' % cfg.moon_sep_deg if cfg.moon_sep_deg > 0 else '不考虑'} | 说明.docx 加分项 |
| 综合指标 | {cfg.w_cov:g} × 覆盖数/10000 + {cfg.w_score:g} × 得分/30062（内部比较口径） | 答疑 #36 |
""")
