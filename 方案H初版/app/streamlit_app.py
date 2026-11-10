"""空间目标观测规划 · 产品界面

运行（在 方案H初版/ 下）：
    uv sync --extra app
    uv run streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

SCHEME_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCHEME_DIR))

from obsplan.app_backend import (ALGOS, CLOUD_THRESHOLD, Run, csv_bytes, explain_target, list_runs,  # noqa: E402
                                 load_problem, orbit_class, pass_profile, replan_changed, replan_run, replay_weather,
                                 run_name, run_planning, save_replan, schedule_csv_bytes, scope_metrics, simulate_days,
                                 station_nights, structural_check, subset_problem, summary_json, task_rows,
                                 validate_replan, validate_run, weather_stats, status_csv_bytes, status_rows,
                                 optimality_gap)
from obsplan.replan import KINDS, MODES, STATUS_ORDER, Event, busiest_station, suggest_targets  # noqa: E402
from obsplan.target_status import STATUS, describe, summarize  # noqa: E402
from obsplan.config import Config  # noqa: E402

RUNS = SCHEME_DIR / "outputs" / "app_runs"
BLUE, ORANGE, AQUA, NAVY = "#2a78d6", "#eb6834", "#1baf7a", "#14213D"
PRI_COLORS = {"高": ORANGE, "中": BLUE, "低": AQUA}
CHANGE_COLORS = {"冻结": "#b9bfca", "保留": BLUE, "移动": ORANGE, "删除": "#d1495b", "新增": AQUA}

st.set_page_config(page_title="空间目标观测规划", page_icon="🛰️", layout="wide")


@st.cache_resource(show_spinner="正在读取数据并计算可见窗口（首次约 30 秒）……")
def base(cfg_json: str):
    cfg = Config(**json.loads(cfg_json))
    ds, pb, timing = load_problem(cfg)
    return cfg, ds, pb, orbit_class(ds), timing


def cfg_key(cfg: Config) -> str:
    return cfg.to_json()


def problem_for(run: Run):
    cfg = Config(**run.cfg)
    cfg_, ds, pb, orbit, _ = base(cfg_key(cfg))
    if run.subset:
        pb = subset_problem(pb, run.subset["target_frac"], run.subset["stations"], run.subset.get("seed", 0))
    return cfg_, ds, pb, orbit


# ------------------------------------------------------------------ 侧栏：选择结果 / 新建规划
st.sidebar.title("🛰️ 观测规划")
runs = list_runs(RUNS)
labels = {}
for p in runs:
    try:
        r = Run.load(p)
        m = r.metrics
        labels[p] = f"{run_name(r)} · 覆盖 {m['covered']} · 得分 {m['score']:.0f}" + (" · 演示" if r.subset else "")
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
    d0 = st.date_input("规划开始（UTC 0 点）", value=date(2026, 8, 1))
    days = st.number_input("规划天数", 1.0, 7.0, 6.0, 1.0)
    c1, c2 = st.columns(2)
    elev_min = c1.number_input("最低仰角 °", 0.0, 30.0, 10.0, 1.0)
    elev_max = c2.number_input("最高仰角 °", 60.0, 90.0, 85.0, 1.0)
    sun_max = st.number_input("站点太阳高度角上限 °", -18.0, 0.0, -6.0, 1.0)
    use_sunlit = st.checkbox("目标须被太阳照亮", value=True)
    moon = st.slider("月光规避角 °（0 = 不考虑，加分项）", 0, 45, 0, 5)
    w_cov = st.slider("覆盖数权重（其余给得分）", 0.0, 1.0, 0.5, 0.05)
    algo = st.radio("算法", list(ALGOS), format_func=lambda k: ALGOS[k])
    mode = st.radio("模式", ["快速（全量约 40 秒）", "高质量（全量约 1 分钟）"], horizontal=False)
    fast = mode.startswith("快速")
    iters = 150 if fast else 300
    weather = st.checkbox("按历史天气做期望价值规划（加分项，仅 H）", value=False, disabled=algo != "H")
    evolved = st.checkbox("用伐谋演化的放置规则（仅 H，推荐）", value=True, disabled=algo != "H" or weather,
                          help="价格协调之后按伐谋演化、并已通过全量复评与 Skyfield 校验的规则放置任务；"
                               "关闭则用人工的「价值 − 价格」。本赛题数据上综合指标高约 0.8%。"
                               "勾选时人工规则也会同时排一份，自动取综合高的（稳健模式），换一份数据也不会比人工规则差，用时基本不变")
    refine_on = st.checkbox("按秒精修（推荐）", value=True,
                            help="网格计划之后，把每个任务的起止按秒外推到真实可见边界，用线性规划重新分配观测时长；全量约多 3 秒")
    st.markdown("**演示模式**（缩小规模，几秒出结果）")
    frac = st.slider("目标比例", 0.05, 1.0, 1.0, 0.05)
    stations = st.multiselect("站点", list(range(1, 11)), default=list(range(1, 11)))
    go_btn = st.button("运行规划", type="primary", use_container_width=True)

if go_btn:
    cfg = Config(start_utc=f"{d0:%Y-%m-%d}T00:00:00", days=float(days), elev_min=elev_min, elev_max=elev_max,
                 sun_max=sun_max, use_sunlit=use_sunlit, w_cov=w_cov, w_score=1 - w_cov, moon_sep_deg=float(moon))
    st.info("正在运行新的规划，完成后这里自动换成新结果（下方变灰的内容是上一份结果）。")
    cfg_, ds, pb, orbit, _ = base(cfg_key(cfg))
    subset = {}
    if frac < 1 or len(stations) < 10:
        subset = {"target_frac": float(frac), "stations": [s - 1 for s in stations], "seed": 0}
        pb = subset_problem(pb, subset["target_frac"], subset["stations"], 0)
    bars = [st.progress(0.0, text="开始规划……"), st.sidebar.progress(0.0, text="开始规划……")]   # 主区域和侧栏各一条

    def tick(f, t):
        for b in bars:
            b.progress(min(1.0, f), text=t)

    run = run_planning(pb, algo, iters=iters, progress=tick, weather=weather and algo == "H", fast=fast,
                       refine_seconds=refine_on, full_problem=not subset, evolved_rule=evolved)
    run.subset = subset
    tick(1.0, f"完成，用时 {run.runtime_s:.0f} 秒")
    run.save(RUNS)
    st.session_state["selected"] = run.run_id
    st.rerun()

if choice is None:
    st.title("空间目标观测规划")
    st.info("左侧「新建规划」运行一次，结果会保存在这里。全量 10000 颗目标快速模式约 40 秒、高质量模式约 1 分钟（首次另需约半分钟计算可见窗口）；演示模式几秒钟。")
    st.stop()

run = Run.load(choice)
cfg, ds, pb, orbit = problem_for(run)
m = run.metrics
sc = scope_metrics(cfg, ds, run)
in_scope = sc["mask"]              # 演示结果：参与规划的目标；全量为 None
n_scope = sc["targets"]
hi_all = ds.priority == max(cfg.priority_score.values())
pri_total = {lab: int((np.array(ds.priority_label) == lab)[in_scope if in_scope is not None else slice(None)].sum())
             for lab in ("高", "中", "低")}


@st.cache_data(show_spinner="整理任务明细与目标状态……", max_entries=8)
def run_tables(run_id: str, _pb, _run, _orbit):
    """一份结果的任务明细与目标状态表（按结果编号缓存，切换页签不重复计算），以及各自用时。"""
    t0 = time.time()
    tdf = pd.DataFrame(task_rows(_pb, _run, _orbit))
    t1 = time.time()
    srows_ = status_rows(_pb, _run)
    return tdf, srows_, {"rows_s": t1 - t0, "status_s": time.time() - t1}


@st.cache_data(show_spinner=False, max_entries=8)
def run_structural(run_id: str, _pb, _run):
    t0 = time.time()
    return structural_check(_pb, _run), time.time() - t0


df, srows, table_timing = run_tables(run.run_id, pb, run, orbit)
if in_scope is not None and len(srows) == len(in_scope):
    srows = [r for r, keep in zip(srows, in_scope) if keep]      # 演示结果只统计参与规划的目标


def goto(name: str):
    """页签之间跳转（按钮回调）。"""
    st.session_state["page"] = name


def links(names, key: str):
    row = st.container(horizontal=True, gap="small")
    for n in names:
        row.button(f"查看「{n}」→", key=f"go_{key}_{n}", on_click=goto, args=(n,), type="tertiary")


def fmt_s(x) -> str:
    if x is None:
        return "—"
    return f"{x * 1000:.0f} 毫秒" if x < 1 else (f"{x:.1f} 秒" if x < 90 else f"{x / 60:.1f} 分钟")


def utc_of(seconds) -> datetime:
    return (cfg.start_dt + timedelta(seconds=float(seconds))).replace(tzinfo=None)


st.title("空间目标观测规划")
scope = "全量" if not run.subset else (f"演示：{run.subset['target_frac']:.0%} 目标（{n_scope:,} 颗）× "
                                         f"{len(run.subset['stations'])} 站")
extra = (f" · 月光规避 {cfg.moon_sep_deg:g}°" if cfg.moon_sep_deg > 0 else "") + \
    (" · 按历史天气期望规划" if run.notes.get("weather_aware") else "") + \
    (" · 按秒精修" if run.notes.get("refine", {}).get("refined") else "")
st.caption(f"{ALGOS.get(run.algo, run.algo)} · {cfg.start_utc[:10]} 起 {cfg.days:g} 天（UTC）· {scope} · "
           f"仰角 {cfg.elev_min:g}°–{cfg.elev_max:g}° · 太阳角 < {cfg.sun_max:g}° · 目标受照 {'是' if cfg.use_sunlit else '否'}{extra}")

# ------------------------------------------------------------------ 概览
def page_overview():
    k = st.columns(6)
    v = run.validation
    gap = optimality_gap(cfg, run)
    tiles = [
        ("覆盖目标", f"{m['covered']}", f"可观测 {(pb.max_obs > 0).sum()} / " +
         (f"全部 {ds.n_targets}" if not run.subset else f"本次参与 {n_scope}")),
        ("累计得分", f"{m['score']:.0f}", "含 48 小时后的重复观测"),
        ("综合指标", f"{sc['combined']:.4f}", (f"离可证明的上界 {gap['ub']:.4f} 最多差 {gap['gap_pct']:.1f}%（{gap['label']}）"
                                              if gap else f"覆盖权重 {cfg.w_cov:g}（内部口径）") if not run.subset else
         f"按本次参与的 {n_scope} 颗目标归一" + ("；目标少、站点有富余，重复观测多，所以超过 1" if sc["combined"] > 1 else "")),
        ("观测任务", f"{m['tasks']}", f"其中重复观测 {m['repeat_tasks']}"),
        ("求解用时", f"{run.runtime_s:.0f} 秒", "不含窗口预处理"),
        ("Skyfield 校验", "未校验" if v is None else ("全部合法" if v["legal"] else "有违规"),
         "到「校验与导出」运行" if v is None else f"违规 {sum(v['violations'].values())} 项"),
    ]
    for col, (label, value, note) in zip(k, tiles):
        col.metric(label, value)
        col.caption(note)
    c1, c2, c3 = st.columns(3)
    pri = pd.DataFrame({"优先级": ["高", "中", "低"],
                        "已覆盖": [m["covered_高"], m["covered_中"], m["covered_低"]],
                        "未覆盖": [pri_total[lab] - m[f"covered_{lab}"] for lab in ("高", "中", "低")]})
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
    ssum = summarize(srows)
    st.write(describe(ssum))
    cnt = pd.DataFrame(ssum["by_priority"]).T
    cnt = cnt.loc[:, (cnt != 0).any(axis=0)]
    cnt.index = [f"{p}优先级" for p in cnt.index]
    st.dataframe(cnt, use_container_width=True)
    pick = st.selectbox("查看明细", ["未覆盖与未观测满"] + list(STATUS.values()), key="status_pick", persist_state="page")
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
def page_map():
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
def page_gantt():
    nights = station_nights(pb)
    c1, c2, c3 = st.columns([1, 2, 1])
    s_sel = c1.selectbox("站点", list(range(1, pb.n_stations + 1)), format_func=lambda s: f"{s}号站",
                         key="gantt_station", persist_state="page")
    ns = [(n, a, b) for (s, n, a, b) in nights if s == s_sel - 1]
    if not ns:
        st.info("该站在规划期内没有黑夜时段。")
    else:
        n_sel = c2.selectbox("夜晚", ns, format_func=lambda x: f"第 {x[0] + 1} 夜：{cfg.start_dt + timedelta(seconds=x[1] * cfg.cell_s):%m-%d %H:%M} – "
                                                          f"{cfg.start_dt + timedelta(seconds=x[2] * cfg.cell_s):%m-%d %H:%M} UTC")
        color_by = c3.radio("颜色", ["优先级", "轨道"], horizontal=True, key="gantt_color", persist_state="page")
        a = cfg.start_dt + timedelta(seconds=n_sel[1] * cfg.cell_s)
        b = cfg.start_dt + timedelta(seconds=n_sel[2] * cfg.cell_s)
        part = df[(df["站点"] == s_sel) & (df["开始"] >= a) & (df["结束"] <= b)] if len(df) else df
        st.caption(f"这一夜安排了 {len(part)} 个任务，占用约 {(part['时长(秒)'].sum() + 30 * len(part)) / 60:.0f} 分钟 / "
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
def page_price():
    if run.mu is None:
        st.info("时间价格只有价格协调方案（H）才有。")
    else:
        mu = run.mu.astype(float)
        day = st.slider("查看第几天", 1, int(cfg.days), 2, key="price_day", persist_state="page")
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
def page_explain():
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
def page_weather():
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
    year = c[0].selectbox("用哪一年的真实天气", list(range(2016, 2026)), index=8, key="dyn_year", persist_state="page")
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
def skyfield_button(key: str):
    """校验当前结果（Skyfield 逐秒），结果写回结果文件。"""
    if st.button("开始 Skyfield 校验（全量约 1.5 分钟）", type="primary", key=key):
        with st.spinner("逐秒校验中……"):
            validate_run(pb, run).save(RUNS)
        st.rerun()


def download_buttons(key: str):
    c1, c2, c3 = st.columns(3)
    c1.download_button("下载观测计划（CSV）", lambda: csv_bytes(pb, run), file_name=f"plan_{run.run_id}.csv",
                       mime="text/csv", use_container_width=True, key=f"{key}_plan")
    c2.download_button("下载目标状态表（CSV）", lambda: status_csv_bytes(srows),
                       file_name=f"target_status_{run.run_id}.csv", mime="text/csv", use_container_width=True,
                       key=f"{key}_status")
    c3.download_button("下载结果摘要（JSON）", summary_json(run), file_name=f"summary_{run.run_id}.json",
                       mime="application/json", use_container_width=True, key=f"{key}_summary")


def page_export():
    if run.validation is None:
        st.write("用 Skyfield（评审参考标准）对每个任务逐秒重算仰角、站点太阳角和目标受照，并复算得分。")
        skyfield_button("validate_export")
    else:
        v = run.validation
        if v["legal"]:
            st.success(f"全部 {m['tasks']} 个任务通过 Skyfield 逐秒校验。")
        else:
            st.error(f"发现违规：{v['violations']}")
        mg = v["margins"]
        st.write(f"最小余量：仰角下限 {mg[0]:.3f}°，仰角上限 {mg[1]:.3f}°，太阳角 {mg[2]:.3f}°")
    download_buttons("export")

# ------------------------------------------------------------------ 结果对比
def page_compare():
    recs = []
    for p in runs:
        try:
            r = Run.load(p)
        except Exception:
            continue
        rc = r.cfg
        rpn = r.notes.get("replan")
        try:
            comb = scope_metrics(Config(**rc), ds, r)["combined"]     # 演示结果按参与目标归一
        except Exception:
            comb = r.metrics["combined"]
        recs.append({
            "名称": run_name(r), "算法": r.algo,
            "模式": " · ".join(x for x in (r.notes.get("mode", ""), f"重排（{MODES.get(rpn['mode'], rpn['mode'])}）" if rpn else "") if x),
            "范围": "全量" if not r.subset else f"演示 {r.subset['target_frac']:.0%}×{len(r.subset['stations'])}站",
            "规划期": rc["start_utc"][:10], "月光规避": f"{rc.get('moon_sep_deg', 0):g}°" if rc.get("moon_sep_deg") else "—",
            "按天气规划": "是" if r.notes.get("weather_aware") else "—",
            "覆盖": r.metrics["covered"], "得分": round(r.metrics["score"], 1), "综合": round(comb, 4),
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
        st.caption("综合 = w × 覆盖/10000 + (1 − w) × 得分/30062（内部比较口径），不同覆盖权重的运行之间不宜直接比较综合值；"
                   "演示结果只排了一部分目标，综合值按参与的目标归一，只适合和同一演示范围的结果比较"
                   "（满分参考按每颗目标满分观测一次计，目标少、重复观测多时会超过 1）。"
                   "重排结果的用时是重排本身的用时。")

# ------------------------------------------------------------------ 规则说明
def page_rules():
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

# ------------------------------------------------------------------ 突发重规划（产品演示与加分项）
@st.cache_data(show_spinner=False, max_entries=16)
def replan_candidates(run_id: str, kind: str, t_s: int, _pb, _run) -> list[int]:
    return suggest_targets(_pb, _run.schedule(), t_s, kind)


def target_label(x: int) -> str:
    n = int((run.target == x).sum())
    return f"{ds.satnum[x]} · {ds.names[x] or '无名称'} · {ds.priority_label[x]}优先级 · 原计划观测 {n} 次"


def default_event(t_s: int, hours: float = 6.0) -> Event:
    """示例事件：事件时刻之后 hours 小时内任务最多的站点停机。"""
    return Event("down", int(t_s), station=busiest_station(run.schedule(), pb.n_stations, t_s, hours), hours=float(hours))


def run_replan(ev: Event, mode: str) -> dict:
    bar = st.progress(0.0, text="准备重排……")
    res = replan_run(pb, run, ev, mode=mode, progress=lambda f, t: bar.progress(min(1.0, f), text=t))
    bar.empty()
    n = st.session_state.get("replan", {}).get("n", 0) + 1
    st.session_state["replan"] = {"run_id": run.run_id, "res": res, "n": n}
    return res


def current_replan():
    got = st.session_state.get("replan")
    return got if got and got["run_id"] == run.run_id else None


def obs_text(obs) -> str:
    return "、".join(f"{s + 1}号站 {utc_of(a):%m-%d %H:%M:%S} 起 {d} 秒" for s, a, d in obs) or "无"


def show_replan(got: dict):
    res, rid = got["res"], got["n"]
    ev, d = res["event"], res["diff"]
    b, n0, a = res["before"], res["no_replan"], res["after"]
    old, new = run.schedule(), res["schedule"]
    o_st, n_st = res["old_status"], res["new_status"]
    st.success(f"**{res['describe']}** · {MODES[res['mode']]} · 用时 {res['seconds']:.0f} 秒")
    base_name = "只取消受影响的任务" if ev.kind == "down" else "原计划"
    k = st.columns(4)
    k[0].metric("覆盖目标", f"{a['covered']}", f"{a['covered'] - n0['covered']:+d}（对比{base_name}）")
    k[0].caption(f"原计划 {b['covered']}" + (f" → 只取消 {n0['covered']}" if ev.kind == "down" else "") + f" → 重排后 {a['covered']}")
    k[1].metric("累计得分", f"{a['score']:.0f}", f"{a['score'] - n0['score']:+.0f}")
    k[1].caption(f"原计划 {b['score']:.0f}" + (f" → 只取消 {n0['score']:.0f}" if ev.kind == "down" else "")
                 + f" → 重排后 {a['score']:.0f}" + ("（按新优先级计分）" if ev.kind == "priority" else ""))
    k[2].metric("综合指标", f"{a['combined']:.4f}", f"{a['combined'] - n0['combined']:+.4f}")
    k[2].caption("0.5 × 覆盖/目标数 + 0.5 × 得分/满分（内部口径）")
    k[3].metric("原计划任务保留", f"{d['保留'] / max(d['原计划 t 之后'], 1):.1%}")
    k[3].caption(f"事件之后原有 {d['原计划 t 之后']} 个任务，保留 {d['保留']} 个")

    st.markdown("**对原计划的扰动**（只看事件之后的任务；同一站、开始和结束都相差不超过 1 分钟算保留）")
    k = st.columns(5)
    k[0].metric("冻结", d["冻结"])
    k[0].caption("事件前已开始的任务，原样不动")
    k[1].metric("保留", d["保留"])
    k[1].caption(f"其中 {d['完全不变']} 个起止时刻完全不变，其余按秒微调")
    k[2].metric("移动", d["移动"])
    k[2].caption("同一目标换了站点、时间或时长")
    k[3].metric("删除", d["删除"])
    k[3].caption(f"其中 {d['停机取消'] + d['停机中断'] - d['取消后改到别处']} 个因停机取消、没能改到别处" if ev.kind == "down"
                 else "为腾出时间去掉的低价值观测")
    k[4].metric("新增", d["新增"])
    k[4].caption("原计划里没有的观测")
    if ev.kind == "down":
        st.write(f"停机时段内 {ev.station + 1} 号站原有 {d['停机取消']} 个任务取消"
                 + (f"，另有 {d['停机中断']} 个正在执行的任务中断" if d["停机中断"] else "")
                 + f"；其中 {d['取消后改到别处']} 个改到别的站或别的时间观测（计入移动），其余计入删除。")
    elif "target" in res:
        x, ti = ev.target, res["target"]
        bo = [o for o in ti["before"] if o[1] >= ev.t_s]
        ao = [o for o in ti["after"] if o[1] >= ev.t_s]
        if ev.kind == "priority":
            st.write(f"目标 {ds.satnum[x]} 升为高优先级：事件之后原计划观测 {len(bo)} 次（{sum(o[2] for o in bo)} 秒）"
                     f"→ 重排后 {len(ao)} 次（{sum(o[2] for o in ao)} 秒）。重排后：{obs_text(ao)}。")
            if len(ao) <= len(bo) and sum(o[2] for o in ao) <= sum(o[2] for o in bo):
                st.caption("没有找到更好的安排：48 小时后的可见时段已被价值更高的任务占满，或剩余时间不够再观测一次。")
        elif ti.get("ok"):
            st.write(f"必须观测的目标 {ds.satnum[x]} 已安排：{obs_text(ao)}。")
        else:
            why = ("已观测满 3 次" if ti["frozen_obs"] >= 3 else
                   "事件之后没有可见窗口（或与之前的观测间隔不足 48 小时）" if ti["windows_after_t"] == 0 else
                   "可见时段都被不能调整的任务占用")
            st.warning(f"目标 {ds.satnum[x]} 没能安排：{why}。")

    st.markdown("**受影响站点的甘特图对比**")
    chg_o = np.isin(o_st, ["移动", "删除"])
    chg_n = np.isin(n_st, ["移动", "新增"])
    per = (np.bincount(old.station[chg_o], minlength=pb.n_stations)
           + np.bincount(new.station[chg_n], minlength=pb.n_stations))
    affected = [int(s) for s in np.argsort(-per, kind="stable") if per[s] > 0]
    if ev.kind == "down":
        affected = [ev.station] + [s for s in affected if s != ev.station]
    t0d, t1d = utc_of(0), utc_of(cfg.days * 86400)
    if ev.kind == "down":
        lo, hi = utc_of(ev.t_s - 3600), utc_of(ev.down[1] + 3 * 3600)
    else:
        times = np.concatenate([old.start_s[chg_o], new.start_s[chg_n]])
        c0 = float(times.min()) if len(times) else float(ev.t_s)
        lo, hi = utc_of(c0 - 2 * 3600), utc_of(c0 + 4 * 3600)
    lo, hi = max(lo, t0d), min(hi, t1d)
    c1, c2 = st.columns([2, 3])
    stations = c1.multiselect("对比的站点", list(range(pb.n_stations)), default=affected[:3],
                              format_func=lambda s: f"{s + 1}号站（改动 {per[s]}）", key=f"rp_gst_{rid}")
    rng = c2.slider("时间范围（UTC）", min_value=t0d, max_value=t1d, value=(lo, hi), step=timedelta(minutes=30),
                    format="MM-DD HH:mm", key=f"rp_grange_{rid}")
    a_s, b_s = (rng[0] - t0d).total_seconds(), (rng[1] - t0d).total_seconds()
    rows = []
    for plan, sch, stt in (("原计划", old, o_st), ("新计划", new, n_st)):
        e_ = sch.start_s + sch.dur_s
        for i in np.nonzero(np.isin(sch.station, stations) & (sch.start_s < b_s) & (e_ > a_s))[0]:
            x = int(sch.target[i])
            rows.append({"行": f"{int(sch.station[i]) + 1}号站 · {plan}", "开始": utc_of(sch.start_s[i]),
                         "结束": utc_of(e_[i]), "状态": stt[i], "目标编号": int(ds.satnum[x]),
                         "优先级": res["ds"].priority_label[x], "时长(秒)": int(sch.dur_s[i])})
    if rows:
        order = [f"{s + 1}号站 · {p}" for s in stations for p in ("原计划", "新计划")]
        gdf = pd.DataFrame(rows)
        fig = px.timeline(gdf, x_start="开始", x_end="结束", y="行", color="状态", color_discrete_map=CHANGE_COLORS,
                          category_orders={"行": order, "状态": STATUS_ORDER},
                          hover_data=["目标编号", "优先级", "时长(秒)"])
        fig.add_vline(x=utc_of(ev.t_s), line_dash="dash", line_color=NAVY, line_width=1.5)
        fig.add_annotation(x=utc_of(ev.t_s), y=1.0, yref="paper", text="事件时刻", showarrow=False, yanchor="bottom",
                           font=dict(color=NAVY))
        if ev.kind == "down" and ev.station in stations:
            # 只标在停机站的两行：类别轴上的位置按图里实际的类别顺序取（px.timeline 会把顺序倒过来，第一行在最上面）
            cats = list(fig.layout.yaxis.categoryarray or order)
            pos = [cats.index(f"{ev.station + 1}号站 · {p}") for p in ("原计划", "新计划")]
            fig.add_shape(type="rect", x0=utc_of(ev.down[0]), x1=utc_of(ev.down[1]), y0=min(pos) - 0.5,
                          y1=max(pos) + 0.5, xref="x", yref="y", fillcolor="#d1495b", opacity=0.08, line_width=0,
                          layer="below")
            fig.add_annotation(x=utc_of(min(ev.down[1], ev.down[0] + 1800)), y=1.0, yref="paper", xanchor="left",
                               text=f"{ev.station + 1}号站停机", showarrow=False, yanchor="bottom",
                               font=dict(color="#d1495b"))
        fig.update_layout(height=110 + 46 * len(order), xaxis_title="UTC", yaxis_title="", legend_title_text="",
                          margin=dict(t=30, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.25 if len(order) > 4 else -0.4))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("每个站上下两行：原计划与新计划。灰色为事件前已执行（冻结），蓝色保留，橙色移动，红色删除，绿色新增；"
                   "停机时段用淡红色标出。可以拖动选择范围放大。")
    else:
        st.info("所选站点在这个时间范围内没有任务。")

    st.markdown("**约束校验**")
    viol = res["violations"]
    k = st.columns(3 if ev.kind == "down" else 2)
    k[0].metric("结构检查", "全部合法" if not viol else f"{sum(len(v) for v in viol.values())} 项违规")
    k[0].caption("时长 120–240 秒、同站切换 30 秒、同一目标间隔 48 小时、任务都在可见窗口内（obsplan/checks.py）")
    if viol:
        st.error({kk: len(vv) for kk, vv in viol.items()})
    k[1].metric("冻结部分", "未改动" if res["frozen_unchanged"] else "有改动")
    k[1].caption(f"事件前已开始的 {d['冻结']} 个任务与原计划逐一相同")
    if ev.kind == "down":
        k[2].metric("停机时段内的任务", f"{res['outage_tasks']} 个")
        k[2].caption(f"{ev.station + 1} 号站 {utc_of(ev.down[0]):%m-%d %H:%M} – {utc_of(ev.down[1]):%m-%d %H:%M}")
    changed = replan_changed(res)
    c1, c2 = st.columns([3, 2])
    with c1:
        allk = st.checkbox("同时校验原样保留的任务（较慢）", key=f"rp_valall_{rid}")
        n_chk = int((n_st != "冻结").sum()) if allk else len(changed)
        go_val = st.button(f"Skyfield 逐秒校验{'事件后的全部' if allk else '改动过的'} {n_chk} 个任务", key=f"rp_val_{rid}",
                           type="primary")
    if go_val:
        with st.spinner("Skyfield 逐秒校验中……"):
            st.session_state["replan_val"] = (rid, validate_replan(pb, res, include_kept=allk))
    val = st.session_state.get("replan_val")
    if val and val[0] == rid:
        v = val[1]
        (c2.success if v["legal"] else c2.error)(
            f"Skyfield 校验{v['scope']} {v['tasks']} 个：" + ("全部合法" if v["legal"] else f"违规 {v['violations']}")
            + f"（{v['seconds']:.0f} 秒）")
    else:
        c2.caption("原样保留和冻结的任务与原计划完全相同" + ("，原计划已通过 Skyfield 校验。" if run.validation and run.validation["legal"]
                                                  else "；原计划可在「校验与导出」页签校验。"))

    c1, c2, c3 = st.columns(3)
    c1.download_button("下载重排后的计划（CSV）", lambda: schedule_csv_bytes(cfg, res["ds"], new, f"replan_{run.run_id}"),
                       file_name=f"plan_{run.run_id}_replan.csv", mime="text/csv", use_container_width=True,
                       key=f"rp_dl_{rid}")
    crow = []
    for plan, sch, stt in (("原计划", old, o_st), ("新计划", new, n_st)):
        for i in np.nonzero(np.isin(stt, ["移动", "删除", "新增"]))[0]:
            x = int(sch.target[i])
            crow.append({"状态": stt[i], "计划": plan, "目标编号": int(ds.satnum[x]), "名称": ds.names[x],
                         "优先级": res["ds"].priority_label[x], "站点": int(sch.station[i]) + 1,
                         "开始": f"{utc_of(sch.start_s[i]):%m-%d %H:%M:%S}",
                         "结束": f"{utc_of(sch.start_s[i] + sch.dur_s[i]):%m-%d %H:%M:%S}", "时长(秒)": int(sch.dur_s[i])})
    cdf = pd.DataFrame(crow)
    if len(cdf):
        cdf = cdf.sort_values(["目标编号", "计划", "开始"])
    c2.download_button("下载改动清单（CSV）", ("﻿" + cdf.to_csv(index=False)).encode("utf-8"),
                       file_name=f"changes_{run.run_id}.csv", mime="text/csv", use_container_width=True,
                       key=f"rp_dlc_{rid}")
    if ev.kind == "priority":
        c3.caption("优先级变化后的结果不另存：其他页签按原始优先级计分。")
    elif c3.button("另存为新结果（用其他页签查看）", use_container_width=True, key=f"rp_save_{rid}"):
        new_run = save_replan(run, res, RUNS)
        st.session_state["selected"] = new_run.run_id
        st.rerun()
    if len(cdf):
        with st.expander(f"改动明细（{len(cdf)} 行）"):
            st.dataframe(cdf, hide_index=True, use_container_width=True, height=320)


def page_replan():
    st.markdown("正式赛题没有突发情况（答疑 #32），这里演示系统应对突发事件的闭环：**状态更新 → 动态重规划 → 约束校验**。"
                "选一个时刻和事件：此前已经开始的任务冻结不动，之后的时段用同一套价格协调重排，并统计对原计划的改动。")
    t0d = utc_of(0)
    t1d = utc_of(cfg.days * 86400)
    ev = None
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 3, 2])
        when = c1.slider("事件时刻（UTC）", min_value=t0d, max_value=t1d - timedelta(hours=1),
                         value=t0d + timedelta(hours=min(60.0, cfg.days * 12)), step=timedelta(minutes=10),
                         format="MM-DD HH:mm", key="rp_when", persist_state="page")
        t_s = int((when - t0d).total_seconds())
        kind = c2.radio("事件类型", list(KINDS), format_func=KINDS.get, horizontal=True, key="rp_kind",
                        persist_state="page")
        mode = c3.radio("重排方式", list(MODES), format_func=lambda k_: MODES[k_] + ("（推荐）" if k_ == "minimal" else ""),
                        key="rp_mode", persist_state="page",
                        help="扰动最小：事件之后仍然可行的原任务原样保留，只为受影响的目标挤让少量任务；"
                             "全面重排：事件之后从头规划，九成以上的任务会变，质量也不一定更好"
                             "（在默认的 H+ 结果上实测综合指标反而略低）。")
        c4, c5 = st.columns([3, 2])
        if kind == "down":
            busy = np.bincount(run.station[(run.start_s >= t_s) & (run.start_s < t_s + 6 * 3600)],
                               minlength=pb.n_stations)
            station = c4.selectbox("停机站点", list(range(pb.n_stations)), index=int(busy.argmax()),
                                   format_func=lambda s: f"{s + 1}号站（事件后 6 小时内原有 {busy[s]} 个任务）",
                                   key="rp_station", persist_state="page")
            hours = c5.slider("停机时长（小时）", 1, 24, 6, key="rp_hours", persist_state="page")
            ev = Event("down", t_s, station=int(station), hours=float(hours))
        else:
            cand = replan_candidates(run.run_id, kind, t_s, pb, run)
            tip = ("推荐的是原计划只观测 1 次、48 小时后还有可见窗口的中低优先级目标，升级后有机会多观测一次" if kind == "priority"
                   else "推荐的是事件之后没有安排观测、但还有可见窗口的目标，原计划没覆盖的排在前面")
            x = c4.selectbox("目标（推荐）", cand, format_func=target_label, key=f"rp_target_{kind}", help=tip) if cand else None
            num = c5.text_input("或输入目标编号（NORAD）", key=f"rp_num_{kind}", persist_state="page",
                                help="填了就用这里的编号，留空则用左边选中的目标")
            if num.strip():
                hit = np.nonzero(ds.satnum == int(num))[0] if num.strip().isdigit() else []
                x = int(hit[0]) if len(hit) else None
                if x is None:
                    c5.warning("没有这个编号。")
            if x is not None and kind == "priority" and ds.priority_label[x] == "高":
                c5.warning("这颗目标已经是高优先级。")
                x = None
            if x is None and not cand and not num.strip():
                c4.info("没有合适的推荐目标，请在右侧输入编号。")
            ev = Event(kind, t_s, target=int(x)) if x is not None else None
        go_btn = st.button("执行重排", type="primary", disabled=ev is None, key="rp_go")
        st.caption("全量 10000 颗目标一次重排约 10 秒，演示模式几秒。")
    if go_btn and ev is not None:
        run_replan(ev, mode)
    got = current_replan()
    if got is None:
        st.info("设定事件后点「执行重排」。")
        return
    show_replan(got)


# ------------------------------------------------------------------ 演示：企业赛题的任务链
PIPE_CSS = """<style>
.pipe {display:flex; flex-wrap:wrap; gap:6px; align-items:stretch; margin:2px 0 10px;}
.pipe .s {flex:1 1 0; min-width:104px; border:1px solid #d9dce3; border-radius:8px; padding:7px 9px; background:#ffffff;}
.pipe .s b {display:block; font-size:0.82rem; color:#14213D;}
.pipe .s span {display:block; font-size:1.02rem; font-weight:600; color:#2a78d6; margin-top:2px;}
.pipe .s small {display:block; color:#8A93A3; font-size:0.75rem;}
.pipe .ar {align-self:center; color:#b9bfca; font-size:1.1rem;}
.stepno {display:inline-block; min-width:1.7em; height:1.7em; line-height:1.7em; border-radius:50%; text-align:center;
         background:#14213D; color:#ffffff; font-weight:600; margin-right:0.45em; font-size:0.9rem;}
</style>"""


def step_box(no: int, title: str, did: str, took: str, numbers: list, tabs_: list):
    """演示页的一步：做了什么、用时、关键数字，以及到详细页签的链接。返回右侧容器，供追加按钮等。"""
    with st.container(border=True):
        c1, c2 = st.columns([5, 7], gap="medium")
        c1.markdown(f"<span class='stepno'>{no}</span>**{title}**", unsafe_allow_html=True)
        c1.markdown(did)
        c1.caption(f"用时：{took}")
        with c1:
            links(tabs_, key=f"s{no}")
        cols = c2.columns(len(numbers))
        for col, (lab, val, note) in zip(cols, numbers):
            col.metric(lab, val)
            if note:
                col.caption(note)
        return c2


def page_demo():
    tm = base(cfg_key(cfg))[4]
    t_pri0 = time.time()
    labs = pd.Series(ds.priority_label).value_counts()
    obs_mask = pb.max_obs > 0
    pairs = np.unique(pb.w_tg * pb.n_stations + pb.w_st)              # (目标, 站) 组合
    n_st_t = np.bincount(pairs // pb.n_stations, minlength=pb.n_targets)
    scarce = obs_mask & (n_st_t <= 2)
    hi = hi_all if in_scope is None else hi_all & in_scope
    if run.mu is not None:
        busiest, busiest_note = int(run.mu.astype(float).mean(axis=1).argmax()), "平均价格最高"
    else:                                   # 没有保存价格的结果（重排结果、方案 E / B）：看哪个站排得最满
        dark_s = np.array([sum(b - a for a, b in runs_) * cfg.cell_s / 3600 for runs_ in pb.win.dark])
        used = np.array(m["station_busy_h"]) / np.maximum(dark_s, 1e-9)
        busiest, busiest_note = int(used.argmax()), f"黑夜时间用掉 {used.max():.0%}，各站最高"
    t_pri = time.time() - t_pri0
    sv, sv_t = run_structural(run.run_id, pb, run)
    v = run.validation
    ssum = summarize(srows)
    hi_rows = ssum["by_priority"].get("高", {})
    hi_cov = sum(n for kk, n in hi_rows.items() if kk.startswith("已观测"))
    hi_obs = int((hi & obs_mask).sum())
    rp = current_replan()
    dark_h = sum(b - a for runs_ in pb.win.dark for a, b in runs_) * cfg.cell_s / 3600
    sub_note = "" if not run.subset else f"（演示模式：{run.subset['target_frac']:.0%} 目标 × {len(run.subset['stations'])} 站）"
    t_ev = int(min(60 * 3600, cfg.days * 43200))

    st.markdown(PIPE_CSS, unsafe_allow_html=True)
    st.markdown(f"按企业赛题给出的任务链，把当前结果{sub_note}从读数据到输出走一遍：每一步做了什么、用了多久、关键数字，"
                "点「查看」进入对应的详细页签。")
    chips = [("① 数据读取", f"{ds.n_targets:,} 颗目标", fmt_s(tm["load_s"])),
             ("② 机会分析", f"{int(obs_mask.sum()):,} 颗可观测", fmt_s(tm["windows_s"] + tm["build_s"])),
             ("③ 优先级判断", f"{hi_obs:,} 颗高优先", fmt_s(t_pri)),
             ("④ 任务规划", f"{m['tasks']:,} 个任务", fmt_s(run.runtime_s)),
             ("⑤ 约束校验", "全部合法" if (not sv and v is not None and v["legal"]) else
              ("结构合法" if not sv else "有违规"), fmt_s(sv_t + (v.get("seconds", 0) if v else 0))),
             ("⑥ 状态更新", f"覆盖 {ssum['covered']:,} 颗", fmt_s(table_timing["status_s"])),
             ("⑦ 动态重规划", f"保留 {rp['res']['diff']['保留'] / max(rp['res']['diff']['原计划 t 之后'], 1):.0%} 原任务" if rp
              else "待演示", fmt_s(rp["res"]["seconds"]) if rp else "约 10 秒"),
             ("⑧ 结果输出", "CSV / JSON", "点击即下载")]
    html = "<div class='pipe'>" + "<span class='ar'>›</span>".join(
        f"<div class='s'><b>{a}</b><span>{b}</span><small>{c}</small></div>" for a, b, c in chips) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

    step_box(1, "任务数据读取",
             "读取每颗目标的轨道根数（TLE：两行数字描述一颗卫星的轨道，是推算位置的输入）、10 个地面站的经纬度与海拔、"
             "高/中/低优先级清单，并检查 TLE 校验位、编号与优先级清单是否一一对应。",
             fmt_s(tm["load_s"]) + "（本次启动时读取）",
             [("目标", f"{ds.n_targets:,}", f"高 {labs.get('高', 0)} · 中 {labs.get('中', 0)} · 低 {labs.get('低', 0)}"),
              ("地面站", f"{ds.n_stations}", "经度 / 纬度 / 海拔"),
              ("数据问题", f"{len(ds.issues)} 条", "；".join(ds.issues[:2]) if ds.issues else "校验位、编号均无误")],
             ["站点地图", "规则说明"])

    step_box(2, "观测机会分析",
             f"每 {cfg.cell_s} 秒推算一次 {cfg.days:g} 天内每颗目标的位置（SGP4 轨道外推），逐站判断三件事是否同时满足："
             f"仰角 {cfg.elev_min:g}°–{cfg.elev_max:g}°（目标在天上合适的高度）、站点太阳高度角 < {cfg.sun_max:g}°（站点已天黑）、"
             "目标被太阳照亮（不在地球影子里，才看得见）。连续满足不少于 120 秒的时段就是一个可见窗口。",
             (f"读缓存 {fmt_s(tm['windows_s'])}，建索引 {fmt_s(tm['build_s'])}（首次计算约 30 秒）" if tm["windows_cached"]
              else f"计算 {fmt_s(tm['windows_s'])}，建索引 {fmt_s(tm['build_s'])}"),
             [("可见窗口", f"{len(pb.w_k0):,}", f"平均每颗可观测目标 {len(pb.w_k0) / max(int(obs_mask.sum()), 1):.0f} 个"),
              ("可观测目标", f"{int(obs_mask.sum()):,}", f"外推异常 {int(pb.win.sgp4_failed.sum())} 颗" +
               (f" · 本次参与 {n_scope:,} 颗" if run.subset else "")),
              ("站点黑夜", f"{dark_h:.0f} 小时", "10 站合计，北半球夏季夜短")],
             ["任务解释"])

    step_box(3, "目标优先级判断",
             "高、中、低优先级每次观测分别计 5、3、2 分（观测 240 秒得满分，120 秒得 60%），每覆盖一颗目标另计覆盖分。"
             "价格协调给每个站点每 30 秒定一个“价格”：被越多目标争抢的时段越贵；只有少数站看得到的稀缺目标没有别的选择，"
             "会先拿到时段。",
             fmt_s(t_pri),
             [("高优先级", f"{hi_obs:,}", f"可观测 / {'本次参与' if run.subset else '全部'} {int(hi.sum()):,}"),
              ("稀缺目标", f"{int(scarce.sum()):,}", f"只有 1–2 个站看得到，其中高优先级 {int((scarce & hi).sum())} 颗"),
              ("最拥挤的站", f"{busiest + 1} 号站", busiest_note)],
             ["时间价格", "概览"])

    pick = run.notes.get("rule_pick")
    by_famou = bool(run.notes.get("famou_rule"))
    rule_flag = ["伐谋放置规则"] if by_famou else (["人工放置规则（两条都排，这份数据上人工的更好）"] if pick else [])
    rpn = run.notes.get("replan")
    if rpn:                                 # 重排结果：用时是重排本身的用时
        flags = [f"突发重排，{MODES.get(rpn['mode'], rpn['mode'])}"]
    else:
        flags = [run.notes.get("mode", "")] + rule_flag + (["按秒精修"] if run.notes.get("refine", {}).get("refined") else [])
    flag_text = "、".join(f for f in flags if f)
    place_text = ("按伐谋演化出的放置规则" if by_famou else "按「价值 − 价格」的放置规则") + "把任务排进各站时间线" + \
        (f"（伐谋、人工两条规则各排一份，取综合高的：伐谋 {pick['objective']['rule']:.4f}，人工 {pick['objective']['manual']:.4f}）"
         if pick else "")
    step_box(4, "任务规划",
             ("价格协调：每颗目标按当前价格独立挑选最划算的至多 3 次观测（间隔 48 小时），被争抢的时段涨价、没人要的降价，"
              f"反复迭代；收敛后{place_text}，再做局部搜索（挤让插入、压紧、补空、延长），"
              "最后按秒精修起止时间。" if run.algo == "H" else f"{ALGOS.get(run.algo, run.algo)}，之后按秒精修起止时间。") +
             (f"这份结果是在原计划上处理突发事件“{rpn['event']}”后重排得到的。" if rpn else ""),
             fmt_s(run.runtime_s) + (f"（{flag_text}）" if flag_text else ""),
             [("观测任务", f"{m['tasks']:,}", f"其中重复观测 {m['repeat_tasks']:,}"),
              ("覆盖", f"{m['covered']:,}", f"高 {m['covered_高']} · 中 {m['covered_中']} · 低 {m['covered_低']}"),
              ("得分", f"{m['score']:,.0f}", f"综合 {sc['combined']:.4f}" + ("（按参与目标归一）" if run.subset else ""))],
             ["甘特图", "时间价格"])

    box = step_box(5, "约束校验",
                   "先做结构检查：时长 120–240 秒、同站相邻任务留 30 秒切换、同一目标两次观测间隔 48 小时、每个任务都在可见窗口内；"
                   "再用评审参考标准 Skyfield 逐秒重算每个任务的仰角、站点太阳角和目标受照（与规划器的代码完全独立）。",
                   f"结构检查 {fmt_s(sv_t)}；Skyfield " + ("未校验" if v is None else fmt_s(v.get("seconds")) if v.get("seconds")
                                                       else "已校验"),
                   [("结构检查", "全部合法" if not sv else f"{sum(sv.values())} 项违规", "；".join(f"{kk} {n}" for kk, n in sv.items())),
                    ("Skyfield 逐秒", "未校验" if v is None else ("全部合法" if v["legal"] else "有违规"),
                     "" if v is None else f"仰角下限余量 {v['margins'][0]:.2f}°"),
                    ("校验任务数", f"{len(run.target):,}", "")],
                   ["校验与导出"])
    if v is None:
        with box:
            skyfield_button("validate_demo")

    by = ssum["by_priority"]
    tot = {kk: sum(by[p].get(kk, 0) for p in by) for kk in STATUS.values()}
    step_box(6, "状态更新",
             "逐目标统计观测完成情况（答疑 #26 的“目标状态分析”）：观测了几次（只计满足 48 小时间隔的观测，与计分口径相同）、"
             "总观测时长、首次与最近一次观测时间，以及没有覆盖或没有观测满的原因。",
             fmt_s(table_timing["status_s"]),
             [("已覆盖", f"{ssum['covered']:,}", f"可观测 {ssum['observable']:,} 颗"),
              ("观测 2 次以上", f"{ssum['targets_observed_2plus']:,}", f"3 次 {tot.get('已观测 3 次', 0):,} 颗"),
              ("高优先级覆盖", f"{hi_cov:,} / {sum(hi_rows.values()):,}", f"未覆盖（全部） {tot.get('未覆盖', 0)} 颗")],
             ["概览"])

    if rp:
        r, dd = rp["res"], rp["res"]["diff"]
        nums = [("事件", KINDS[r["event"].kind], r["describe"]),
                ("覆盖", f"{r['after']['covered']:,}", f"原计划 {r['before']['covered']:,}" +
                 (f" · 只取消 {r['no_replan']['covered']:,}" if r["event"].kind == "down" else "")),
                ("原任务保留", f"{dd['保留'] / max(dd['原计划 t 之后'], 1):.1%}", f"移动 {dd['移动']} · 删除 {dd['删除']} · 新增 {dd['新增']}")]
        took = f"{fmt_s(r['seconds'])}（{MODES[r['mode']]}）"
    else:
        gain = "—"
        dyn_path = SCHEME_DIR / "results" / "dynamic.json"
        if dyn_path.exists():
            g = [v_["policies"]["dynamic_clim"]["combined"] / v_["policies"]["static"]["combined"] - 1
                 for v_ in json.loads(dyn_path.read_text(encoding="utf-8"))["years"].values()
                 if {"static", "dynamic_clim"} <= set(v_["policies"])]
            gain = f"{min(g):+.1%}" if g else "—"
        nums = [("事件", "未运行", "可在「突发重规划」自选事件"), ("每天按天气重排", gain, "对比一次排完，真实天气回放"),
                ("重排用时", "约 10 秒", "全量 10000 颗目标")]
        took = "—"
    box = step_box(7, "动态重规划",
                   "执行中出现突发事件（站点停机、目标升为高优先级、临时加一个必须观测的目标）时：冻结已经开始的任务，"
                   "按事件更新状态，用同一套价格协调重排之后的时段，并尽量少改动原计划。天气版本是“每天按执行结果重排”。",
                   took, nums, ["突发重规划", "天气与月光"])
    with box:
        ev0 = default_event(t_ev)
        if st.button(f"运行示例：{ev0.describe(cfg, ds)}（扰动最小）", key="demo_replan"):
            run_replan(ev0, "minimal")
            st.rerun()

    box = step_box(8, "结果输出",
                   "导出赛题要求的观测计划（站点、目标、起止时间、时长、切换释放时间、Skyfield 复核的仰角与太阳角、得分、首次或重复观测），"
                   "以及目标状态表和结果摘要。",
                   "点击下载时生成，约 1 秒",
                   [("观测计划", f"{len(run.target):,} 行", "CSV，UTF-8"), ("目标状态表", f"{len(srows):,} 行", "CSV"),
                    ("结果摘要", "JSON", "指标、规则与校验结果")],
                   ["校验与导出"])
    with box:
        download_buttons("demo")


# ------------------------------------------------------------------ 页签（只计算当前打开的那一个）
PAGES = {"演示": page_demo, "概览": page_overview, "站点地图": page_map, "甘特图": page_gantt, "时间价格": page_price,
         "任务解释": page_explain, "天气与月光": page_weather, "突发重规划": page_replan, "校验与导出": page_export,
         "结果对比": page_compare, "规则说明": page_rules}
for (name, page), tab in zip(PAGES.items(), st.tabs(list(PAGES), key="page", on_change="rerun")):
    if tab.open:
        with tab:
            page()
