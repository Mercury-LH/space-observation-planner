"""空间目标观测规划 · 产品界面

运行（在 方案H初版/ 下）：
    uv sync --extra app
    uv run streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SCHEME_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCHEME_DIR))

from obsplan.app_backend import (ALGOS, CLOUD_THRESHOLD, Run, csv_bytes, explain_target, list_runs,  # noqa: E402
                                 orbit_class, replay_weather, run_planning, station_nights, subset_problem,
                                 summary_json, task_rows, validate_run, weather_stats)
from obsplan.config import Config  # noqa: E402
from obsplan.data import load  # noqa: E402
from obsplan.problem import Problem  # noqa: E402
from obsplan.visibility import get_windows  # noqa: E402

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
    st.caption("规则默认取自答疑文档，可按需修改。")
    d0 = st.date_input("规划开始（UTC 0 点）", value=date(2026, 10, 1))
    days = st.number_input("规划天数", 1.0, 7.0, 6.0, 1.0)
    c1, c2 = st.columns(2)
    elev_min = c1.number_input("最低仰角 °", 0.0, 30.0, 10.0, 1.0)
    elev_max = c2.number_input("最高仰角 °", 60.0, 90.0, 85.0, 1.0)
    sun_max = st.number_input("站点太阳高度角上限 °", -18.0, 0.0, -6.0, 1.0)
    use_sunlit = st.checkbox("目标须被太阳照亮", value=True)
    moon = st.slider("月光规避角 °（0 = 不考虑，加分项）", 0, 45, 0, 5)
    w_cov = st.slider("覆盖数权重（其余给得分）", 0.0, 1.0, 0.5, 0.05)
    algo = st.radio("算法", list(ALGOS), format_func=lambda k: ALGOS[k])
    mode = st.radio("模式", ["快速（全量约 1 分钟）", "高质量（全量约 2 分钟）"], horizontal=False)
    fast = mode.startswith("快速")
    iters = 150 if fast else 300
    weather = st.checkbox("按历史天气做期望价值规划（加分项，仅 H）", value=False, disabled=algo != "H")
    st.markdown("**演示模式**（缩小规模，几秒出结果）")
    frac = st.slider("目标比例", 0.05, 1.0, 1.0, 0.05)
    stations = st.multiselect("站点", list(range(1, 11)), default=list(range(1, 11)))
    go_btn = st.button("运行规划", type="primary", use_container_width=True)

if go_btn:
    cfg = Config(start_utc=f"{d0:%Y-%m-%d}T00:00:00", days=float(days), elev_min=elev_min, elev_max=elev_max,
                 sun_max=sun_max, use_sunlit=use_sunlit, w_cov=w_cov, w_score=1 - w_cov, moon_sep_deg=float(moon))
    cfg_, ds, pb, orbit = base(cfg_key(cfg))
    subset = {}
    if frac < 1 or len(stations) < 10:
        subset = {"target_frac": float(frac), "stations": [s - 1 for s in stations], "seed": 0}
        pb = subset_problem(pb, subset["target_frac"], subset["stations"], 0)
    bar = st.sidebar.progress(0.0, text="开始规划……")
    run = run_planning(pb, algo, iters=iters, progress=lambda f, t: bar.progress(min(1.0, f), text=t),
                       weather=weather and algo == "H", fast=fast)
    run.subset = subset
    bar.progress(1.0, text=f"完成，用时 {run.runtime_s:.0f} 秒")
    run.save(RUNS)
    st.session_state["selected"] = run.run_id
    st.rerun()

if choice is None:
    st.title("空间目标观测规划")
    st.info("左侧「新建规划」运行一次，结果会保存在这里。全量 10000 颗目标约 2 分钟；演示模式几秒钟。")
    st.stop()

run = Run.load(choice)
cfg, ds, pb, orbit = problem_for(run)
m = run.metrics
rows = task_rows(pb, run, orbit)
df = pd.DataFrame(rows)

st.title("空间目标观测规划")
scope = "全量" if not run.subset else f"演示：{run.subset['target_frac']:.0%} 目标 × {len(run.subset['stations'])} 站"
extra = (f" · 月光规避 {cfg.moon_sep_deg:g}°" if cfg.moon_sep_deg > 0 else "") + \
    (" · 按历史天气期望规划" if run.notes.get("weather_aware") else "")
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
with tabs[3]:
    if run.mu is None:
        st.info("时间价格只有价格协调方案（H）才有。")
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
        st.markdown("价格越高，说明这一时段被越多目标争抢。高纬度站通常黄昏、黎明最贵（低轨目标扎堆），"
                    "低纬度站午夜前后更贵。规划时每个任务优先放进它窗口内最便宜的时段。")

# ------------------------------------------------------------------ 任务解释
with tabs[4]:
    c1, c2 = st.columns([1, 2])
    covered = set(run.target.tolist())
    examples = {"自己输入编号": None}
    hi_cov = [t for t in np.nonzero(ds.priority == 5.0)[0] if t in covered]
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
        fig = px.bar(wdf, x="站点", y="晴空概率", color_discrete_sequence=[BLUE], title="各站 10 月上旬晴空概率（全天统计，过去 10 年）",
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
        st.info("结论：只靠气候统计能改进的空间有限（实验中样本外约 +0.5%）；云量变化需要结合短期天气预报做动态重规划。")

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
        if v["legal"]:
            st.success(f"全部 {m['tasks']} 个任务通过 Skyfield 逐秒校验。")
        else:
            st.error(f"发现违规：{v['violations']}")
        mg = v["margins"]
        st.write(f"最小余量：仰角下限 {mg[0]:.3f}°，仰角上限 {mg[1]:.3f}°，太阳角 {mg[2]:.3f}°")
    c1, c2 = st.columns(2)
    c1.download_button("下载观测计划（CSV）", csv_bytes(pb, run), file_name=f"plan_{run.run_id}.csv", mime="text/csv",
                       use_container_width=True)
    c2.download_button("下载结果摘要（JSON）", summary_json(run), file_name=f"summary_{run.run_id}.json",
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
| 规划期 | {cfg.start_utc[:10]} 起 {cfg.days:g} 天（UTC） | 答疑 #26、#42 |
| 仰角 | {cfg.elev_min:g}°–{cfg.elev_max:g}°（规划时各留 {cfg.elev_margin:g}° 余量） | 站址文件、答疑 #26 |
| 站点太阳高度角 | < {cfg.sun_max:g}°（留 {cfg.sun_margin:g}° 余量） | 答疑 #26 |
| 目标受照 | {'须被太阳照亮（地影边界留 %g km 余量）' % cfg.sunlit_margin_km if cfg.use_sunlit else '不考虑'} | 答疑 #26、#36、#44 |
| 观测时长与计分 | 120–240 秒，60%→100% 线性 | 答疑 #30 |
| 切换 | {cfg.switch_s} 秒；每站同一时刻一个目标 | 说明.docx |
| 重复观测 | 上次结束 {cfg.repeat_gap_h:g} 小时后可再计分 | 答疑 #30、#32 |
| 月光规避（加分项） | {'月亮在地平线以上时，目标与月亮角距 ≥ %g°' % cfg.moon_sep_deg if cfg.moon_sep_deg > 0 else '不考虑'} | 说明.docx 加分项 |
| 综合指标 | {cfg.w_cov:g} × 覆盖数/10000 + {cfg.w_score:g} × 得分/30062（内部比较口径） | 答疑 #36 |
""")
