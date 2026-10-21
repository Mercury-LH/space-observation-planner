"""根据 outputs/materials/ 的数据生成项目书图表（PNG，适合贴进文档）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "materials"
OUT = ROOT / "docs" / "项目书素材"

for f in ("/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams.update({
    "axes.unicode_minus": False, "figure.dpi": 200, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b9b8b0",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": "#0b0b0b",
    "axes.grid": True, "grid.color": "#e9e8e3", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT2, MUTED = "#52514e", "#8a8983"
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_windows():
    stats = json.loads((DATA / "window_stats.json").read_text(encoding="utf-8"))
    names = [s["name"].replace("−", "-") for s in stats]
    leo = np.array([s["leo_window_h"] for s in stats]) / 1000
    oth = np.array([s["other_window_h"] for s in stats]) / 1000
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, leo, w, color=BLUE, label="低轨（LEO，8502 颗）")
    b2 = ax.bar(x + w / 2 + 0.01, oth, w, color=ORANGE, label="中高轨（1498 颗）")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8, f"{b.get_height():.0f}",
                    ha="center", va="bottom", fontsize=9, color=TEXT2)
    ax.set_xticks(x, names)
    ax.set_ylabel("合法可见窗口总时长（千小时）")
    ax.set_title("约束逐层叠加后，低轨目标的可观测时间只剩约 1/5")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "图1_约束对可观测时间的影响.png")


def _night_profile():
    """每个窗口按它覆盖的每一格计入夜间位置直方图（而不是只看窗口中点）。"""
    sys.path.insert(0, str(ROOT))
    from obsplan.config import Config
    from obsplan.data import load
    from obsplan.visibility import get_windows
    cfg = Config()
    ds = load(cfg)
    w = get_windows(cfg, ds)
    leo = np.array([float(l[52:63]) for l in ds.line2]) > 11.25
    bins = np.linspace(0, 1, 21)
    out = {True: np.zeros(20), False: np.zeros(20)}
    T = w.n_samples + 1
    for s, runs in enumerate(w.dark):
        pos = np.full(T, np.nan)
        for a, b in runs:
            pos[a:b + 1] = (np.arange(a, b + 1) - a) / max(b - a, 1)
        idx = np.clip(np.digitize(pos, bins) - 1, 0, 19)
        sel = w.station == s
        for cls in (True, False):
            k = sel & (leo[w.target] == cls)
            cov = np.zeros(T + 1)
            np.add.at(cov, w.k0[k], 1)
            np.add.at(cov, w.k1[k] + 1, -1)
            cov = np.cumsum(cov)[:T]
            ok = ~np.isnan(pos)
            out[cls] += np.bincount(idx[ok], weights=cov[ok], minlength=20)
    return bins, out[True], out[False]


def fig_night_position(z):
    bins, hl, ho = _night_profile()
    centers = (bins[:-1] + bins[1:]) / 2
    leo = hl / hl.sum() * 100
    oth = ho / ho.sum() * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(centers, leo, color=BLUE, lw=2, marker="o", ms=4, label="低轨")
    ax.plot(centers, oth, color=ORANGE, lw=2, marker="o", ms=4, label="中高轨")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["天黑（黄昏）", "", "午夜前后", "", "天亮前（黎明）"])
    ax.set_ylabel("可见时长占比（%）")
    ax.set_xlabel("在站点黑夜中的相对位置")
    ax.set_title("低轨目标的可观测时间挤在黄昏和黎明，中高轨整夜可见")
    ax.legend(frameon=False, loc="upper center")
    save(fig, "图2_可观测时间在夜间的分布.png")


def fig_price(z):
    mu, dark = z["mu"], z["dark"]
    S, T = mu.shape
    cells_10min = 20
    day0, days = 1, 2                                    # 展示第 2、3 天
    a, b = day0 * 2880, (day0 + days) * 2880
    m = mu[:, a:b]
    m = m[:, : (m.shape[1] // cells_10min) * cells_10min].reshape(S, -1, cells_10min).mean(axis=2)
    m = m / max(m.max(), 1e-12)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw={"width_ratios": [2.3, 1]})
    im = ax.imshow(m, aspect="auto", cmap=SEQ, interpolation="nearest",
                   extent=[day0 * 24, (day0 + days) * 24, S + 0.5, 0.5])
    ax.set_yticks(range(1, S + 1), [f"{i}号站" for i in range(1, S + 1)])
    ax.set_xticks(range(day0 * 24, (day0 + days) * 24 + 1, 6),
                  [f"10月{1 + h // 24}日" if h % 24 == 0 else f"{h % 24:02d}时"
                   for h in range(day0 * 24, (day0 + days) * 24 + 1, 6)], fontsize=8)
    ax.grid(False)
    ax.set_title("时间格价格（UTC，颜色越深越抢手）")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("相对价格", color=TEXT2)
    cb.outline.set_visible(False)
    # 右：按夜间位置的平均价格，高纬站 vs 低纬站
    bins = np.linspace(0, 1, 11)
    groups = {"高纬度站（4、5、8、9、10 号）": [3, 4, 7, 8, 9], "低纬度站（1、2、3、6、7 号）": [0, 1, 2, 5, 6]}
    for (name, st), color in zip(groups.items(), (BLUE, ORANGE)):
        tot, cnt = np.zeros(10), np.zeros(10)
        for s, a0, b0 in dark:
            if s not in st or b0 - a0 < 60:
                continue
            pos = (np.arange(a0, b0 + 1) - a0) / (b0 - a0)
            idx = np.clip(np.digitize(pos, bins) - 1, 0, 9)
            np.add.at(tot, idx, mu[s, a0:b0 + 1])
            np.add.at(cnt, idx, 1)
        prof = tot / cnt
        ax2.plot((bins[:-1] + bins[1:]) / 2, prof / prof.max(), color=color, lw=2, marker="o", ms=4, label=name)
    ax2.set_xticks([0, 0.5, 1], ["黄昏", "午夜", "黎明"])
    ax2.set_ylabel("平均价格（组内最大 = 1）")
    ax2.set_title("哪段最紧张因站而异")
    ax2.legend(frameon=False, fontsize=8, loc="lower center")
    ax2.set_ylim(0.4, 1.05)
    fig.tight_layout()
    save(fig, "图3_时间格价格热力图.png")


def fig_gantt(z):
    st, s0, d, P, leo = z["station"], z["start_s"], z["dur_s"], z["P"], z["leo"]
    dark = z["dark"]
    j = 2                                                  # 3 号站
    runs = dark[dark[:, 0] == j]
    a, b = runs[1, 1] * 30, runs[1, 2] * 30                # 该站第二个夜晚
    sel = (st == j) & (s0 >= a) & (s0 + d <= b)
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    for cls, color, label, y in ((True, BLUE, "低轨", 1), (False, ORANGE, "中高轨", 0)):
        k = sel & (leo == cls)
        ax.broken_barh([((s - a) / 60, dd / 60) for s, dd in zip(s0[k], d[k])], (y - 0.35, 0.7),
                       facecolors=color, edgecolor="#fcfcfb", linewidth=0.4, label=f"{label}（{k.sum()} 个任务）")
    ax.set_yticks([0, 1], ["中高轨", "低轨"])
    ax.set_xlabel("该夜天黑后的分钟数")
    ax.set_title(f"方案 H 在 3 号站一个夜晚的安排（共 {sel.sum()} 个观测任务）")
    ax.set_xlim(0, (b - a) / 60)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.32))
    save(fig, "图4_单站一夜甘特图.png")


def fig_compare():
    d = json.loads((ROOT / "results" / "compare.json").read_text(encoding="utf-8"))
    rows = d["rows"]
    labels = {"A": ("A 随机", (8, 0), "left"), "B": ("B 优先级贪心", (8, 0), "left"),
              "C": ("C 人工多因素", (8, 0), "left"), "D": ("D AI进化权重贪心", (-8, 0), "right"),
              "E": ("E AI贪心+局部搜索", (-8, -9), "right"), "G": ("G AI筛选+CP-SAT", (-8, 0), "right"),
              "G+": ("G+局部搜索", (0, -16), "center"), "H": ("H 价格协调（新算法）", (-8, 13), "right")}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for r in rows:
        key = r["algo"].split()[0]
        is_h = key == "H"
        ax.scatter(r["runtime_s"], r["combined"], s=70 if is_h else 46, color=ORANGE if is_h else BLUE,
                   edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        text, off, ha = labels[key]
        ax.annotate(text, (r["runtime_s"], r["combined"]), xytext=off, textcoords="offset points",
                    fontsize=9, color="#0b0b0b" if is_h else TEXT2, va="center", ha=ha,
                    fontweight="bold" if is_h else "normal")
    ax.set_xscale("log")
    ax.set_xlabel("求解用时（秒，对数刻度）")
    ax.set_ylabel("综合指标")
    ax.axhline(1.056, color=MUTED, lw=1, ls="--")
    ax.text(0.3, 1.063, "网络流上界 1.056（偏松，任何计划都不可能超过）", color=MUTED, fontsize=8)
    ax.set_ylim(0.58, 1.09)
    ax.set_xlim(0.2, 2000)
    ax.set_title("全量 10000 目标：H 综合最高，用时约 2 分钟（全部通过 Skyfield 校验）")
    save(fig, "图5_方案对比.png")


def fig_sweep():
    p = DATA / "sweep.json"
    if not p.exists():
        print("sweep.json 不存在，跳过取舍对比")
        return
    pts = json.loads(p.read_text(encoding="utf-8"))
    ws = [1.0, 0.75, 0.5, 0.25, 0.0]
    obj = {}
    for x in pts:
        wc = x["w_cov"]
        obj[(x["algo"], wc)] = wc * x["covered"] / 10000 + (1 - wc) * x["score"] / 30062
    diff = [(obj[("H", w)] - obj[("E", w)]) / obj[("E", w)] * 100 for w in ws]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = [ORANGE if d > 0 else BLUE for d in diff]
    bars = ax.bar(range(len(ws)), diff, 0.55, color=colors)
    for b, d in zip(bars, diff):
        ax.text(b.get_x() + b.get_width() / 2, d + (0.08 if d > 0 else -0.08), f"{d:+.2f}%",
                ha="center", va="bottom" if d > 0 else "top", fontsize=9, color=TEXT2)
    ax.axhline(0, color="#b9b8b0", lw=1)
    ax.set_xticks(range(len(ws)), ["只看覆盖\n(1.0)", "0.75", "等权重\n(0.5，答疑口径)", "0.25", "只看得分\n(0.0)"])
    ax.set_xlabel("覆盖数的权重")
    ax.set_ylabel("H 相对 E 的综合指标差（%）")
    ax.set_ylim(min(diff) - 0.8, max(diff) + 0.8)
    ax.grid(axis="x", visible=False)
    ax.text(4.35, max(diff) + 0.5, "橙色：H 更好\n蓝色：E 更好", ha="right", va="top", fontsize=8, color=MUTED)
    ax.set_title("等权重附近 H 更好；只看单一指标时 E（每次重训权重）更好")
    save(fig, "图6_覆盖与得分取舍对比.png")


def fig_convergence(z):
    ub, lb = z["hist"], z["hist_lb"]
    it = np.arange(len(ub))
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(it, ub, color=BLUE, lw=2, label="对偶上界（价格下每颗目标各自最优之和）")
    ax.plot(it, np.where(np.isfinite(lb), lb, np.nan), color=ORANGE, lw=2, label="当前最好可行计划")
    ax.set_xlabel("价格迭代次数")
    ax.set_ylabel("综合指标（局部搜索前）")
    ax.set_title("价格迭代：上界下降、可行解上升，二者之差衡量最优性")
    ax.legend(frameon=False)
    save(fig, "图7_价格迭代收敛.png")


def fig_environment():
    p = DATA / "environment.json"
    if not p.exists():
        print("environment.json 不存在，跳过加分项图")
        return
    env = json.loads(p.read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT))
    from obsplan.config import Config
    from obsplan.data import load
    from obsplan.environment import fetch_cloud_history
    cfg = Config()
    cloud = fetch_cloud_history(cfg, load(cfg), verbose=False)
    clear = np.nanmean(cloud <= env["threshold_pct"], axis=(0, 2))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    bars = ax.bar(range(1, len(clear) + 1), clear * 100, color=BLUE, width=0.62)
    for b, v in zip(bars, clear):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v:.0%}", ha="center", fontsize=8, color=TEXT2)
    ax.set_xticks(range(1, len(clear) + 1), [f"{i}号" for i in range(1, len(clear) + 1)])
    ax.set_ylabel(f"云量 ≤ {env['threshold_pct']:g}% 的时间占比（%）")
    ax.set_title("各站 10 月上旬晴空概率（2016–2025 年同期）")
    ax.set_ylim(0, 90)
    ax.grid(axis="x", visible=False)
    rows = env["plans"]
    comb = [0.5 * r["exp_covered"] / 10000 + 0.5 * r["exp_score"] / 30062 for r in rows]
    names = [r["name"] for r in rows]
    y = np.arange(len(rows))[::-1]
    colors = [ORANGE if n == "考虑天气" else BLUE for n in names]
    # 差别只有千分之几：用点图放大局部（柱状图截断坐标轴会夸大差异）
    ax2.hlines(y, 0.425, comb, color="#d9dce3", lw=1.5)
    ax2.scatter(comb, y, s=90, color=colors, zorder=3, edgecolor="#fcfcfb", linewidth=1.5)
    for yy, c, r in zip(y, comb, rows):
        ax2.text(c + 0.0006, yy, f"{c:.4f}（期望覆盖 {r['exp_covered']:.0f}）", va="center", fontsize=8, color=TEXT2)
    ax2.set_yticks(y, names)
    ax2.set_xlim(0.425, 0.4425)
    ax2.set_xticks([0.426, 0.430, 0.434, 0.438, 0.442])
    ax2.set_ylim(-0.6, len(rows) - 0.4)
    ax2.set_xlabel("2021–2025 真实天气下的期望综合指标")
    ax2.set_title("用 2016–2020 年规划，2021–2025 年检验")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "图8_天气与月光加分项.png")


if __name__ == "__main__":
    z = np.load(DATA / "h_run.npz")
    fig_windows()
    fig_night_position(z)
    fig_price(z)
    fig_gantt(z)
    fig_compare()
    fig_convergence(z)
    fig_sweep()
    fig_environment()
    print("done", file=sys.stderr)
