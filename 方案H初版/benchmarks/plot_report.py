"""Render a saved benchmark summary without rerunning or re-scoring solvers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "fusion_baseline": "H+ / G+ / E, full fusion",
    "fusion_short_g": "H+ / G+ / E, short G",
    "hplus_fast": "H+ 150, fast search",
    "hplus_self": "H+ self fusion",
    "hplus_e": "H+ / E fusion",
    "hplus_300_full": "H+ 300, full search",
    "manual_greedy_refined": "Manual greedy (control)",
    "priority_refined": "Priority greedy (control)",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recommended", help="Highlight a measured method; does not alter ranking")
    parser.add_argument("--subtitle", default="", help="State the verified measurement conditions on the figure")
    args = parser.parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = [r for r in report["methods"] if r["rankable"]]
    if not rows:
        parser.error("No validated methods to plot")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    weight = str(report["default_weight"])
    rows.sort(key=lambda r: r["utility_by_weight"][weight], reverse=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, max(5.5, 0.35 * len(rows) + 1.8)))
    fig.patch.set_facecolor("#f7f9fc")
    colors = []
    labels = []
    for row in rows:
        name = row["preset"]
        color = ("#138a63" if name == args.recommended else
                 "#7c4dcc" if name == "fusion_baseline" else
                 "#a4aab5" if name.endswith("_refined") else "#2876bd")
        label = LABELS.get(name, name.replace("hplus_", "H+ ").replace("_", " "))
        colors.append(color)
        labels.append(label)
        left.scatter(row["solver_s_median"], row["quality_median"], s=65,
                     color=color, edgecolor="white", zorder=3)
        if name in (args.recommended, "fusion_baseline", "hplus_10_fast", "hplus_300_full", "manual_greedy_refined"):
            baseline = name == "fusion_baseline"
            left.annotate(label, (row["solver_s_median"], row["quality_median"]),
                          xytext=(-6 if baseline else 6, -17 if name == "hplus_300_full" else 7),
                          textcoords="offset points", fontsize=8,
                          ha="right" if baseline else "left")
    left.set_xscale("log")
    left.xaxis.set_major_formatter(ScalarFormatter())
    left.set_xlabel("Complete solver wall time (seconds, logarithmic scale)")
    left.set_ylabel("Q: internal quality metric")
    left.set_title("Higher quality, less computation")
    left.grid(alpha=0.2)
    left.margins(x=0.25, y=0.18)
    values = [r["utility_by_weight"][weight] for r in rows]
    right.barh(range(len(rows)), values, color=colors)
    right.set_yticks(range(len(rows)), labels=labels, fontsize=9)
    right.invert_yaxis()
    right.set_xlim(0, max(values) * 1.11)
    for i, value in enumerate(values):
        right.text(value + 0.6, i, f"{value:.2f}", va="center", fontsize=9)
    right.set_xlabel("J: internal utility (not an official contest score)")
    right.set_title(f"Time weight = {report['default_weight']:.4f}; scale = {report['time_scale_s']:g}s")
    right.grid(axis="x", alpha=0.15)
    right.set_axisbelow(True)
    title = "Measured quality and runtime trade-off"
    if args.subtitle:
        title += "\n" + args.subtitle
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(pad=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
