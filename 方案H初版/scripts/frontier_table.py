"""把 results/frontier/*.json（scripts/frontier_bench.py 的实测）汇总成"多少时间到多少分"的表，文档里的这张表都由它生成。

    uv run python scripts/frontier_table.py            # 生成 results/frontier/table.json、table.md，并更新各文档里的表
    uv run python scripts/frontier_table.py --check    # 只检查文档里的表是不是最新的（不一致返回 1）

文档里用一对标记圈出表的位置，脚本只改标记之间的内容：

    <!-- frontier-table:begin -->
    ……
    <!-- frontier-table:end -->

网页「新建规划」里弹射链选项的帮助文字也读 table.json（obsplan/tiers.py 的 chain_help），所以数字只有这一个来源。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from obsplan.tiers import RECOMMENDED_CHAIN_MIN, chain_label  # noqa: E402

SRC = ROOT / "results" / "frontier"
BEGIN, END = "<!-- frontier-table:begin -->", "<!-- frontier-table:end -->"
DOCS = [ROOT / "README.md", ROOT / "docs" / "进度报告_2026-11-12.md",
        ROOT.parent / "README.md"]


def _opt(args: list[str], name: str, default):
    return args[args.index(name) + 1] if name in args else default


def rows() -> list[dict]:
    out = []
    for p in sorted(SRC.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict) or "runs" not in d or "plan_args" not in d:
            continue                                        # structure.json、diag.json、table.json 不是测速结果
        a = d["plan_args"]
        chain = 0.0 if "--no-refine" in a else float(_opt(a, "--chain", RECOMMENDED_CHAIN_MIN))
        runs, m = d["runs"], d["median"]
        ph = {k: round(statistics.median(r["phases_s"].get(k, 0.0) for r in runs), 1)
              for k in ("windows", "solve", "refine", "chain")}
        out.append({"name": d["name"], "label": d.get("label") or d["name"], "note": d.get("note", ""),
                    "command": "python -m obsplan plan" + ("".join(" " + x for x in a)),
                    "mode": _opt(a, "--mode", "fast"), "chain_minutes": chain, "runs": len(runs),
                    "wall_s": m["wall_s"], "wall_min_s": min(r["wall_s"] for r in runs),
                    "wall_max_s": max(r["wall_s"] for r in runs), "combined": m["combined"],
                    "combined_min": min(r["combined"] for r in runs), "combined_max": max(r["combined"] for r in runs),
                    "covered": m["covered"], "score": m["score"], "avg_cores": m["avg_cores"], "phases_s": ph,
                    "all_legal": d.get("all_legal"), "machine": d.get("machine", {}), "time": d.get("time")})
    out.sort(key=lambda r: (r["wall_s"], r["name"]))
    return out


def markdown(rs: list[dict]) -> str:
    if not rs:
        return "（还没有测速结果：先跑 `scripts/frontier_bench.py`）"
    head = ("| 档位 | 命令 | 全程用时（秒） | 综合 | 覆盖 / 得分 | 平均占用核数 | Skyfield 逐秒校验 |\n"
            "|---|---|---:|---:|---:|---:|---|\n")
    body = ""
    for r in rs:
        legal = "—" if r["all_legal"] is None else (f"{r['runs']} 次全部合法" if r["all_legal"] else "有违规")
        span = "" if r["runs"] < 2 else f"（{r['wall_min_s']:.0f}–{r['wall_max_s']:.0f}）"
        cspan = "" if r["runs"] < 2 else f"（{r['combined_min']:.4f}–{r['combined_max']:.4f}）"
        cores = "—" if r["avg_cores"] is None else f"{r['avg_cores']:.1f}"
        body += (f"| {r['label']} | `{r['command']}` | {r['wall_s']:.0f}{span} | {r['combined']:.4f}{cspan} | "
                 f"{r['covered']:.0f} / {r['score']:.0f} | {cores} | {legal} |\n")
    mach = rs[0]["machine"]
    foot = (f"\n每档从启动进程到写出计划端到端跑 {max(r['runs'] for r in rs)} 次取中位数（括号里是最小–最大），"
            f"可见窗口已缓存，用时不含 Skyfield 校验；测速机器 {mach.get('platform', '?')}，{mach.get('cpus', '?')} 核。"
            f"原始记录在 `results/frontier/`，本表由 `scripts/frontier_table.py` 生成。")
    return head + body + foot


def update_doc(path: Path, md: str, check: bool) -> str:
    if not path.exists():
        return "不存在"
    s = path.read_text(encoding="utf-8")
    if BEGIN not in s or END not in s:
        return "没有标记"
    i, j = s.index(BEGIN) + len(BEGIN), s.index(END)
    new = s[:i] + "\n" + md + "\n" + s[j:]
    if new == s:
        return "已是最新"
    if not check:
        path.write_text(new, encoding="utf-8")
    return "需要更新" if check else "已更新"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="只检查，不改文件")
    a = ap.parse_args()
    rs = rows()
    md = markdown(rs)
    stale = False
    if not a.check:
        (SRC / "table.json").write_text(json.dumps({"rows": rs}, ensure_ascii=False, indent=1), encoding="utf-8")
        (SRC / "table.md").write_text(md + "\n", encoding="utf-8")
    for p in DOCS:
        state = update_doc(p, md, a.check)
        stale |= state == "需要更新"
        print(f"{p.relative_to(ROOT.parent)}：{state}")
    print(md)
    for r in rs:
        if r["mode"] == "fast":
            print(f"  网页帮助：{chain_label(r['chain_minutes'])} → 全程 {r['wall_s']:.0f} 秒、综合 {r['combined']:.4f}")
    return 1 if (a.check and stale) else 0


if __name__ == "__main__":
    raise SystemExit(main())
