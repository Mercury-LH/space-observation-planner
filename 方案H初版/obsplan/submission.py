"""主办方提交格式规范 v3.0（2026-11-23，原文见仓库 docs/官方提交格式规范_v3.0.md）：导出提交用的 JSON，并按规范复算得分。

导出：`to_records` / `write_json`。生成 JSON 数组，每条有 noradId、siteCode、startTime、endTime 四个字段；
先按站点分组（site_1 … site_10），组内按开始时间升序。

复算：`score_submission`。按规范逐条判为正常、软错误或硬错误，再算有效得分、错误率系数和最终得分。

规范里的"预报窗口"由主办方给出，我们拿不到。这里用 Skyfield 逐秒重算来近似，判定条件与 validator.py 相同：
仰角 10–85°、站点太阳高度角 < −6°、目标受照、轨道外推可信（不考虑月光，规范里没有）。
规范没写清楚的地方，除 48 小时一条外都取更严的理解，算出来的分只会偏低：
  - 偏差：观测落在预报窗口外的秒数，开始早于窗口起点、结束晚于窗口终点两端相加；
  - 有效时长：观测与预报窗口的交集。即使偏差 ≤ 1 秒，交集不足 120 秒也判硬错误；
  - 偏差超过 5 秒一律判硬错误（规范原文是"超过 5 秒且无有效交集"）；
  - 30 秒切换、48 小时冷却都从此前同站（同卫星）观测中最晚的结束时刻算到这一条开始，违规的算后一条。
    按开始时间排序后，被前面一条长观测整个盖住的观测也会被查出来；
  - 48 小时：规范 7.3 前两句按"相邻两次"判，第三句说"第二次起都不计分"。默认只判违规的那一条；
    cascade_48h=True 时，该卫星从这一条起之后的观测都判硬错误；
  - 覆盖卫星数（错误率的分母）：方案里至少有一条不是硬错误的卫星数。规范第 9 节的示例只说明未知编号不算、
    有软错误的算，"只有硬错误的已知卫星算不算"它分不出来；这里取不算（分母更小、系数更低），
    把"编号认得出就算"的结果另外给出（factor_loose、final_score_loose）。
逐秒判定只看整秒，窗口边界的真实位置在相邻两个整秒之间，所以这里的偏差最多比真实值多算不到 1 秒。
格式已判硬错误的记录（编号、站点、时间格式不对或超出时间范围）不再计算逐秒可见性。

shrink_s > 0 表示把每个预报窗口的两端各收紧这么多秒，用来估计：如果主办方的窗口比我们算的短几秒，会丢多少分。
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sgp4.api import Satrec

from .config import Config
from .data import Dataset
from .schedule import Schedule, score_ref, task_score

SPEC_VERSION = "v3.0"
SPEC_START = "2026-07-31T00:00:00Z"            # 规范第 4 节：时间范围（含两端）
SPEC_END = "2026-08-06T23:59:59Z"
TOL_NORMAL_S = 1.0                              # 偏差 ≤ 1 秒正常计分
TOL_SOFT_S = 5.0                                # 1–5 秒为软错误（该条 ×50%），超过 5 秒为硬错误
SOFT_FACTOR = 0.5
FIELDS = ("noradId", "siteCode", "startTime", "endTime")
TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"
_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")

NORMAL, SOFT, HARD = "正常", "软错误", "硬错误"


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def site_code(j: int) -> str:
    """站点下标（从 0 起）→ 规范的站点编码。规范第 3 节的 10 个站与站址文件的顺序、坐标一一对应。"""
    return f"site_{int(j) + 1}"


# ---------------------------------------------------------------- 导出

def to_records(cfg: Config, ds: Dataset, sch: Schedule) -> list[dict]:
    """计划 → 规范格式的观测列表（已按站点分组、组内按开始时间升序）。起止时刻取整到秒。"""
    t0 = cfg.start_dt
    start = np.rint(np.asarray(sch.start_s, float)).astype(np.int64)
    end = np.rint(np.asarray(sch.start_s, float) + np.asarray(sch.dur_s, float)).astype(np.int64)
    order = np.lexsort((start, sch.station))
    return [{"noradId": int(ds.satnum[sch.target[i]]),
             "siteCode": site_code(sch.station[i]),
             "startTime": (t0 + timedelta(seconds=int(start[i]))).strftime(TIME_FMT),
             "endTime": (t0 + timedelta(seconds=int(end[i]))).strftime(TIME_FMT)} for i in order]


def write_json(cfg: Config, ds: Dataset, sch: Schedule, path: str | Path) -> dict:
    """写出提交文件（UTF-8、JSON 数组），返回需要提醒的情况：编号不是 5 位的条数、超出规范时间范围的条数。"""
    recs = to_records(cfg, ds, sch)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lo, hi = _utc(SPEC_START), _utc(SPEC_END)
    return {"path": str(path), "records": len(recs),
            "not_5_digit": sum(len(str(r["noradId"])) != 5 for r in recs),
            "six_digit": sum(len(str(r["noradId"])) > 5 for r in recs),
            "out_of_range": sum(not (lo <= _utc(r["startTime"]) and _utc(r["endTime"]) <= hi) for r in recs)}


def load_json(path: str | Path) -> list:
    """读提交文件。规范示例里有整行的 // 注释，标准 JSON 不允许；读不进来时去掉整行注释再试一次。"""
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads("\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("//")))
    if isinstance(data, dict):
        data = data.get("observations", data.get("tasks", []))
    if not isinstance(data, list):
        raise ValueError("提交文件应为 JSON 数组")
    return data


# ---------------------------------------------------------------- 逐条格式检查

@dataclass
class Parsed:
    """逐条解析结果。station / target 为 −1、start_s / end_s 为 nan 表示这一条解析不出来。"""
    station: np.ndarray
    target: np.ndarray
    start_s: np.ndarray            # 相对 cfg.start_dt 的秒数
    end_s: np.ndarray
    hard: list                     # 每条的硬错误原因（列表）
    warnings: dict = field(default_factory=dict)


def parse_records(cfg: Config, ds: Dataset, rows: list, spec_start: str = SPEC_START,
                  spec_end: str = SPEC_END) -> Parsed:
    """按规范第 2–4、10 节检查每一条：字段齐全、类型、编号和站点认得出、时间格式、时间范围。
    排序要求单独统计在 warnings 里（规范没说排序不对怎么罚）。"""
    n = len(rows)
    idx = {int(s): i for i, s in enumerate(ds.satnum)}
    sites = {site_code(j): j for j in range(ds.n_stations)}
    t0 = cfg.start_dt
    lo, hi = (_utc(spec_start) - t0).total_seconds(), (_utc(spec_end) - t0).total_seconds()
    st, tg = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    a, b = np.full(n, np.nan), np.full(n, np.nan)
    hard = [[] for _ in range(n)]
    warn = defaultdict(int)
    for k, r in enumerate(rows):
        if not isinstance(r, dict) or any(f not in r for f in FIELDS):
            hard[k].append("缺少必填字段")
            continue
        nid = r["noradId"]
        if isinstance(nid, bool) or not isinstance(nid, int):
            hard[k].append("noradId 不是整数")
        elif nid not in idx:
            hard[k].append("未知卫星编号")
        else:
            tg[k] = idx[nid]
            if len(str(nid)) > 5:
                warn["noradId 超过 5 位（Alpha-5 编号转成的数字，写法待主办方确认）"] += 1
            elif len(str(nid)) < 5:
                warn["noradId 不足 5 位（整数没有前导 0）"] += 1
        if not isinstance(r["siteCode"], str) or r["siteCode"] not in sites:
            hard[k].append("未知站点编码")
        else:
            st[k] = sites[r["siteCode"]]
        ts = [r["startTime"], r["endTime"]]
        if not all(isinstance(x, str) and _TIME_RE.match(x) for x in ts):
            hard[k].append("时间格式不对（须为 ISO 8601 UTC，以 Z 结尾）")
            continue
        try:
            a[k], b[k] = ((_utc(x) - t0).total_seconds() for x in ts)
        except ValueError:
            hard[k].append("时间格式不对（须为 ISO 8601 UTC，以 Z 结尾）")
            a[k] = b[k] = np.nan
            continue
        if b[k] <= a[k]:
            hard[k].append("结束时间不晚于开始时间")
        if a[k] < lo or b[k] > hi:
            hard[k].append("超出规定时间范围")
    # 排序：同一站点连续排列、站点按 site_1 … 的顺序、组内开始时间升序
    seq = [st[k] for k in range(n) if st[k] >= 0]
    if seq:
        groups = [s for i, s in enumerate(seq) if i == 0 or s != seq[i - 1]]
        if len(groups) != len(set(groups)):
            warn["同一站点的观测没有连续排列"] += 1
        elif groups != sorted(groups):
            warn["站点分组没有按 site_1、site_2 … 的顺序"] += 1
    for j in range(ds.n_stations):
        k = np.nonzero((st == j) & np.isfinite(a))[0]
        if np.any(np.diff(a[k]) < 0):
            warn["组内没有按开始时间升序"] += 1
    return Parsed(st, tg, a, b, hard, dict(warn))


def resource_errors(cfg: Config, p: Parsed, cascade_48h: bool = False) -> list:
    """规范第 7.1、7.3 节：同站相邻观测间隔 < 30 秒、同一卫星相邻观测间隔 < 48 小时，都判后一条为硬错误。
    按开始时间排序后，拿每一条的开始和此前所有观测中最晚的结束比（不只是前一条），重叠的观测都能查出来。"""
    hard = [[] for _ in range(len(p.station))]
    ok_t = np.isfinite(p.start_s) & np.isfinite(p.end_s)
    for key, arr, gap, why in ((0, p.station, cfg.switch_s, "同站相邻观测间隔不足 30 秒"),
                               (1, p.target, cfg.repeat_gap_h * 3600, "同一卫星相邻观测间隔不足 48 小时")):
        for v in np.unique(arr[(arr >= 0) & ok_t]):
            k = np.nonzero((arr == v) & ok_t)[0]
            k = k[np.argsort(p.start_s[k], kind="stable")]
            broken = False
            last_end = p.end_s[k[0]]
            for y in k[1:]:
                bad = p.start_s[y] < last_end + gap
                if key == 1 and cascade_48h:
                    broken |= bad
                    bad = broken
                if bad:
                    hard[y].append(why)
                last_end = max(last_end, p.end_s[y])
    return hard


# ---------------------------------------------------------------- 预报窗口（Skyfield 逐秒近似）

def visibility_masks(cfg: Config, ds: Dataset, station: np.ndarray, target: np.ndarray,
                     first_s: np.ndarray, last_s: np.ndarray, verbose: bool = False) -> list:
    """每条观测在整秒 first_s … last_s（相对 cfg.start_dt）上是否满足观测条件，逐秒用 Skyfield 计算。
    判定与 validator.validate 相同：仰角 [elev_min, elev_max]、站点太阳高度角 < sun_max、目标受照、轨道可信。"""
    from skyfield.api import EarthSatellite, wgs84

    from .validator import SUN_CHUNK, _ephemeris_dir, _sky

    ts, eph = _sky(_ephemeris_dir(cfg))
    earth, sun = eph["earth"], eph["sun"]
    st0 = cfg.start_dt
    n = len(station)
    secs = [np.arange(int(first_s[i]), int(last_s[i]) + 1, dtype=float) for i in range(n)]
    ok = [np.ones(len(s), bool) for s in secs]

    def times(x):
        return ts.utc(st0.year, st0.month, st0.day, st0.hour, st0.minute, st0.second + x)

    sites = [wgs84.latlon(float(ds.st_lat[j]), float(ds.st_lon[j]), elevation_m=float(ds.st_alt_m[j]))
             for j in range(ds.n_stations)]
    if cfg.use_sun:
        for j in range(ds.n_stations):
            idx = np.nonzero(station == j)[0]
            if len(idx) == 0:
                continue
            allx = np.concatenate([secs[i] for i in idx])
            alt = np.concatenate([(earth + sites[j]).at(times(allx[c:c + SUN_CHUNK])).observe(sun).apparent()
                                  .altaz()[0].degrees for c in range(0, len(allx), SUN_CHUNK)])
            pos = 0
            for i in idx:
                m = len(secs[i])
                ok[i] &= alt[pos:pos + m] < cfg.sun_max
                pos += m
    by_target = defaultdict(list)
    for i in range(n):
        by_target[int(target[i])].append(i)
    for done, (t, idx) in enumerate(by_target.items()):
        sat = EarthSatellite(ds.line1[t], ds.line2[t], ts=ts)
        raw = Satrec.twoline2rv(ds.line1[t], ds.line2[t])
        a_km = (398600.4418 / (raw.no_kozai / 60.0) ** 2) ** (1 / 3)
        allx = np.concatenate([secs[i] for i in idx])
        tt = times(allx)
        geo = sat.at(tt)
        rn = np.linalg.norm(geo.position.km, axis=0)
        good = np.isfinite(rn) & (rn >= 0.9 * a_km * (1 - raw.ecco)) & (rn <= 1.1 * a_km * (1 + raw.ecco))
        if cfg.use_sunlit:
            good &= geo.is_sunlit(eph)
        pos = 0
        for i in idx:
            m = len(secs[i])
            sl = slice(pos, pos + m)
            pos += m
            alt = (sat - sites[int(station[i])]).at(tt[sl]).altaz()[0].degrees
            ok[i] &= good[sl] & (alt >= cfg.elev_min) & (alt <= cfg.elev_max)
        if verbose and (done + 1) % 2000 == 0:
            print(f"  [复算] 已算 {done + 1}/{len(by_target)} 颗目标", flush=True)
    return ok


def judge_window(start: float, end: float, first: int, ok: np.ndarray, shrink_s: float = 0.0) -> tuple:
    """在逐秒可见性 ok（整秒 first, first+1, … 上的布尔数组）里找与观测 [start, end] 交集最大的预报窗口，
    返回 (有交集, 偏差秒数, 有效时长秒数)。偏差是观测落在这个窗口外的秒数（两端相加）。
    shrink_s 把窗口两端各收紧这么多秒；碰到数组两头的窗口，真实边界在数组之外，那一端不收紧。"""
    last = first + len(ok) - 1
    v = np.concatenate([[False], ok, [False]]).astype(np.int8)
    d = np.diff(v)
    ra = np.nonzero(d == 1)[0] + first                 # 每段连续可见的第一秒
    rb = np.nonzero(d == -1)[0] - 1 + first            # 最后一秒
    if len(ra) == 0:
        return False, np.inf, 0.0
    ra = np.where(ra > first, ra + shrink_s, ra).astype(float)
    rb = np.where(rb < last, rb - shrink_s, rb).astype(float)
    keep = ra <= rb
    ra, rb = ra[keep], rb[keep]
    if len(ra) == 0:
        return False, np.inf, 0.0
    inter = np.minimum(end, rb) - np.maximum(start, ra)
    k = int(np.argmax(inter))
    if inter[k] < 0:
        return False, np.inf, 0.0
    dev = max(0.0, ra[k] - start) + max(0.0, end - rb[k])
    return True, float(dev), float(inter[k])


# ---------------------------------------------------------------- 计分

def classify(cfg: Config, overlap: bool, dev: float, eff: float) -> tuple[str, str | None]:
    """规范第 7.2、7.4 节：按偏差和有效时长判一条观测的状态，返回 (状态, 原因)。"""
    if not overlap:
        return HARD, "与预报窗口没有交集"
    if dev > TOL_SOFT_S:
        return HARD, "偏差超过 5 秒"
    if eff < cfg.obs_min_s:
        return HARD, "有效时长不足 120 秒"
    if dev > TOL_NORMAL_S:
        return SOFT, "偏差 1–5 秒"
    return NORMAL, None


def summarize(cfg: Config, priority: np.ndarray, target: np.ndarray, status: list, eff: np.ndarray,
              n_targets_total: int, ref: float) -> dict:
    """规范第 8 节：单条得分 = 优先级分 × 时长得分，软错误 ×50%、硬错误 0 分；
    错误率系数 = max(0, 1 − (硬错误数 + 软错误数) / 覆盖卫星数)；最终得分 = 有效得分之和 × 错误率系数。
    覆盖卫星数取至少有一条不是硬错误的卫星数（covered_scoring）；按"编号认得出就算"（covered）
    算出的系数和最终得分放在 factor_loose、final_score_loose 里，只作对照。
    priority 是每条的优先级分（未知编号填 0），target 是每条的目标下标（未知编号为 −1）。"""
    status = np.asarray(status, dtype=object)
    base = task_score(cfg, np.asarray(priority, float), np.asarray(eff, float))
    got = np.where(status == NORMAL, base, np.where(status == SOFT, SOFT_FACTOR * base, 0.0))
    target = np.asarray(target)
    covered = len(np.unique(target[target >= 0]))
    cov_ok = len(np.unique(target[(target >= 0) & (status != HARD)]))
    n_soft, n_hard = int((status == SOFT).sum()), int((status == HARD).sum())
    n_err = n_soft + n_hard
    factor = max(0.0, 1.0 - n_err / cov_ok) if cov_ok else 0.0
    factor_loose = max(0.0, 1.0 - n_err / covered) if covered else 0.0
    valid = float(got.sum())
    final = valid * factor
    return {"records": int(len(status)), "covered": covered, "covered_scoring": cov_ok,
            "normal": int((status == NORMAL).sum()), "soft": n_soft, "hard": n_hard,
            "valid_score": valid, "factor": factor, "final_score": final,
            "factor_loose": factor_loose, "final_score_loose": valid * factor_loose,
            "combined": cfg.w_cov * cov_ok / n_targets_total + cfg.w_score * final / ref,
            "task_score": got}


def score_submission(cfg: Config, ds: Dataset, rows: list, shrinks=(0.0,), cascade_48h: bool = False,
                     verbose: bool = False, spec_start: str = SPEC_START, spec_end: str = SPEC_END) -> dict:
    """按规范 v3.0 复算一份提交（rows 为 load_json 读出的列表）。逐秒可见性只算一次，shrinks 里每个收紧秒数各出一份结果。

    返回 {"parsed": Parsed, "reports": [每个 shrink 一份汇总], "details": [每个 shrink 一份逐条明细]}。
    汇总里 reasons 是各原因的条数（一条有多个原因时都计入，状态按最重的算）。"""
    p = parse_records(cfg, ds, rows, spec_start, spec_end)
    res_hard = resource_errors(cfg, p, cascade_48h)
    n = len(rows)
    fmt_ok = np.array([not h for h in p.hard], bool)          # 格式已判硬错误的不再算逐秒可见性
    vis = np.nonzero(fmt_ok & (p.station >= 0) & (p.target >= 0) & np.isfinite(p.start_s)
                     & np.isfinite(p.end_s) & (p.end_s > p.start_s))[0]
    pad = int(np.ceil(TOL_SOFT_S + max(shrinks, default=0.0))) + 2
    first = np.floor(p.start_s[vis]).astype(np.int64) - pad
    last = np.ceil(p.end_s[vis]).astype(np.int64) + pad
    masks = visibility_masks(cfg, ds, p.station[vis], p.target[vis], first, last, verbose) if len(vis) else []
    pri = np.where(p.target >= 0, ds.priority[np.maximum(p.target, 0)], 0.0)
    ref = score_ref(cfg, ds)
    reports, details = [], []
    for sh in shrinks:
        status = [HARD] * n
        reasons = [list(p.hard[k]) + list(res_hard[k]) for k in range(n)]
        dev, eff = np.full(n, np.nan), np.zeros(n)
        for m, k in enumerate(vis):
            ov, dev[k], eff[k] = judge_window(p.start_s[k], p.end_s[k], int(first[m]), masks[m], sh)
            s, why = classify(cfg, ov, dev[k], eff[k])
            if why:
                reasons[k].append(why)
            status[k] = HARD if (p.hard[k] or res_hard[k]) else s
        rep = summarize(cfg, pri, p.target, status, eff, ds.n_targets, ref)
        cnt = defaultdict(int)
        for r in reasons:
            for w in r:
                cnt[w] += 1
        got = rep.pop("task_score")
        rep.update(shrink_s=float(sh), reasons=dict(sorted(cnt.items(), key=lambda x: -x[1])),
                   warnings=p.warnings, cascade_48h=cascade_48h, spec=SPEC_VERSION)
        reports.append(rep)
        details.append({"status": status, "reasons": reasons, "deviation_s": dev, "effective_s": eff,
                        "score": got})
    return {"parsed": p, "reports": reports, "details": details}


def details_rows(rows: list, det: dict) -> list[dict]:
    """逐条明细，便于另存为 CSV：原始四个字段 + 状态、原因、偏差、有效时长、得分。"""
    out = []
    for k, r in enumerate(rows):
        r = r if isinstance(r, dict) else {}
        out.append({**{f: r.get(f, "") for f in FIELDS}, "status": det["status"][k],
                    "reasons": "；".join(det["reasons"][k]),
                    "deviation_s": "" if not np.isfinite(det["deviation_s"][k]) else f"{det['deviation_s'][k]:g}",
                    "effective_s": f"{det['effective_s'][k]:g}", "score": f"{det['score'][k]:.4f}"})
    return out
