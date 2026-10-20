"""伐谋评测：让伐谋演化"候选任务打分函数"（规划方案里的 AI 价值函数）。

任务刻意做得小：候选代码只负责给每个候选观测打两个分（目标未覆盖时 / 已覆盖时），
排序、放置、约束全部由本地贪心引擎完成，所以候选代码写得再差也只会得低分，不会产生非法计划。

评测流程：
1. 从全量数据按固定种子抽 2 个子场景（20% 目标 × 3 个站 × 6 天），算好每个候选的特征；
2. 在系统沙箱里运行候选代码（不能联网、读仓库或启动子进程，参考方案E初版的隔离方式），拿到打分；
3. 用打分跑贪心构造，按综合指标与人工权重比较；
4. combined_score = 0.5 × 各场景(候选综合 / 人工权重综合)的平均，人工权重本身得 0.5。
"""
from __future__ import annotations

import json
import math
import os
import pickle
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

import numpy as np

from .config import Config
from .data import load
from .ga import sub_problem
from .greedy import MANUAL_WEIGHTS, ai_values, candidates, feature_matrix, run_greedy, window_features
from .lagrange import objective
from .problem import Problem
from .visibility import get_windows

FEATURES = ["priority", "duration_cells", "task_score", "scarcity", "urgency", "score_per_cell",
            "conflict", "occupation", "target_windows", "window_cells", "time_frac", "max_obs", "night_pos"]
TRAIN_SEEDS = (101, 202)
TIMEOUT_S = float(os.environ.get("FAMOU_CANDIDATE_TIMEOUT_S", "40"))


class Case:
    def __init__(self, pb: Problem):
        self.pb = pb
        self.cw, self.cm = candidates(pb)
        wf = window_features(pb)
        F = feature_matrix(pb, wf, self.cw, self.cm)
        self.base = objective(pb, run_greedy(pb, self.cw, self.cm, *ai_values(F, MANUAL_WEIGHTS)))
        tg = pb.w_tg[self.cw]
        # 窗口在所处黑夜段中的位置：0/1 靠近黄昏/黎明，0.5 为午夜前后
        night = np.full(len(pb.w_k0), 0.5)
        for s, runs in enumerate(pb.win.dark):
            if not runs:
                continue
            d0 = np.array([a for a, _ in runs])
            d1 = np.array([b for _, b in runs])
            idx = np.nonzero(pb.w_st == s)[0]
            mid = (pb.w_k0[idx] + pb.w_k1[idx]) / 2
            r = np.clip(np.searchsorted(d0, mid, side="right") - 1, 0, len(d0) - 1)
            night[idx] = np.clip((mid - d0[r]) / np.maximum(d1[r] - d0[r], 1), 0, 1)
        X = np.column_stack([
            pb.P[tg], self.cm, F[:, 2] * 5.0, F[:, 3], F[:, 4], (F[:, 2] * 5.0) / (self.cm + 1), F[:, 6], F[:, 7],
            wf["n_win"][tg], (pb.w_k1 - pb.w_k0)[self.cw], pb.w_k0[self.cw] / pb.n_samples, pb.max_obs[tg],
            night[self.cw],
        ])
        self.rows = np.round(X, 5).tolist()


def _cache_path(cfg: Config) -> Path:
    return Path(cfg.cache_dir) / f"famou_cases_{cfg.scenario_key()}.pkl"


def load_cases(cfg: Config | None = None) -> list[Case]:
    cfg = cfg or Config()
    path = _cache_path(cfg)
    if path.exists():
        return pickle.loads(path.read_bytes())
    ds = load(cfg)
    pb = Problem.build(cfg, ds, get_windows(cfg, ds))
    cases = [Case(sub_problem(pb, np.random.default_rng(s), target_frac=0.2, n_st=3)) for s in TRAIN_SEEDS]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(cases))
    return cases


# ---------------------------------------------------------------- 沙箱运行候选（参考方案E初版 famou_evaluator.py）
_WORKER = r"""
import json, math, resource, runpy, sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
fsize = int(sys.argv[3])
resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
cpu = int(math.ceil(float(sys.argv[2]))) + 1
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
payload = json.load(sys.stdin)
ns = runpy.run_path(sys.argv[1], run_name="famou_candidate")
score = ns.get("score")
if not callable(score):
    raise ValueError("candidate must define score(names, rows)")
out = []
for rows in payload["cases"]:
    u, c = score(payload["names"], rows)
    out.append([list(u), list(c)])
sys.stdout.write(json.dumps(out, allow_nan=False, separators=(",", ":")))
"""


class CandidateFailure(Exception):
    pass


def _sandbox_command(workdir: Path, candidate: Path, timeout_s: float, fsize: int = 64 << 20) -> list[str]:
    sandbox = Path("/usr/bin/sandbox-exec")
    if sys.platform != "darwin" or not sandbox.is_file():
        raise CandidateFailure("需要 macOS sandbox-exec 才能隔离运行候选代码")
    executable = Path(sys.executable).resolve()
    framework = Path(sys.base_prefix) / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    if framework.is_file():
        executable = framework.resolve()
    roots = {Path(sys.base_prefix).resolve(), Path(sysconfig.get_path("stdlib")).resolve(), executable.parent,
             Path("/System"), Path("/usr/lib"), Path("/usr/share"), Path("/Library/Apple")}
    q = json.dumps
    profile = [
        "(version 1)", "(deny default)", "(allow sysctl-read)", "(allow file-read-metadata)",
        '(allow file-read* (literal "/"))',
        f"(allow process-exec (literal {q(str(executable))}))",
        f"(allow file-read* (subpath {q(str(workdir))}))",
        f"(allow file-write* (literal {q(str(workdir / 'stdout.json'))}))",
        f"(allow file-write* (literal {q(str(workdir / 'stderr.txt'))}))",
        '(allow file-read* (literal "/dev/urandom") (literal "/dev/random"))',
        '(allow file-read* file-write* (literal "/dev/null"))',
    ] + [f"(allow file-read* (subpath {q(str(r))}))" for r in sorted(roots)]
    return [str(sandbox), "-p", "\n".join(profile), str(executable), "-I", "-S", "-c", _WORKER,
            str(candidate), str(timeout_s), str(fsize)]


def run_candidate(source: bytes, cases: list[Case], timeout_s: float = TIMEOUT_S, fsize: int = 64 << 20):
    with tempfile.TemporaryDirectory(prefix="obsplan-famou-") as tmp:
        wd = Path(tmp).resolve()
        cand = wd / "candidate.py"
        cand.write_bytes(source)
        cmd = _sandbox_command(wd, cand, timeout_s, fsize)
        env = {"PATH": "/usr/bin:/bin", "HOME": str(wd), "TMPDIR": str(wd), "LANG": "C.UTF-8"}
        payload = json.dumps({"names": FEATURES, "cases": [c.rows for c in cases]}).encode()
        with (wd / "stdout.json").open("w+b") as out, (wd / "stderr.txt").open("w+b") as err:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=err, cwd=wd, env=env,
                                 start_new_session=True)
            try:
                p.communicate(payload, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.communicate()
                raise CandidateFailure(f"候选运行超过 {timeout_s:.0f} 秒")
            if p.returncode != 0:
                err.seek(0)
                tail = err.read()[-600:].decode("utf-8", "replace")
                raise CandidateFailure(f"候选运行出错: {tail}")
            out.seek(0)
            return json.loads(out.read())


def evaluate_source(source: bytes, cases: list[Case] | None = None) -> dict:
    t0 = time.time()
    cases = cases or load_cases()
    try:
        result = run_candidate(source, cases)
        if not isinstance(result, list) or len(result) != len(cases):
            raise CandidateFailure("返回格式应为每个场景一对 (uncovered, covered) 列表")
        ratios, details = [], []
        for case, (u, c) in zip(cases, result):
            n = len(case.cw)
            if len(u) != n or len(c) != n:
                raise CandidateFailure("打分列表长度必须等于候选数")
            v1 = np.asarray(u, float)
            v2 = np.asarray(c, float)
            if not (np.isfinite(v1).all() and np.isfinite(v2).all()):
                raise CandidateFailure("打分里有 NaN 或无穷大")
            v2 = np.minimum(v1, v2)
            obj = objective(case.pb, run_greedy(case.pb, case.cw, case.cm, v1, v2))
            ratios.append(obj / case.base)
            details.append(round(obj / case.base, 6))
        ratio = float(np.mean(ratios))
        return {"combined_score": float(min(1.0, max(0.0, 0.5 * ratio))), "validity": 1,
                "ratio_vs_manual": ratio, "case_ratios": details, "eval_s": round(time.time() - t0, 2)}
    except CandidateFailure as e:
        return {"combined_score": 0.0, "validity": 0, "error_info": str(e)[:800]}
