"""规则与目标函数参数。所有赛题规则集中在这里，确认后只改这一处。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

SCHEME_DIR = Path(__file__).resolve().parents[1]     # 方案H初版/
REPO_ROOT = SCHEME_DIR.parent                         # 仓库根目录

# 主办方《提交格式规范》v3.0 第 4 节：提交的观测必须落在这个时间范围里（UTC，含两端）
SPEC_START_UTC = "2026-07-31T00:00:00"
SPEC_END_UTC = "2026-08-06T23:59:59"


@dataclass(frozen=True)
class Config:
    # ---- 数据 ----
    # 在该目录（含子目录）里查找赛题数据：数据.zip / 优先级.zip，或解压后的 轨道、站址、高/中/低优先级 txt
    data_dir: str = str(REPO_ROOT / "data")

    # ---- 规划范围 ----
    # 主办方《提交格式规范》v3.0（2026-11-23）第 4 节：2026-07-31 00:00:00 至 08-06 23:59:59 UTC，共 7 天。
    # 规划期的最后 1 秒（8/7 0:00）不在规范范围内，`plan` 会把正好结束在那一刻的任务裁掉 1 秒（submission.fit_to_range）。
    # 原来企业老师 10-21 在群里确认的是 8 月 1 日 0:00 至 8 月 7 日 0:00（6 天）：--start 2026-08-01 --days 6
    start_utc: str = "2026-07-31T00:00:00"
    days: float = 7.0
    cell_s: int = 30                      # 规划用的时间网格；按秒精修后任务起止精确到秒

    # ---- 可观测条件 ----
    elev_min: float = 10.0
    elev_max: float = 85.0
    use_sun: bool = True                  # 站点太阳高度角约束开关
    sun_max: float = -6.0
    use_sunlit: bool = True               # 目标必须被太阳照亮（不在地球影子里），答疑 #26/#36/#44
    # 规划器内部留的余量，避免主办方用不同实现复核时卡在边界上
    elev_margin: float = 0.02
    sun_margin: float = 0.05
    sunlit_margin_km: float = 10.0        # 地影判定：日-星连线离地心距离须超过地球半径 + 余量
    # 按秒的边界余量：每个可见窗口的两端各往里收这么多秒（规划期的起止不算边界）。
    # 主办方规范 v3.0 按"与预报窗口的交集"算有效时长，窗口若比我们算的短几秒，贴边的任务会出错；0 = 不收
    edge_margin_s: int = 0
    # 加分项：月光规避。月亮在地平线以上时，目标与月亮的角距须 ≥ moon_sep_deg（0 = 不考虑）
    moon_sep_deg: float = 0.0
    moon_margin_deg: float = 0.5

    # ---- 观测与资源 ----
    obs_min_s: int = 120
    obs_max_s: int = 240
    switch_s: int = 30
    repeat_gap_h: float = 48.0
    priority_score: dict = field(default_factory=lambda: {"高": 5.0, "中": 3.0, "低": 2.0})

    # ---- 综合指标 ----
    # combined = w_cov * 覆盖数 / N + w_score * 累计得分 / score_ref
    # score_ref = "sumP": 每颗目标满分观测一次的总分；也可填一个数字
    w_cov: float = 0.5
    w_score: float = 0.5
    score_ref: str | float = "sumP"

    # ---- 运行 ----
    cache_dir: str = str(SCHEME_DIR / "cache")
    seed: int = 2026

    @property
    def start_dt(self) -> datetime:
        return datetime.fromisoformat(self.start_utc).replace(tzinfo=timezone.utc)

    @property
    def n_cells(self) -> int:
        return int(round(self.days * 86400 / self.cell_s))

    @property
    def end_s(self) -> int:
        """任务最晚可以结束在第几秒（相对规划开始）。规划期的起点落在规范 v3.0 的时间范围里时，不超过规范的
        截止时刻 08-06 23:59:59——默认规划期的终点 8/7 0:00 本身不在范围里，所以最后 1 秒不能用；其他情况就是规划期终点。"""
        horizon = self.n_cells * self.cell_s
        lo, hi = (datetime.fromisoformat(s).replace(tzinfo=timezone.utc) for s in (SPEC_START_UTC, SPEC_END_UTC))
        if lo <= self.start_dt <= hi:
            return int(min(horizon, (hi - self.start_dt).total_seconds()))
        return int(horizon)

    @property
    def min_cells(self) -> int:
        return self.obs_min_s // self.cell_s

    @property
    def max_cells(self) -> int:
        return self.obs_max_s // self.cell_s

    @property
    def switch_cells(self) -> int:
        return self.switch_s // self.cell_s

    @property
    def gap_cells(self) -> int:
        return int(round(self.repeat_gap_h * 3600 / self.cell_s))

    def scenario_key(self) -> str:
        sun = f"sun{self.sun_max:g}" if self.use_sun else "nosun"
        lit = "_lit" if self.use_sunlit else ""
        moon = f"_moon{self.moon_sep_deg:g}" if self.moon_sep_deg > 0 else ""
        edge = f"_edge{self.edge_margin_s:d}" if self.edge_margin_s > 0 else ""
        return (f"{self.start_utc[:10]}_{self.days:g}d_{sun}{lit}{moon}_el{self.elev_min:g}-{self.elev_max:g}"
                f"_c{self.cell_s}{edge}")

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_saved(cls, d: dict) -> "Config":
        """读回别的机器上存下的参数（比如在 Mac 上算好、拿到服务器上展示的结果）：
        记录里的数据 / 缓存目录在本机不存在时，换成本机的默认目录，其余参数不变。"""
        d = dict(d)
        for k in ("data_dir", "cache_dir"):
            if k in d and not Path(d[k]).exists():
                d[k] = cls.__dataclass_fields__[k].default
        return cls(**d)
