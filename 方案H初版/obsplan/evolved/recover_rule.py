"""方案 H 的放置规则（由百度伐谋演化生成，已通过采纳检查并合入求解器）。

来源：伐谋混合云实验 exp-20261022103120-brwnpe（SchemeHRecoverFull_20261022，150 轮，排行榜第 1 名），
初始程序为人工规则"价值 − 价格"（integrations/famou_h3/init.py），评测在全量实例上进行（obsplan/famou_h_eval.py）。
下面的 score 函数是伐谋生成的原始代码，未做修改；本文件只加了这段说明。

采纳检查（scripts/famou_adopt_check.py，results/famou_adopt_full_rank01.json）：
- 评测口径（价格固定、快速局部搜索 + 按秒精修）：综合指标为人工规则的 1.0083 倍；
- 端到端全量 H（重新做 300 次价格迭代、完整局部搜索、按秒精修）：覆盖 9994、得分 28538、综合 0.9744，
  人工规则同次运行为 9987 / 28084 / 0.9664（+0.82%）；规划用时 115 秒（人工规则 98 秒）；
- Skyfield 逐秒校验 12368 个任务全部合法。

规则要点：子问题推荐的窗口和时长加分；长时长、紧迫、稀缺、目标净收益高的候选加分；已覆盖目标的候选
更看重得分、略轻价格；按窗口拥挤度、站点冲突和可调余地对优先级做乘法折减。
"""

def score(names, rows):
    col = {name: i for i, name in enumerate(names)}
    TS = col["task_score"]
    CV = col["cover_value"]
    PC = col["price_cost"]
    SP = col["sub_pick"]
    SPSD = col["sub_pick_same_duration"]
    WD = col["window_demand"]
    DC = col["duration_cells"]
    UR = col["urgency"]
    SC = col["scarcity"]
    SL = col["slack_cells"]
    TD = col["target_dual"]
    CF = col["conflict"]

    uncovered = []
    covered = []
    u_app = uncovered.append
    c_app = covered.append

    for r in rows:
        ts = r[TS]
        pc = r[PC]
        
        # Positive boosts for desirable traits.
        # - Reward candidates chosen by the subproblem (SP, SPSD)
        # - Prioritize longer durations (harder to fit)
        # - Prioritize high urgency and scarcity (critical for 48h spacing)
        # - Favor targets that are globally efficient (high target_dual)
        boost = (
            r[SP] * 2.0 +
            r[SPSD] * 1.0 +
            (r[DC] - 4.0) * 0.2 +
            r[UR] * 0.5 +
            r[SC] * 0.5 +
            r[TD] * 0.2
        )
        
        # Base profit calculations
        u_base = ts + r[CV] - pc + boost
        c_base = ts * 1.2 - pc * 0.9 + boost
        
        # Multiplicative penalty to avoid dropping highly profitable candidates below zero.
        # This incorporates station conflict alongside window demand and slack to
        # deprioritize candidates competing for bottleneck resources without over-penalizing.
        penalty_mult = 1.0 + r[WD] * 0.05 + r[CF] * 0.1 + r[SL] * 0.005
        
        # Apply penalty: scale down positive scores, scale up (worsen) negative scores
        u = u_base / penalty_mult if u_base > 0.0 else u_base * penalty_mult
        c = c_base / penalty_mult if c_base > 0.0 else c_base * penalty_mult
                
        u_app(u)
        c_app(c)

    return uncovered, covered
