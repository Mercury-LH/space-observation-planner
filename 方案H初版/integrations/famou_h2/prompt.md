Improve the function `score(names, rows)` that decides the placement order in a
price-coordination (Lagrangian relaxation) scheduler for ground-based satellite
observation. Keep the signature. Use only the Python standard library. Return a
tuple `(uncovered, covered)` of two lists of finite floats, each with exactly one
value per input row. Do not print, read or write files, use the network, start
processes, or depend on randomness. Total running time over all rows must stay
well under 60 seconds (there are roughly 1,160,000 rows per evaluation).

## The scheduling problem

10 ground stations observe up to 10,000 satellites over 6 days. An observation
lasts 120-240 s (4-8 cells of 30 s) plus a 30 s switch. A station observes one
target at a time and works only at night. Low-orbit satellites are visible only
around dusk and dawn (sunlit while the station is dark), so those periods are
crowded; high-orbit satellites are visible all night. A target may be observed
again 48 h after its previous observation ends (at most 3 times in 6 days).

Objective (higher is better):
0.5 * covered_targets / 10000 + 0.5 * total_score / 30062,
where an observation scores priority * (0.6 + 0.1 * (duration_cells - 4)).
So covering a new target is worth `cover_value` (about 3.0) score points on top
of its own task score. Station time is the scarce resource; the nights are short.

## How your values are used

Price coordination has already converged: every station time cell has a price,
high where many targets compete. Then a fixed engine runs:
1. Pass 1: candidates sorted by your values (highest first). `uncovered[i]` is
   used while the target has no observation yet, `covered[i]` afterwards (it is
   clipped to <= uncovered). Each candidate is placed at the cheapest free
   position of its window if the station is free and the 48 h spacing holds.
   Pass 1 stops at the first value <= 0.
2. Pass 2: all remaining candidates by true value, filling free time.
3. A fixed local search (insert missing targets by moving others, compact,
   add repeats, extend durations).
Your function only decides the order and which candidates pass 1 takes.
The hand-written rule is reduced profit: task_score + cover_value - price_cost
(uncovered) and task_score - price_cost (covered).

## Features (columns, by name; score units are points like task_score)

- priority: 5, 3 or 2
- duration_cells: 4..8 (4 = 120 s, 8 = 240 s); each window appears once per duration
- task_score: priority * (0.6 + 0.1 * (duration_cells - 4))
- cover_value: extra points for covering a new target (about 3.0)
- price_cost: total price of the cheapest placement of this duration in the window
- price_per_cell: price_cost / (duration_cells + 1)
- window_cells: window length in cells; slack_cells: window_cells - duration_cells
- target_windows: number of windows this target has over 6 days
- scarcity: 1 / (1 + number of this target's windows after this one)
- urgency: 1 / (1 + number of this target's windows in the next 48 h)
- conflict: how many other windows overlap this one at the station, scaled to [0, 1]
- night_pos: position inside the station's night, 0 = dusk, 1 = dawn
- time_frac: window start / horizon, in [0, 1]
- max_obs: maximum observations this target can get (0..3)
- sub_pick: 1 if, at the converged prices, this target's own best plan uses
  this window (the relaxation "recommends" it), else 0
- sub_pick_same_duration: 1 if it also uses this duration
- sub_count: how many observations the target's own best plan has (0..3)
- target_dual: net value of the target's own best plan at these prices
  (its total value minus prices); high = the target is cheap to serve
- window_demand: average number of targets whose best plan uses the cells of
  this window (above 1 means over-subscribed)

## Score

The evaluator reports 0.5 * (your objective / hand-written rule objective),
averaged over two sub-scenarios (all 10,000 targets, 3 or 4 of the 10
stations, 6 days); the initial rule scores 0.5. These are as crowded as the
full problem; rules are later checked on unseen sub-scenarios and on the full
problem with all 10 stations. Invalid output scores 0. The result is measured after
the local search, so prefer orderings that leave a good structure for it.
Prefer simple, robust rules over data-specific constants.

## Lessons from an earlier run

An earlier run scored rules on smaller, less crowded sub-scenarios. Rules that
pushed hard for covering every target looked good there (+2.6%) but did not
help on the full problem: there, almost every observable target gets covered
anyway, and total score from repeat observations and longer durations is
where the remaining gains are. Avoid trading a lot of score for a few extra
covered targets.
