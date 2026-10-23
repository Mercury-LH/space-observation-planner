Improve the function `score(names, rows)` in a satellite observation scheduler.
Keep the signature. Use only the Python standard library. Return a tuple
`(uncovered, covered)` of two lists of finite floats, each with exactly one value
per input row. Do not print, read or write files, use the network, start
processes, or depend on randomness. Total running time over all rows must stay
well under 20 seconds (there are roughly 300,000 rows per evaluation).

## What the values do

A fixed greedy engine sorts candidate observations by your values (highest first)
and places each one at the best free position inside its visibility window if
the station is still free and the target's 48-hour spacing is respected. The
engine enforces every hard constraint; your function only decides the order.
`uncovered[i]` is used while the candidate's target has no observation yet,
`covered[i]` once it already has one (the engine clips covered to <= uncovered).

## Problem

10 ground stations observe up to 10,000 satellites over 6 days. Each observation
lasts 120-240 s (4-8 cells of 30 s) plus a 30 s switch. A station observes one
target at a time and is usable only at night. Low-orbit satellites are visible
only around dusk and dawn (they must be sunlit while the station is dark), so
those periods are crowded; high-orbit satellites are visible all night.
A target may be observed again 48 h after its previous observation ends.

Objective (higher is better): 0.5 * covered_targets / 10000 + 0.5 * total_score / 30062,
where an observation scores priority * (0.6 + 0.1 * (duration_cells - 4)).
Covering a new target is worth about 3 score points on top of its own score.
Station time is the scarce resource.

## Features (columns, by name)

- priority: 5, 3 or 2
- duration_cells: 4..8 (4 = 120 s, 8 = 240 s)
- task_score: priority * (0.6 + 0.1 * (duration_cells - 4))
- scarcity: 1 / (1 + number of this target's windows after this one)
- urgency: 1 / (1 + number of this target's windows in the next 48 h)
- score_per_cell: task_score / (duration_cells + 1)
- conflict: average number of windows overlapping this one at the station, scaled to [0, 1]
- occupation: (duration_cells + 1) / 9
- target_windows: total windows of this target over 6 days
- window_cells: length of this window in 30 s cells
- time_frac: window start / horizon, in [0, 1]
- max_obs: maximum number of observations this target can get (0..3)
- night_pos: position of the window inside the station's night, 0 = dusk, 1 = dawn

## Score

The evaluator reports 0.5 * (your objective / baseline objective), averaged over
two held scenarios; the initial linear rule scores 0.5. Invalid output scores 0.
Prefer simple, robust rules over brittle data-specific constants.
