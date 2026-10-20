# Observation-scheduling development experiment

This hybrid Famou experiment evolves the algorithm in `init.py`. The local
`evaluator.py` loads already computed, reproducible real-orbit development
instances and independently rechecks every submitted schedule, including
geometric elevation limits.

Default case: 300 targets, two stations, one hour, using the existing
`congested_300_1h_run01` output. Optional additional saved cases are selected
locally through `FAMOU_RUN_DIRS`, separated by the operating system path separator
(`:` on macOS). A two-case suite can include `congested_1000_3h_run01`.
No raw data files, observation windows, saved answers, local paths or credentials
are required in the cloud-facing problem description.

The start time, shortened horizon, selected stations, at-most-once target policy,
and coverage-first objective are explicit development assumptions. This is not
the full-scale competition task, evidence of model novelty, or a promised
improvement. The full six-day baseline is already close to its current numerical
upper bound, so short congested cases provide a more informative integration
test. Candidate results must ultimately be compared against the saved classical
baselines under equal budgets and held-out cases.

Metrics include `validity`, `combined_score`, `coverage_count`,
`observation_score`, `geometry_checked`, and per-case numeric totals. The score
is `(C + S/(P+1))/(N+1)`, with aggregate coverage `C`, duration-weighted priority
score `S`, total available priority weight `P`, and target count `N`. One extra
covered target always outweighs any possible secondary-score change.

Numeric error codes: 0 success, 1 candidate process failure, 2 timeout,
3 malformed result, 4 structural constraint violation, 5 geometry violation,
6 missing/oversized source or unavailable supported sandbox.
Infrastructure or changed input-file errors abort evaluation rather than
masquerading as poor algorithm quality.
