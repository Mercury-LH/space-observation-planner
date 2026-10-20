Improve a self-contained Python observation scheduler. Keep the public function
`solve(instance: dict) -> list[dict]`. Use only Python's standard library. Return
exactly `target_id` (int), `station_id` (str), `start` (finite number), and `end`
(finite number) for every observation. Do not print anything.

The local evaluator supplies all data as JSON-compatible values:
`windows` is a list of dictionaries with those four fields, `priorities` maps
string target IDs to positive integer weights, `rules` contains scheduling
constraints, `horizon_s` is the positive time horizon, and `seed` is fixed.
Times are seconds relative to the scenario start. Every observation must lie
entirely inside one supplied target/station window and the planning horizon.
Respect minimum duration, station non-overlap, switching time, and at most one
observation per target. All geometric feasibility is checked independently.

Optimize aggregate distinct-target coverage first, priority-duration score
second. Full duration earns the full target priority. Minimum duration earns
60%; below full duration use the rule's `step` (60%) or `linear` convention.
The normalized evaluator score is
`(coverage + priority_duration_score / (priority_sum + 1)) / (target_count + 1)`.
For a suite, all quantities are summed across cases before this formula.
This is a development objective, not the official competition total score.
Invalid programs receive zero; every case must be valid.

Improve search ordering, insertion, exchange, removal and repair, or neighborhood
selection. Preserve deterministic behavior with the provided seed and a fixed
iteration budget. The initial per-case execution budget is 20 seconds. Do not
read or write files, access the network, spawn processes, inspect the evaluator,
hard-code data-dependent answers, or use external services. Candidate code runs
in an isolated subprocess with filesystem/network restrictions. Only your
algorithm source and aggregate numeric evaluation feedback are available to
the cloud; raw scenario data and the evaluator remain local.
