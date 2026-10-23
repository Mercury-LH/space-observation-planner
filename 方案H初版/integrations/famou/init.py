"""Candidate-observation scoring for a greedy satellite-observation scheduler.

score(names, rows) receives one row of features per candidate observation and
returns two lists of floats with one value per row:
  uncovered[i]: priority of candidate i if its target has not been observed yet
  covered[i]:   priority of candidate i if its target already has an observation
A greedy engine processes candidates in descending order of these values and
places each one if it still fits. Higher value = scheduled earlier.

This initial version is the hand-tuned linear rule used as the baseline.
"""


def score(names, rows):
    col = {name: i for i, name in enumerate(names)}
    P, TS = col["priority"], col["task_score"]
    SC, UR = col["scarcity"], col["urgency"]
    SPC, CF, OC = col["score_per_cell"], col["conflict"], col["occupation"]
    uncovered, covered = [], []
    for r in rows:
        base = (1.0 * r[P] / 5.0 + 0.5 * r[TS] / 5.0 + 0.3 * r[SC] + 0.2 * r[UR]
                + 0.5 * r[SPC] / 0.6 - 0.2 * r[CF] - 0.2 * r[OC])
        uncovered.append(base + 1.0)
        covered.append(base)
    return uncovered, covered
