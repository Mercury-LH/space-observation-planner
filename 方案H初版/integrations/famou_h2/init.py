"""Placement priority for the price-coordination scheduler (scheme H).

score(names, rows) receives one row of features per candidate observation
(a visibility window plus a duration) and returns two lists of floats, one value
per row:
  uncovered[i]: priority of candidate i while its target has no observation yet
  covered[i]:   priority of candidate i once its target already has one
A greedy engine takes candidates in descending order of these values, places
each one that still fits, and stops at the first value <= 0. Everything left
is then filled by true value and improved by a fixed local search.

This initial version is the hand-written rule: reduced profit = value - price.
"""


def score(names, rows):
    col = {name: i for i, name in enumerate(names)}
    TS, CV, PC = col["task_score"], col["cover_value"], col["price_cost"]
    uncovered, covered = [], []
    for r in rows:
        uncovered.append(r[TS] + r[CV] - r[PC])
        covered.append(r[TS] - r[PC])
    return uncovered, covered
