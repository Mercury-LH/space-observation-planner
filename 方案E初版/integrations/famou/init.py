"""Self-contained deterministic observation scheduler; Python standard library."""
import random


def solve(instance):
    rules = instance["rules"]
    minimum = float(rules["min_duration_s"])
    full = float(rules["full_duration_s"])
    switch = float(rules["switch_s"])
    horizon = float(instance["horizon_s"])
    priorities = {int(key): value for key, value in instance["priorities"].items()}
    windows = {}
    stations = set()
    for item in instance["windows"]:
        if item["end"] - item["start"] + 1e-8 >= minimum:
            windows.setdefault(item["target_id"], []).append(item)
            stations.add(item["station_id"])
    for values in windows.values():
        values.sort(key=lambda item: (item["end"], item["start"], item["station_id"]))
    targets = sorted(windows)
    rng = random.Random(instance.get("seed", 20261020))

    def merit(plan):
        score = 0.0
        for item in plan:
            duration = item["end"] - item["start"]
            if duration + 1e-7 >= full:
                fraction = 1.0
            elif rules["score_mode"] == "linear" and full > minimum:
                fraction = 0.6 + 0.4 * (duration - minimum) / (full - minimum)
            else:
                fraction = 0.6
            score += fraction * priorities[item["target_id"]]
        return len(plan), score

    def repair(initial, order, prefer_full=False):
        table = {station: [] for station in sorted(stations)}
        used = set()
        for original in initial:
            item = dict(original)
            table[item["station_id"]].append(item)
            used.add(item["target_id"])
        for values in table.values():
            values.sort(key=lambda item: item["start"])
        for target in order:
            if target in used:
                continue
            best = None
            for window in windows[target]:
                station = window["station_id"]
                schedule = table[station]
                for index in range(len(schedule) + 1):
                    left = 0.0 if index == 0 else schedule[index - 1]["end"] + switch
                    right = horizon if index == len(schedule) else schedule[index]["start"] - switch
                    start = max(left, window["start"])
                    end_limit = min(right, window["end"])
                    if end_limit - start + 1e-8 < minimum:
                        continue
                    duration = full if prefer_full and end_limit - start >= full else minimum
                    # Prefer earliest completion, with compact gaps as a tie-breaker.
                    key = (start + duration, end_limit - start, station, index)
                    if best is None or key < best[0]:
                        best = key, station, index, start, duration
            if best is not None:
                _, station, index, start, duration = best
                table[station].insert(index, {
                    "target_id": target, "station_id": station,
                    "start": start, "end": start + duration,
                })
                used.add(target)
        # Extend already placed tasks without moving the next task or its switch.
        for station in sorted(table):
            schedule = table[station]
            for index, item in enumerate(schedule):
                right = horizon if index + 1 == len(schedule) else schedule[index + 1]["start"] - switch
                for window in windows[item["target_id"]]:
                    if (window["station_id"] == station
                            and window["start"] <= item["start"] + 1e-8
                            and item["end"] <= window["end"] + 1e-8):
                        right_limit = min(right, window["end"])
                        if right_limit - item["start"] >= full:
                            item["end"] = item["start"] + full
                        break
        return [item for station in sorted(table) for item in table[station]]

    orders = [
        sorted(targets, key=lambda target: (len(windows[target]), windows[target][0]["end"], -priorities[target], target)),
        sorted(targets, key=lambda target: (windows[target][0]["end"], -priorities[target], target)),
        sorted(targets, key=lambda target: (-priorities[target], len(windows[target]), target)),
    ]
    best = []
    for order in orders:
        for prefer_full in (False, True):
            candidate = repair([], order, prefer_full)
            if merit(candidate) > merit(best):
                best = candidate
    current = best
    # Fixed iteration budget yields repeatable candidates; no wall-clock scoring.
    for iteration in range(180):
        if not current:
            break
        count = min(len(current), 2 + iteration % 9)
        removed = set(rng.sample(range(len(current)), count))
        partial = [item for index, item in enumerate(current) if index not in removed]
        order = list(targets)
        rng.shuffle(order)
        if iteration % 3 == 0:
            order.sort(key=lambda target: len(windows[target]))
        candidate = repair(partial, order, iteration % 7 == 0)
        if merit(candidate) >= merit(current):
            current = candidate
        elif iteration % 19 == 0:
            current = candidate
        if merit(current) > merit(best):
            best = current
    return sorted(best, key=lambda item: (item["station_id"], item["start"], item["target_id"]))
