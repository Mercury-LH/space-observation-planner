# Local deterministic derivative of a Baidu Famou cloud candidate.
# Source experiment: exp-20261020225940-6h7mrd
# Original candidate: outputs/famou_connection/candidates_snapshot/strategy_rank03.py
# Original SHA256: 1fd23f07d7c49771ceb48ed49a125f2225195a865066c8e85280cc462c8bcde3
# Only algorithmic edit: sort the active station set before RNG selection.
# Raw cloud candidate is preserved; its hash-dependent station order was unstable.
"""Self-contained deterministic observation scheduler; Python standard library."""
import random
import math

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
    priority_sum = sum(priorities.values())

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

    def get_scalar(plan):
        cov, score = merit(plan)
        return cov + score / (priority_sum + 1.0)

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
            best_opt = None
            for window in windows[target]:
                station = window["station_id"]
                w_start = window["start"]
                w_end = window["end"]
                schedule = table[station]
                
                early_bound = w_end - minimum + 1e-8
                late_bound = w_start + minimum - 1e-8
                
                for index in range(len(schedule) + 1):
                    left = 0.0 if index == 0 else schedule[index - 1]["end"] + switch
                    if left > early_bound:
                        break
                    right = horizon if index == len(schedule) else schedule[index]["start"] - switch
                    if right < late_bound:
                        continue
                        
                    start = left if left > w_start else w_start
                    end_limit = right if right < w_end else w_end
                    if end_limit - start + 1e-8 < minimum:
                        continue
                        
                    duration = full if prefer_full and end_limit - start >= full else minimum
                    key = (start + duration, end_limit - start, station, index)
                    if best_opt is None or key < best_opt[0]:
                        best_opt = key, station, index, start, duration
            if best_opt is not None:
                _, station, index, start, duration = best_opt
                table[station].insert(index, {
                    "target_id": target, "station_id": station,
                    "start": start, "end": start + duration,
                })
                used.add(target)
                
        # Extend already placed tasks without moving the next task or its switch.
        for station in sorted(table):
            schedule = table[station]
            for index, item in enumerate(schedule):
                left = 0.0 if index == 0 else schedule[index - 1]["end"] + switch
                right = horizon if index + 1 == len(schedule) else schedule[index + 1]["start"] - switch
                for window in windows[item["target_id"]]:
                    if (window["station_id"] == station
                            and window["start"] <= item["start"] + 1e-8
                            and item["end"] <= window["end"] + 1e-8):
                        left_limit = left if left > window["start"] else window["start"]
                        right_limit = right if right < window["end"] else window["end"]
                        max_duration = right_limit - left_limit
                        current_dur = item["end"] - item["start"]
                        
                        if max_duration + 1e-8 >= full:
                            if current_dur < full:
                                ext_left = min(item["start"] - left_limit, full - current_dur)
                                if ext_left > 0:
                                    item["start"] -= ext_left
                                    current_dur += ext_left
                                ext_right = min(right_limit - item["end"], full - current_dur)
                                if ext_right > 0:
                                    item["end"] += ext_right
                        elif rules["score_mode"] == "linear" and max_duration > current_dur:
                            item["start"] = left_limit
                            item["end"] = right_limit
                        break
        return [item for station in sorted(table) for item in table[station]]

    orders_init = [
        sorted(targets, key=lambda target: (len(windows[target]), windows[target][0]["end"], -priorities[target], target)),
        sorted(targets, key=lambda target: (windows[target][0]["end"], -priorities[target], target)),
        sorted(targets, key=lambda target: (-priorities[target], len(windows[target]), target)),
    ]
    
    best = []
    best_scalar = -1.0
    for order in orders_init:
        for prefer_full in (False, True):
            candidate = repair([], order, prefer_full)
            candidate_scalar = get_scalar(candidate)
            if candidate_scalar > best_scalar:
                best = candidate
                best_scalar = candidate_scalar
                
    current = best
    current_scalar = best_scalar

    budget = 2500
    T_start = 1.0
    T_end = 0.001

    for iteration in range(budget):
        if not current:
            break
            
        count = min(len(current), 2 + iteration % 9)
        strategy = rng.random()
        if strategy < 0.4:
            removed = set(rng.sample(range(len(current)), count))
        elif strategy < 0.7:
            active_stations = sorted({item["station_id"] for item in current})
            station = rng.choice(active_stations)
            station_items = [i for i, item in enumerate(current) if item["station_id"] == station]
            removed = set(rng.sample(station_items, min(len(station_items), count)))
        else:
            center_time = rng.choice(current)["start"]
            sorted_indices = sorted(range(len(current)), key=lambda i: abs(current[i]["start"] - center_time))
            removed = set(sorted_indices[:count])
            
        partial = [item for index, item in enumerate(current) if index not in removed]
        
        order = list(targets)
        rng.shuffle(order)
        order_strategy = iteration % 4
        if order_strategy == 0:
            order.sort(key=lambda target: len(windows[target]))
        elif order_strategy == 1:
            order.sort(key=lambda target: priorities[target], reverse=True)
        elif order_strategy == 2:
            order.sort(key=lambda target: -priorities[target] / (len(windows[target]) + 1))
            
        candidate = repair(partial, order, iteration % 7 == 0)
        candidate_scalar = get_scalar(candidate)
        
        if candidate_scalar >= current_scalar:
            current = candidate
            current_scalar = candidate_scalar
        else:
            T = T_start * math.exp(iteration / budget * math.log(T_end / T_start))
            if rng.random() < math.exp((candidate_scalar - current_scalar) / T):
                current = candidate
                current_scalar = candidate_scalar
                
        if current_scalar > best_scalar:
            best = current
            best_scalar = current_scalar

    return sorted(best, key=lambda item: (item["station_id"], item["start"], item["target_id"]))