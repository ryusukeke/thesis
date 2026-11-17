"""TOS停止時の復旧待ち・手動FIFO・短い荷役優先を比較する計算。"""

import heapq
import math

import numpy as np
import pandas as pd

METHOD_LABELS = {
    "normal": "停止なし", "wait": "復旧待ち",
    "fifo": "手動FIFO", "spt": "手動・短い荷役優先",
}
CONDITION_COLUMNS = [
    "condition", "method", "duration", "preparation", "capacity",
]
METRICS = [
    "ships", "mean_wait", "max_wait", "long_mean_wait", "delta_wait",
    "fifo_delta_wait", "fifo_delta_long_wait", "fifo_worse_ships",
    "cohort_ships", "cohort_mean_wait", "cohort_normal_mean_wait",
    "cohort_clearance", "manual_starts", "manual_berth_hours",
]
SCHEDULE_COLUMNS = [
    "ship", "arrival", "handling", "long_handling", "berth",
    "start", "completion", "waiting", "turnaround", "phase",
]
EVENT_COLUMNS = [
    "time", "phase", "start_limit", "busy_before", "starts",
    "busy_after", "observed_waiting",
]


def generate_input(seed, interval, horizon=168.0):
    """到着と荷役を独立系列で生成し、荷役の長い上位25％を固定する。"""
    if any(not math.isfinite(x) or x <= 0 for x in (interval, horizon)):
        raise ValueError("平均到着間隔と生成期間は有限で正の値にしてください。")
    rng = np.random.default_rng(seed)
    arrivals = []
    arrival = rng.exponential(interval)
    while arrival < horizon:
        arrivals.append(arrival)
        arrival += rng.exponential(interval)
    handling = np.random.default_rng(seed + 100000).gamma(2, 2.4, len(arrivals))
    data = pd.DataFrame({
        "ship": [f"V{i + 1:04d}" for i in range(len(arrivals))],
        "arrival": arrivals, "handling": handling,
    })
    return mark_long_handling(data)


def mark_long_handling(data):
    """長時間船を入力だけから決定する。同時間なら船ID順。"""
    result = data.copy()
    order = result.sort_values(["handling", "ship"], ascending=[False, True])
    selected = order.head(math.ceil(len(result) / 4)).ship
    result["long_handling"] = result.ship.isin(selected)
    return result


def validate_input(data, method, outage_start, duration, preparation, capacity, num_berths):
    if method not in METHOD_LABELS:
        raise ValueError("未知の運用方式です。")
    if any(not math.isfinite(x) or x < 0 for x in (outage_start, duration, preparation)):
        raise ValueError("停止開始・停止時間・準備時間は有限で0以上にしてください。")
    if (not isinstance(num_berths, int) or num_berths < 1
            or not isinstance(capacity, int) or not 0 <= capacity <= num_berths):
        raise ValueError("バース数は正の整数、手動上限は0以上バース数以下の整数にしてください。")
    if data.ship.isna().any() or data.ship.duplicated().any():
        raise ValueError("船IDに欠損・重複は認めません。")
    values = data[["arrival", "handling"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (data.arrival < 0).any() or (data.handling <= 0).any():
        raise ValueError("到着は有限で0以上、荷役時間は有限で正の値にしてください。")


def operating_phase(now, method, outage_start, duration, preparation, capacity, num_berths):
    """復旧を準備完了より優先する。開始済み荷役の数はここで制限しない。"""
    restoration = outage_start + duration
    if method == "normal" or duration == 0 or now < outage_start:
        return "通常", num_berths
    if now >= restoration:
        return "復旧後", num_berths
    if method == "wait":
        return "復旧待ち", 0
    if now < outage_start + preparation:
        return "準備中", 0
    return "手動", capacity


def choose_next_ship(waiting, short_first):
    """到着済みの未開始船だけを受け取る。未来の船を判断に使わない。"""
    if short_first:
        return min(waiting, key=lambda s: (s["handling"], s["arrival"], s["ship"]))
    return min(waiting, key=lambda s: (s["arrival"], s["ship"]))


def simulate(data, method="fifo", outage_start=48.0, duration=24.0,
             preparation=2.0, capacity=2, num_berths=4):
    """完了・到着の観測→状態の判定→空きバースへの新規開始を繰り返す。"""
    validate_input(data, method, outage_start, duration, preparation, capacity, num_berths)
    ships = mark_long_handling(data).to_dict("records")
    pending = {s["ship"]: s for s in ships}
    times = {0.0, *(s["arrival"] for s in ships)}
    if method != "normal" and duration > 0:
        times.update([outage_start, outage_start + duration])
        if preparation < duration and method in ("fifo", "spt"):
            times.add(outage_start + preparation)
    clock = list(times)
    heapq.heapify(clock)
    available = [0.0] * num_berths
    rows, events = [], []
    while clock:
        now = heapq.heappop(clock)
        # 同時刻の到着・完了をまとめて処理する。
        while clock and clock[0] == now:
            heapq.heappop(clock)
        phase, limit = operating_phase(
            now, method, outage_start, duration, preparation, capacity, num_berths)
        empty = [b for b, completion in enumerate(available) if completion <= now]
        busy_before = num_berths - len(empty)
        waiting = [s for s in pending.values() if s["arrival"] <= now]
        starts = 0
        while waiting and empty and busy_before + starts < limit:
            ship = choose_next_ship(waiting, short_first=phase == "手動" and method == "spt")
            berth = empty.pop(0)
            completion = now + ship["handling"]
            available[berth] = completion
            heapq.heappush(clock, completion)
            rows.append({
                **ship, "berth": berth + 1, "start": now, "completion": completion,
                "waiting": now - ship["arrival"],
                "turnaround": completion - ship["arrival"], "phase": phase,
            })
            waiting.remove(ship)
            del pending[ship["ship"]]
            starts += 1
        events.append({
            "time": now, "phase": phase, "start_limit": limit,
            "busy_before": busy_before, "starts": starts,
            "busy_after": busy_before + starts, "observed_waiting": len(waiting),
        })
    if pending:
        raise RuntimeError("未開始船が残っています。")
    return {
        "schedule": pd.DataFrame(rows, columns=SCHEDULE_COLUMNS).sort_values("ship").reset_index(drop=True),
        "events": pd.DataFrame(events, columns=EVENT_COLUMNS),
    }


def build_conditions():
    """準備2時間・上限2隻に固定し、停止なしと3停止時間の比較条件を作る。"""
    conditions = [{
        "condition": "C01", "method": "normal", "duration": 0.0,
        "preparation": 0.0, "capacity": 4,
    }]
    for duration in (12.0, 24.0, 48.0):
        for method in ("wait", "fifo", "spt"):
            conditions.append({
                "condition": f"C{len(conditions) + 1:02d}", "method": method,
                "duration": duration,
                "preparation": 0.0 if method == "wait" else 2.0,
                "capacity": 0 if method == "wait" else 2,
            })
    return pd.DataFrame(conditions, columns=CONDITION_COLUMNS)


def measure_run(schedule, normal, fifo, condition, outage_start):
    """船IDで対応を取り、停止中到着群は運用結果によらず同じ入力から決める。"""
    waiting = schedule.set_index("ship").waiting
    normal_waiting = normal.set_index("ship").waiting.reindex(waiting.index)
    long_group = schedule.long_handling
    duration = condition["duration"]
    restoration = outage_start + duration
    cohort = schedule[(schedule.arrival >= outage_start) & (schedule.arrival < restoration)]
    manual = condition["method"] in ("fifo", "spt")
    manual_start = min(outage_start + condition["preparation"], restoration)
    overlap = (np.minimum(schedule.completion, restoration)
               - np.maximum(schedule.start, manual_start)).clip(lower=0)
    metrics = {
        "ships": len(schedule), "mean_wait": schedule.waiting.mean(),
        "max_wait": schedule.waiting.max(),
        "long_mean_wait": schedule.loc[long_group, "waiting"].mean(),
        "delta_wait": (waiting - normal_waiting).mean(),
        "fifo_delta_wait": np.nan, "fifo_delta_long_wait": np.nan,
        "fifo_worse_ships": np.nan,
        "cohort_ships": len(cohort) if duration > 0 else np.nan,
        "cohort_mean_wait": cohort.waiting.mean(),
        "cohort_normal_mean_wait": normal_waiting.reindex(cohort.ship).mean(),
        "cohort_clearance": max(0.0, cohort.start.max() - restoration) if len(cohort) else 0.0,
        "manual_starts": int((schedule.phase == "手動").sum()),
        "manual_berth_hours": overlap.sum() if manual else 0.0,
    }
    if duration == 0:
        metrics["cohort_clearance"] = np.nan
    if fifo is not None:
        difference = waiting - fifo.set_index("ship").waiting.reindex(waiting.index)
        long_ids = schedule.loc[long_group, "ship"]
        metrics.update({
            "fifo_delta_wait": difference.mean(),
            "fifo_delta_long_wait": difference.reindex(long_ids).mean(),
            "fifo_worse_ships": int((difference > 1e-9).sum()),
        })
    return metrics


def summarize_runs(runs):
    """試行を単位に平均・標本SD・対応差の増減回数を計算する。"""
    keys = ["load", *CONDITION_COLUMNS]
    rows = []
    for key, group in runs.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        row["trials"] = len(group)
        for metric in METRICS:
            row[metric] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        for metric, prefix in (("delta_wait", "normal"), ("fifo_delta_wait", "fifo")):
            differences = group[metric].dropna()
            row[f"{prefix}_increases"] = int((differences > 1e-9).sum()) if len(differences) else np.nan
            row[f"{prefix}_decreases"] = int((differences < -1e-9).sum()) if len(differences) else np.nan
            row[f"{prefix}_unchanged"] = int((differences.abs() <= 1e-9).sum()) if len(differences) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(settings):
    """試行・頻度ごとに入力を1回生成し、重複しない各条件を計算する。"""
    trials = settings["trials"]
    if not isinstance(trials, int) or trials < 1:
        raise ValueError("試行数は1以上の整数にしてください。")
    conditions = build_conditions()
    input_tables, schedule_tables, event_tables, run_rows = [], [], [], []
    for trial in range(1, trials + 1):
        seed = settings["seed"] + trial - 1
        for load in settings["loads"]:
            metadata = {"trial": trial, "seed": seed, "load": load}
            data = generate_input(seed, settings["interval"] / load, settings["horizon"])
            input_tables.append(data.assign(**metadata))
            computed = {}
            for condition in conditions.to_dict("records"):
                result = simulate(
                    data, method=condition["method"], outage_start=settings["outage_start"],
                    duration=condition["duration"], preparation=condition["preparation"],
                    capacity=condition["capacity"], num_berths=settings["num_berths"])
                computed[condition["condition"]] = result
            normal = computed["C01"]["schedule"]
            for condition in conditions.to_dict("records"):
                result = computed[condition["condition"]]
                fifo = None
                if condition["method"] in ("fifo", "spt"):
                    same = conditions[
                        (conditions.method == "fifo")
                        & (conditions.duration == condition["duration"])
                        & (conditions.preparation == condition["preparation"])
                        & (conditions.capacity == condition["capacity"])]
                    fifo = computed[same.iloc[0].condition]["schedule"]
                metrics = measure_run(result["schedule"], normal, fifo, condition, settings["outage_start"])
                identity = {**metadata, **condition}
                run_rows.append({**identity, **metrics})
                schedule_tables.append(result["schedule"].assign(**identity))
                if trial == 1:
                    event_tables.append(result["events"].assign(**identity))
    runs = pd.DataFrame(run_rows)
    return {
        "summary": summarize_runs(runs), "runs": runs,
        "inputs": pd.concat(input_tables, ignore_index=True),
        "schedules": pd.concat(schedule_tables, ignore_index=True),
        "events": pd.concat(event_tables, ignore_index=True),
    }
