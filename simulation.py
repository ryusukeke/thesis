"""船舶データの生成、到着時刻の予定ずれ、バース割当、比較実験の計算を行う。"""

import math

import numpy as np
import pandas as pd


# main.pyで使う固定条件。コマンド引数からは変更しない。
NUM_BERTHS = 4
DURATION_HOURS = 168.0
DEMAND_MULTIPLIERS = (0.75, 1.0, 1.5)
REPORT_SHIFT_HOURS = (2, 8)
TARGET_POSITION = 0.10
POLICIES = ("fixed", "same_berth", "reassign")
METRICS = (
    "total_waiting_time",
    "target_waiting_time",
    "max_waiting_time",
    "total_turnaround_time",
    "average_turnaround_time",
    "completion_time",
    "berth_changes",
    "waiting_ships",
)
WAIT_REASONS = (
    "all_berths_busy_wait",
    "assigned_berth_wait",
    "planned_order_wait",
    "planned_start_wait",
)


def generate_ship_data(num_ships=10, num_berths=NUM_BERTHS, seed=42,
                       mean_interarrival_time=3.0, duration_hours=None):
    """到着候補、基準荷役時間、荷物の最寄りバースを乱数で作る。

    荷役係数は1 + 0.1 × 最寄りバースからの番号の差。速度や実距離は扱わない。
    到着候補を実到着と予定到着の両欄に入れて、最初の計画に渡す。
    """
    rng = np.random.default_rng(seed)
    # 固定隻数は従来の計算を保つ。期間指定では、期間内の到着を順に作る。
    if duration_hours is None:
        intervals = rng.exponential(mean_interarrival_time, num_ships)
        arrivals = np.cumsum(intervals)
        arrivals -= arrivals[0]  # 最初の船の到着を時刻0にする
    else:
        arrivals = []
        arrival = rng.exponential(mean_interarrival_time)
        while arrival < duration_hours:
            arrivals.append(arrival)
            arrival += rng.exponential(mean_interarrival_time)
        arrivals = np.array(arrivals)
        num_ships = len(arrivals)
    # 隻数が違っても、同じseedなら先頭から同じ船の属性を使う。
    ship_rng = np.random.default_rng(seed + 100000)
    base_times, cargo_berths = [], []
    for _ in range(num_ships):
        # 時間単位。shape=2、scale=2.4、平均4.8時間という仮定。
        base_times.append(ship_rng.gamma(2, 2.4))
        cargo_berths.append(ship_rng.integers(1, num_berths + 1))
    data = pd.DataFrame({
        "ship": [f"V{i + 1}" for i in range(num_ships)],
        "actual_arrival_time": arrivals, "reported_arrival_time": arrivals.copy(),
        "base_handling_time": base_times, "cargo_nearest_berth": cargo_berths,
    })
    for berth in range(1, num_berths + 1):
        factor = 1.0 + 0.1 * (data["cargo_nearest_berth"] - berth).abs()
        data[f"berth_factor_B{berth}"] = factor
        data[f"handling_time_B{berth}"] = data["base_handling_time"] * factor
    return data


def apply_attack(data, report_shifts):
    """船名と変更量の辞書を使い、対象船全隻の申告を変えたコピーを返す。

    変更量が負なら早い申告、正なら遅い申告。0未満の申告は0に丸める。
    元データ、実到着時刻、荷役時間、他船の申告時刻は変更しない。
    """
    attacked = data.copy(deep=True) # deep=trueにすることでデータをコピー元と共有しない
    if not set(report_shifts).issubset(set(data["ship"])):
        raise ValueError("予定ずれ対象に存在しない船が含まれています。")
    if any(not np.isfinite(value) for value in report_shifts.values()):
        raise ValueError("申告の変更量は有限の値にしてください。")
    target = attacked["ship"].isin(report_shifts)
    original = attacked.loc[target, "actual_arrival_time"] # targetがTrueの行にあるactual_arrival_time列を取り出す
    shifts = attacked.loc[target, "ship"].map(report_shifts)
    # シミュレーション開始前の時刻を申告しないものとする。
    attacked.loc[target, "reported_arrival_time"] = (original + shifts).clip(lower=0)
    return attacked


def create_plan(data):
    """申告時刻順で全船のバース・荷役順・計画時刻を決める。

    実到着時刻は表に記録するが、計画の判断には使わない。
    初期計画の作成と、予定をずらした後の計画更新で使う。
    """
    berth_columns = [c for c in data.columns if c.startswith("handling_time_B")]
    planned_available = np.zeros(len(berth_columns))
    plan = []

    # 同時刻なら入力の行順を維持する。生成した元データでは船番号順になる。
    ordered = data.sort_values("reported_arrival_time", kind="stable")
    for order, (_, row) in enumerate(ordered.iterrows(), 1):
        candidates = []
        for index, column in enumerate(berth_columns):
            start = max(row["reported_arrival_time"], planned_available[index])
            handling = row[column]
            # 開始時刻 → 完了時刻 → バース番号の順で選択する。
            candidates.append((start, start + handling, index, handling))
        start, completion, index, handling = min(candidates)
        planned_available[index] = completion
        plan.append({
            "assignment_order": order,
            "ship": row["ship"],
            "actual_arrival_time": row["actual_arrival_time"],
            "reported_arrival_time": row["reported_arrival_time"],
            "assigned_berth": f"B{index + 1}",
            "handling_time": handling,
            "planned_start_time": start,
            "planned_completion_time": completion,
        })
    return pd.DataFrame(plan)


def coordinate_arrivals(data):
    """最初の計画を作り、各船の実到着と予定到着をその荷役開始予定にそろえる。"""
    plan = create_plan(data)
    start_times = plan.set_index("ship")["planned_start_time"]
    candidate_arrivals = data["actual_arrival_time"]
    arrivals = data["ship"].map(start_times)
    coordinated = data.assign(
        actual_arrival_time=arrivals,
        reported_arrival_time=arrivals,
        candidate_arrival_time=candidate_arrivals,
        arrival_adjustment_time=arrivals - candidate_arrivals,
    )
    # バースと荷役順は保持し、計画に載せる両到着時刻を調整後の値にそろえる。
    plan["actual_arrival_time"] = plan["planned_start_time"]
    plan["reported_arrival_time"] = plan["planned_start_time"]
    return coordinated, plan


def create_execution_result(planned_ship, berth_index, start_time, handling_time):
    """1隻の実行結果を作り、完了・待ち・在港時間を共通の式で計算する。"""
    result = planned_ship.copy()
    result["planned_berth"] = planned_ship["assigned_berth"]
    result["assigned_berth"] = f"B{berth_index + 1}"
    result["handling_time"] = handling_time
    result["start_time"] = start_time
    result["completion_time"] = start_time + handling_time
    result["waiting_time"] = start_time - planned_ship["actual_arrival_time"]
    result["turnaround_time"] = result["completion_time"] - planned_ship["actual_arrival_time"]
    return result


def execute_fixed_plan(plan):
    """計画のバース・順番・開始時刻を守り、実際の荷役を計算する。"""
    planned_ships = plan.to_dict("records")
    num_berths = max(int(row["assigned_berth"][1:]) for row in planned_ships)

    # 各バースの割当順を守る。バースごとに空く時刻を管理し、別バースは並行して動く。
    actual_available = np.zeros(num_berths) # 指定した数の0をnp.zerosで作る
    rows = []
    for row in planned_ships:
        index = int(row["assigned_berth"][1:]) - 1
        # 計画開始・実到着・同じバースの前船の完了をすべて満たしてから開始する。
        start = max(row["planned_start_time"], row["actual_arrival_time"], actual_available[index])
        result = create_execution_result(row, index, start, row["handling_time"])
        actual_available[index] = result["completion_time"]
        rows.append(result)
    return pd.DataFrame(rows)


def select_available_berth(planned_berth, handling_times, actual_available, current_time, policy):
    """使用可能な空きバースから、荷役時間が最短の候補を選ぶ。"""
    candidates = []
    for index, handling_time in enumerate(handling_times):
        if actual_available[index] > current_time:
            continue
        if policy == "same_berth" and planned_berth != f"B{index + 1}":
            continue
        candidates.append((handling_time, index))
    # 同じ荷役時間ならバース番号を優先する。使える空きがなければ待つ。
    return min(candidates) if candidates else None


def execute_adjusted_plan(data, plan, policy="reassign"):
    """到着済みの待ち船を計画順で選び、空きバースへ入れる。

    same_berthは計画バースを維持し、reassignは他バースへの変更も許す。
    両方式とも前倒し時間を制限せず、到着済みなら空き次第開始できる。
    到着・荷役完了のイベントを進め、荷役中の船は動かさない。
    """
    if policy not in ("same_berth", "reassign"):
        raise ValueError("調整方式はsame_berthまたはreassignを指定してください。")
    berth_columns = [c for c in data.columns if c.startswith("handling_time_B")]
    handling_times = data.set_index("ship")[berth_columns]
    pending = plan.sort_values("assignment_order").to_dict("records")
    actual_available = np.zeros(len(berth_columns))
    rows = []
    current_time = 0.0

    while pending:
        # 同時刻の到着と完了をまとめて扱い、到着済みの船を計画順で確認する。
        for row in pending.copy():
            if row["actual_arrival_time"] > current_time:
                continue
            selected = select_available_berth(
                row["assigned_berth"], handling_times.loc[row["ship"]],
                actual_available, current_time, policy,
            )
            if selected is None:
                continue

            handling, index = selected
            result = create_execution_result(row, index, current_time, handling)
            actual_available[index] = result["completion_time"]
            rows.append(result)
            pending.remove(row)

        if pending:
            # 次の到着または完了へ進む。未到着船をバース選択には使わない。
            next_times = [row["actual_arrival_time"] for row in pending
                          if row["actual_arrival_time"] > current_time]
            next_times.extend(time for time in actual_available if time > current_time)
            current_time = min(next_times)

    return pd.DataFrame(rows).sort_values("assignment_order").reset_index(drop=True)


def calculate_metrics(schedule):
    """全船の待ち・在港時間と、待ちが生じた隻数を返す。"""
    return {
        "total_waiting_time": schedule["waiting_time"].sum(),
        "max_waiting_time": schedule["waiting_time"].max(),
        "total_turnaround_time": schedule["turnaround_time"].sum(),
        "average_turnaround_time": schedule["turnaround_time"].mean(),
        "completion_time": schedule["completion_time"].max(),
        "berth_changes": int((schedule["assigned_berth"] != schedule["planned_berth"]).sum()),
        "waiting_ships": int((schedule["waiting_time"] > 1e-9).sum()),
    }


def calculate_waiting_reasons(schedule, num_berths, policy):
    """運用後の時間帯を4状態に分ける。複数条件が重なる場合の優先順を固定する。"""
    arrivals = schedule.actual_arrival_time.to_numpy()
    starts = schedule.start_time.to_numpy()
    completions = schedule.completion_time.to_numpy()
    berths = schedule.assigned_berth.to_numpy()
    planned_starts = schedule.planned_start_time.to_numpy()
    previous_completions = (
        schedule.sort_values("assignment_order").groupby("planned_berth").completion_time
        .shift(fill_value=0).reindex(schedule.index).to_numpy()
    )
    reasons = {name: np.zeros(len(schedule)) for name in WAIT_REASONS}
    times = np.unique(np.r_[arrivals, starts, completions, planned_starts])
    for current, following in zip(times[:-1], times[1:]):
        waiting = (arrivals <= current) & (starts > current)
        if not waiting.any():
            continue
        active = (starts <= current) & (completions > current)
        busy_berths = set(berths[active])
        if len(busy_berths) == num_berths:
            reasons["all_berths_busy_wait"][waiting] += following - current
            continue
        # 空きはあるが指定先が使用中。その次に、計画の順番・時刻の条件を調べる。
        assigned_busy = waiting & np.isin(berths, list(busy_berths))
        reasons["assigned_berth_wait"][assigned_busy] += following - current
        remaining = waiting & ~assigned_busy
        if policy == "fixed":
            order_blocked = remaining & (previous_completions > current)
            reasons["planned_order_wait"][order_blocked] += following - current
            remaining &= ~order_blocked
            time_blocked = remaining & (planned_starts > current)
            reasons["planned_start_wait"][time_blocked] += following - current
            remaining &= ~time_blocked
        if remaining.any():
            raise ValueError("開始条件を満たしたまま待つ船があり、待ちの内訳を分類できません。")
    details = schedule[["ship", "actual_arrival_time", "reported_arrival_time", "planned_berth",
                        "assigned_berth", "planned_start_time", "start_time", "completion_time", "waiting_time"]].copy()
    for name, values in reasons.items():
        details[name] = values
    return details


def count_start_competitions(schedule, num_berths):
    """バース変更で開始枠が不足する到着・完了時刻を1回ずつ数え、船別の参加数も返す。"""
    arrivals = schedule.actual_arrival_time.to_numpy()
    starts = schedule.start_time.to_numpy()
    completions = schedule.completion_time.to_numpy()
    participations = np.zeros(len(schedule), dtype=int)
    count = 0
    for current in np.unique(np.r_[arrivals, completions]):
        # 同時刻の新たな荷役開始を入れる前に、到着と完了をまとめて反映する。
        occupied = (starts < current) & (completions > current)
        free = num_berths - int(occupied.sum())
        candidates = (arrivals <= current) & (starts >= current)
        if 0 < free < int(candidates.sum()):
            count += 1
            participations[candidates] += 1
    return count, participations


def summarize_results(run_metrics):
    """実行した運用の指標を到着頻度・遅れ幅・方式別に集計する。"""
    summaries = []
    for (demand, shift, policy), rows in run_metrics.groupby(
            ["demand_multiplier", "shift_hours", "policy"], sort=False):
        for metric in METRICS:
            values = rows[metric]
            summaries.append({
                "demand_multiplier": demand, "shift_hours": shift,
                "policy": policy, "metric": metric, "trials": len(rows),
                "mean": values.mean(), "std": values.std(ddof=1),
                "minimum": values.min(), "maximum": values.max(),
            })
    return pd.DataFrame(summaries)


def execute_policy(data, plan, policy):
    """3方式の共通入口。調整する2方式には前倒し時間の制限を設けない。"""
    if policy == "fixed":
        return execute_fixed_plan(plan)
    return execute_adjusted_plan(data, plan, policy)


def select_target_ship(data):
    """実到着順で上から10％に達する1隻を選ぶ。同着は入力順。"""
    rank = math.ceil(len(data) * TARGET_POSITION)
    ordered = data.sort_values("actual_arrival_time", kind="stable")
    return ordered.iloc[rank - 1]["ship"], rank


def calculate_plan_diagnostics(data, normal_plan, duration_hours):
    """初期計画の期間内バース使用割合と、到着・完了の範囲をまとめる。"""
    occupied = normal_plan["handling_time"].sum()
    horizon = normal_plan["planned_completion_time"].max()
    if duration_hours is not None:
        horizon = duration_hours
        occupied = (normal_plan["planned_completion_time"].clip(upper=horizon)
                    - normal_plan["planned_start_time"].clip(upper=horizon)).sum()
    berths = sum(column.startswith("handling_time_B") for column in data)
    return {
        "normal_occupancy": occupied / (berths * horizon),
        "last_actual_arrival": data["actual_arrival_time"].max(),
        "last_planned_completion": normal_plan["planned_completion_time"].max(),
    }


def run_comparison_cases(data, shared, policies):
    """遅れ幅ごとに計画を更新して各方式を実行する。"""
    num_berths = sum(column.startswith("handling_time_B") for column in data)
    schedules, metrics, waiting_details = [], [], []
    for shift in REPORT_SHIFT_HOURS:
        arrivals = apply_attack(data, {shared["target_ship"]: shift})
        plan = create_plan(arrivals)
        for policy in policies:
            schedule = execute_policy(arrivals, plan, policy)
            labels = {**shared, "shift_hours": shift, "policy": policy}
            schedules.append(schedule.assign(**labels))
            details = calculate_waiting_reasons(schedule, num_berths, policy)
            competition_events = np.nan
            details["competition_participations"] = np.nan
            if policy == "reassign":
                competition_events, participations = count_start_competitions(schedule, num_berths)
                details["competition_participations"] = participations
            waiting_details.append(details.assign(**labels))
            affected = schedule.loc[schedule.ship == shared["target_ship"], "waiting_time"]
            metrics.append({
                **labels, "num_ships": len(data),
                **calculate_metrics(schedule),
                "target_waiting_time": affected.iloc[0],
                **{f"total_{name}": details[name].sum() for name in WAIT_REASONS},
                "competition_events": competition_events,
            })
    return schedules, metrics, waiting_details


def example_total_wait(metrics, policy):
    """説明図に使う最大の遅れ幅での全船の総待ち時間を返す。"""
    rows = pd.DataFrame(metrics)
    rows = rows[rows.policy == policy].set_index("shift_hours")
    total_waiting = rows["total_waiting_time"]
    return total_waiting.loc[max(REPORT_SHIFT_HOURS)]


def run_experiment(trials=1, num_ships=10, num_berths=NUM_BERTHS, seed=42,
                   policies=POLICIES, duration_hours=DURATION_HOURS,
                   mean_interarrival_time=3.0, keep_schedules=False,
                   demand_multipliers=DEMAND_MULTIPLIERS):
    """到着頻度ごとに対象1隻を選び、同じ入力で予定ずれの幅と運用方式を比較する。"""
    if trials < 1 or num_berths < 1 or seed < 0 or (duration_hours is None and num_ships < 2):
        raise ValueError("試行数・バース数は1以上、seedは0以上、固定隻数は2以上にしてください。")
    for value, name in ((mean_interarrival_time, "到着間隔"), (duration_hours, "生成期間")):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name}は有限で正の値にしてください。")
    if not policies or len(set(policies)) != len(policies) or any(p not in POLICIES for p in policies):
        raise ValueError("比較ルールは既知の値から重複なく選んでください。")
    if (not demand_multipliers or len(set(demand_multipliers)) != len(demand_multipliers)
            or any(not np.isfinite(demand) or demand <= 0 for demand in demand_multipliers)):
        raise ValueError("到着頻度の倍率は有限で正の値を重複なく選んでください。")

    inputs, schedules, metrics, diagnostics, waiting_details = [], [], [], [], []
    schedule_examples = {demand: [] for demand in demand_multipliers}
    largest_wait = dict.fromkeys(schedule_examples, -np.inf)
    for trial_seed in range(seed, seed + trials):
        for demand in demand_multipliers:
            data = generate_ship_data(
                num_ships=num_ships, num_berths=num_berths, seed=trial_seed,
                mean_interarrival_time=mean_interarrival_time / demand, duration_hours=duration_hours,
            )
            if len(data) < 2:
                raise ValueError(f"seed={trial_seed}、到着頻度{demand:g}倍：船が2隻未満です。")
            data, normal_plan = coordinate_arrivals(data)
            target, rank = select_target_ship(data)
            shared = {"trial": trial_seed - seed + 1, "seed": trial_seed, "demand_multiplier": demand,
                      "target_ship": target, "target_rank": rank}
            plan_by_ship = normal_plan.set_index("ship")
            inputs.append(data.assign(
                planned_berth=data.ship.map(plan_by_ship["assigned_berth"]),
                planned_completion_time=data.ship.map(plan_by_ship["planned_completion_time"]),
                trial=shared["trial"], seed=trial_seed, demand_multiplier=demand,
                is_target=data.ship.eq(target),
            ))
            case_schedules, case_metrics, case_details = run_comparison_cases(data, shared, policies)
            waiting_details.extend(case_details)
            diagnostics.append({**shared, **calculate_plan_diagnostics(data, normal_plan, duration_hours)})
            metrics.extend(case_metrics)
            if keep_schedules:
                schedules.extend(case_schedules)
            else:
                total_wait = example_total_wait(case_metrics, policies[0])
                if total_wait > largest_wait[demand]:
                    largest_wait[demand] = total_wait
                    schedule_examples[demand] = [case for case in case_schedules
                                                  if case.shift_hours.iloc[0] == max(REPORT_SHIFT_HOURS)]
    if not keep_schedules:
        schedules = [case for example in schedule_examples.values() for case in example]
    run_metrics = pd.DataFrame(metrics)
    return {
        "ship_data": pd.concat(inputs, ignore_index=True),
        "schedules": pd.concat(schedules, ignore_index=True),
        "run_metrics": run_metrics,
        "summary": summarize_results(run_metrics),
        "plan_diagnostics": pd.DataFrame(diagnostics),
        "waiting_details": pd.concat(waiting_details, ignore_index=True),
    }
