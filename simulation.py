"""船舶データの生成、到着時刻の予定ずれ、バース割当、比較実験の計算を行う。"""

import numpy as np
import pandas as pd


# main.pyで使う固定条件。コマンド引数からは変更しない。
NUM_BERTHS = 4
DURATION_HOURS = 168.0
REPORT_SHIFT_CHOICES = (-8, -4, -2, 2, 4, 8)
POLICIES = ("fixed", "same_berth", "reassign")
METRICS = (
    "total_waiting_time",
    "average_waiting_time",
    "max_waiting_time",
    "total_turnaround_time",
    "average_turnaround_time",
    "completion_time",
    "berth_changes",
    "increased_waiting_ships",
)
WAIT_REASONS = (
    "all_berths_busy_wait",
    "assigned_berth_wait",
    "planned_order_wait",
    "planned_start_wait",
)


def generate_ship_data(num_ships=10, num_berths=NUM_BERTHS, seed=42,
                       mean_interarrival_time=3.0, duration_hours=None):
    """実到着、基準荷役時間、荷物の最寄りバースを乱数で作る。

    荷役係数は1 + 0.1 × 最寄りバースからの番号の差。速度や実距離は扱わない。
    ずれなしの予定情報は実到着と同じにし、計画作成後も実到着を動かさない。
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
    ずれなし条件でも予定ずれ条件でも、この同じ計算を使う。
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


def calculate_metrics(schedule, normal=None):
    """全船の待ち・在港時間と、同じ方式のずれなし条件より待ちが増えた隻数を返す。"""
    changes = pd.Series(0.0, index=schedule.index)
    if normal is not None:
        baseline = schedule["ship"].map(normal.set_index("ship")["waiting_time"])
        changes = schedule["waiting_time"] - baseline
    return {
        "total_waiting_time": schedule["waiting_time"].sum(),
        "average_waiting_time": schedule["waiting_time"].mean(),
        "max_waiting_time": schedule["waiting_time"].max(),
        "total_turnaround_time": schedule["turnaround_time"].sum(),
        "average_turnaround_time": schedule["turnaround_time"].mean(),
        "completion_time": schedule["completion_time"].max(),
        "berth_changes": int((schedule["assigned_berth"] != schedule["planned_berth"]).sum()),
        "increased_waiting_ships": int((changes > 1e-9).sum()),
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
    """試行・方式が同じずれなしの値との差を求め、方式ごとに10試行を集計する。"""
    normal = run_metrics[run_metrics.scenario == "normal"].set_index(["trial", "policy"])
    attacks = run_metrics[run_metrics.scenario == "attack"]
    joined = attacks.join(normal[list(METRICS)], on=["trial", "policy"], rsuffix="_normal")
    summaries = []
    for policy, rows in joined.groupby("policy", sort=False):
        for metric in METRICS:
            values, baseline = rows[metric], rows[f"{metric}_normal"]
            differences = values - baseline
            summaries.append({
                "policy": policy, "metric": metric, "trials": len(rows),
                "mean": values.mean(), "std": values.std(ddof=1),
                "normal_mean": baseline.mean(),
                "mean_difference": differences.mean(), "std_difference": differences.std(ddof=1),
                "min_difference": differences.min(), "max_difference": differences.max(),
                "increased_trials": int((differences > 1e-9).sum()),
                "decreased_trials": int((differences < -1e-9).sum()),
                "unchanged_trials": int((differences.abs() <= 1e-9).sum()),
            })
    return pd.DataFrame(summaries)


def execute_policy(data, plan, policy):
    """3方式の共通入口。調整する2方式には前倒し時間の制限を設けない。"""
    if policy == "fixed":
        return execute_fixed_plan(plan)
    return execute_adjusted_plan(data, plan, policy)


def create_report_shifts(data, seed):
    """全船について、6種類の変更量から独立に等確率で1つずつ選ぶ。"""
    rng = np.random.default_rng(seed + 200000)
    shifts = rng.choice(REPORT_SHIFT_CHOICES, size=len(data))
    return dict(zip(data["ship"], shifts.tolist()))


def calculate_plan_diagnostics(data, normal_plan, duration_hours):
    """期間内のずれなしの計画のバース使用割合と、到着・完了の範囲をまとめる。"""
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


def run_comparison_cases(data, attacked, shared, policies):
    """ずれなしの計画と予定ずれありの計画を各1回作り、同じ船データで3方式を比較する。"""
    normal_plan = create_plan(data)
    attacked_plan = create_plan(attacked)
    schedules, metrics, waiting_details = [], [], []
    num_berths = sum(column.startswith("handling_time_B") for column in data)
    for policy in policies:
        normal = execute_policy(data, normal_plan, policy)
        attack = execute_policy(attacked, attacked_plan, policy)
        for scenario, schedule in (("normal", normal), ("attack", attack)):
            labels = {**shared, "scenario": scenario, "policy": policy}
            schedules.append(schedule.assign(**labels))
            details = calculate_waiting_reasons(schedule, num_berths, policy)
            competition_events = np.nan
            details["competition_participations"] = np.nan
            if policy == "reassign":
                competition_events, participations = count_start_competitions(schedule, num_berths)
                details["competition_participations"] = participations
            waiting_details.append(details.assign(**labels))
            metrics.append({
                **labels, "num_ships": len(data),
                **calculate_metrics(schedule, normal if scenario == "attack" else None),
                **{f"average_{name}": details[name].mean() for name in WAIT_REASONS},
                "competition_events": competition_events,
            })
    return normal_plan, schedules, metrics, waiting_details


def example_wait_increase(metrics, policy):
    """説明用の試行を選ぶため、その方式のずれなし条件からの平均待ち増分を求める。"""
    rows = pd.DataFrame(metrics)
    rows = rows[rows.policy == policy].set_index("scenario")
    return rows.loc["attack", "average_waiting_time"] - rows.loc["normal", "average_waiting_time"]


def run_experiment(trials=10, num_ships=10, num_berths=NUM_BERTHS, seed=42,
                   policies=POLICIES, duration_hours=DURATION_HOURS,
                   mean_interarrival_time=3.0, keep_schedules=False):
    """全船の申告をランダムに変え、ずれなし条件との3方式の対応比較を行う。

    各船の変更量は−8・−4・−2・＋2・＋4・＋8時間から独立に等確率で選ぶ。
    実到着・荷役条件と抽選した予定ずれ内容は、同じ試行の全方式で共有する。
    keep_schedules=Falseでは、先頭方式の平均待ち増分が最大の1試行を図用に保持する。
    """
    if trials < 1 or num_berths < 1 or seed < 0 or (duration_hours is None and num_ships < 2):
        raise ValueError("試行数・バース数は1以上、seedは0以上、固定隻数は2以上にしてください。")
    for value, name in ((mean_interarrival_time, "到着間隔"), (duration_hours, "生成期間")):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name}は有限で正の値にしてください。")
    if not policies or len(set(policies)) != len(policies) or any(p not in POLICIES for p in policies):
        raise ValueError("比較ルールは既知の値から重複なく選んでください。")
    inputs, schedules, metrics, diagnostics, waiting_details = [], [], [], [], []
    best_increase = -np.inf
    for trial in range(1, trials + 1):
        trial_seed = seed + trial - 1
        data = generate_ship_data(
            num_ships=num_ships, num_berths=num_berths, seed=trial_seed,
            mean_interarrival_time=mean_interarrival_time, duration_hours=duration_hours,
        )
        if len(data) < 2:
            raise ValueError(f"seed={trial_seed}：船が2隻未満です。期間・間隔・隻数を見直してください。")
        shifts = create_report_shifts(data, trial_seed)
        attacked = apply_attack(data, shifts)
        shared = {"trial": trial, "seed": trial_seed, "attack_seed": trial_seed + 200000}
        # 入力は船ごとに1行とし、抽選値と0未満の申告を丸めた後の値を区別する。
        inputs.append(data.assign(
            **shared, requested_report_shift=data.ship.map(shifts),
            attacked_reported_arrival_time=attacked.reported_arrival_time,
            actual_report_shift=attacked.reported_arrival_time - data.actual_arrival_time,
        ))
        normal_plan, case_schedules, case_metrics, case_details = run_comparison_cases(data, attacked, shared, policies)
        waiting_details.extend(case_details)
        diagnostics.append({**shared, **calculate_plan_diagnostics(data, normal_plan, duration_hours)})
        if keep_schedules:
            schedules.extend(case_schedules)
        else:
            increase = example_wait_increase(case_metrics, policies[0])
            if increase > best_increase:
                best_increase, schedules = increase, case_schedules
        metrics.extend(case_metrics)
    run_metrics = pd.DataFrame(metrics)
    return {
        "ship_data": pd.concat(inputs, ignore_index=True),
        "schedules": pd.concat(schedules, ignore_index=True),
        "run_metrics": run_metrics,
        "summary": summarize_results(run_metrics),
        "plan_diagnostics": pd.DataFrame(diagnostics),
        "waiting_details": pd.concat(waiting_details, ignore_index=True),
    }
