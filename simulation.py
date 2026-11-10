"""船舶データの生成、到着時刻の改ざん、バース割当、比較実験の計算を行う。"""

import numpy as np
import pandas as pd


# main.pyで使う固定条件。コマンド引数からは変更しない。
NUM_BERTHS = 4
DURATION_HOURS = 168.0
SHIFT_HOURS = 4.0  # 通常実行の申告変更幅（早い・遅いの両方向）
POLICIES = ("fixed", "same_berth", "reassign")
METRICS = (
    "total_waiting_time",
    "average_waiting_time",
    "max_waiting_time",
    "total_turnaround_time",
    "average_turnaround_time",
    "completion_time",
    "target_waiting_time",
    "other_average_waiting_time",
    "other_waiting_ships",
    "other_max_waiting_time",
    "berth_changes",
)


def generate_ship_data(num_ships=10, num_berths=NUM_BERTHS, seed=42,
                       mean_interarrival_time=3.0, duration_hours=None):
    """実到着、基準荷役時間、荷物の最寄りバースを乱数で作る。

    荷役係数は1 + 0.1 × 最寄りバースからの番号の差。速度や実距離は扱わない。
    正常申告は実到着と同じにし、計画作成後も実到着を動かさない。
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


def apply_attack(data, target_ship, shift_hours):
    """対象船1隻の申告時刻を実到着時刻からずらしたコピーを返す。

    shift_hoursが負なら早い申告、正なら遅い申告。0未満の申告は0に丸める。
    元データ、実到着時刻、荷役時間、他船の申告時刻は変更しない。
    """
    attacked = data.copy(deep=True) # deep=trueにすることでデータをコピー元と共有しない
    target = attacked["ship"] == target_ship
    original = attacked.loc[target, "actual_arrival_time"] # targetがTrueの行にあるactual_arrival_time列を取り出す
    # シミュレーション開始前の時刻を申告しないものとする。
    attacked.loc[target, "reported_arrival_time"] = (original + shift_hours).clip(lower=0)
    return attacked


def create_plan(data):
    """申告時刻順で全船のバース・荷役順・計画時刻を決める。

    実到着時刻は表に記録するが、計画の判断には使わない。
    正常条件でも改ざん条件でも、この同じ計算を使う。
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


def calculate_metrics(schedule, target_ship):
    """1試行・1条件の全船、対象船、対象船以外の評価指標11個を辞書で返す。

    待ち時間は実到着から開始まで、在港時間は実到着から完了までを表す。
    全船の荷役完了時刻は、共通の時刻軸上での最大完了時刻。
    """
    target = schedule["ship"] == target_ship
    return {
        "total_waiting_time": schedule["waiting_time"].sum(),
        "average_waiting_time": schedule["waiting_time"].mean(),
        "max_waiting_time": schedule["waiting_time"].max(),
        "total_turnaround_time": schedule["turnaround_time"].sum(),
        "average_turnaround_time": schedule["turnaround_time"].mean(),
        "completion_time": schedule["completion_time"].max(),
        "target_waiting_time": schedule.loc[target, "waiting_time"].iloc[0],
        "other_average_waiting_time": schedule.loc[~target, "waiting_time"].mean(),
        "other_waiting_ships": int((schedule.loc[~target, "waiting_time"] > 1e-9).sum()),
        "other_max_waiting_time": schedule.loc[~target, "waiting_time"].max(),
        "berth_changes": int((schedule["assigned_berth"] != schedule["planned_berth"]).sum()),
    }


def summarize_results(run_metrics):
    """同じ入力・ルールの正常値との差を集計する。幅が違っても正常計画は共有する。"""
    normal_keys = ["experiment_group", "demand_multiplier", "trial", "target_position", "policy"]
    normal = run_metrics[run_metrics["scenario"] == "normal"].set_index(normal_keys)
    attacks = run_metrics[run_metrics["scenario"] != "normal"].copy()
    joined = attacks.join(normal[list(METRICS)], on=normal_keys, rsuffix="_normal")
    cells = ["experiment_group", "demand_multiplier", "target_position", "shift_hours", "scenario", "policy"]
    summaries = []
    for key, rows in joined.groupby(cells, sort=False):
        labels = dict(zip(cells, key))
        for metric in METRICS:
            values, baseline = rows[metric], rows[f"{metric}_normal"]
            differences = values - baseline
            summaries.append({**labels, "metric": metric, "trials": len(rows),
                                "mean": values.mean(), "std": values.std(ddof=1),
                                "normal_mean": baseline.mean(),
                                "mean_difference": differences.mean(), "std_difference": differences.std(ddof=1),
                                "min_difference": differences.min(), "max_difference": differences.max(),
                                "increased_trials": int((differences > 1e-9).sum()),
                                "decreased_trials": int((differences < -1e-9).sum()),
                                "unchanged_trials": int((differences.abs() <= 1e-9).sum())})
    summary = pd.DataFrame(summaries)
    match = [c for c in cells if c != "policy"] + ["metric"]
    fixed = summary[summary["policy"] == "fixed"].set_index(match)["mean_difference"]
    summary["mean_mitigation_vs_fixed"] = (
        summary.set_index(match).index.map(fixed) - summary["mean_difference"].to_numpy()
    )
    return summary


def execute_policy(data, plan, policy):
    """3方式の共通入口。調整する2方式には前倒し時間の制限を設けない。"""
    if policy == "fixed":
        return execute_fixed_plan(plan)
    return execute_adjusted_plan(data, plan, policy)


def select_targets(data, full_study, target_ship):
    """到着順の10・50・90%の位置を選ぶ。対象を結果から選ばない。"""
    if not full_study:
        return [("specified", target_ship)]
    ordered = data.sort_values("actual_arrival_time", kind="stable")["ship"].tolist()
    return [(label, ordered[max(0, int(np.ceil(len(ordered) * fraction)) - 1)])
            for label, fraction in (("early", 0.1), ("middle", 0.5), ("late", 0.9))]


def calculate_plan_diagnostics(data, normal_plan):
    """設定の記録に使う、最後の実到着と正常計画の最終完了をまとめる。"""
    return {
        "last_actual_arrival": data["actual_arrival_time"].max(),
        "last_planned_completion": normal_plan["planned_completion_time"].max(),
    }


def run_normal_cases(data, normal_plan, shared, policies):
    """各方式の正常運用を1回ずつ計算する。正常時にも待ちを認める。"""
    normal_schedules = {}
    schedules = []
    for policy in policies:
        normal = execute_policy(data, normal_plan, policy)
        normal_schedules[policy] = normal
        schedules.append(normal.assign(
            **shared,
            experiment_group="main",
            policy=policy,
            target_position="baseline",
            target_ship="",
            shift_hours=0.0,
            scenario="normal",
        ))
    return normal_schedules, schedules


def run_attack_case(data, normal_plan, labels, policies):
    """1隻・1方向・1改ざん幅の計画を作り、各方式の運用と指標を計算する。"""
    target = labels["target_ship"]
    width = labels["shift_hours"]
    shift = -width if labels["scenario"] == "early" else width
    attacked = apply_attack(data, target, shift)
    report_changed = not attacked["reported_arrival_time"].equals(data["reported_arrival_time"])
    if report_changed:
        plan = create_plan(attacked)
    else:
        plan = normal_plan.copy(deep=True)

    schedules = []
    metrics = []
    for policy in policies:
        schedule = execute_policy(attacked, plan, policy)
        schedules.append(schedule.assign(
            **labels, experiment_group="main", policy=policy,
        ))
        metrics.append({
            **labels,
            "experiment_group": "main",
            "policy": policy,
            "num_ships": len(data),
            **calculate_metrics(schedule, target),
        })
    return schedules, metrics


def run_comparison_cases(data, normal_plan, shared, targets, widths, policies):
    """同じ入力の正常運用を共有し、対象船・改ざん幅・方向を比較する。"""
    normal_schedules, schedules = run_normal_cases(
        data, normal_plan, shared, policies,
    )
    metrics = []
    for position, target in targets:
        for policy in policies:
            normal = normal_schedules[policy]
            metrics.append({
                **shared,
                "experiment_group": "main",
                "target_position": position,
                "target_ship": target,
                "shift_hours": 0.0,
                "scenario": "normal",
                "policy": policy,
                "num_ships": len(data),
                **calculate_metrics(normal, target),
            })
        for width in widths:
            for scenario in ("early", "late"):
                labels = {
                    **shared,
                    "target_position": position,
                    "target_ship": target,
                    "shift_hours": width,
                    "scenario": scenario,
                }
                case_schedules, case_metrics = run_attack_case(
                    data, normal_plan, labels,
                    policies=policies,
                )
                schedules.extend(case_schedules)
                metrics.extend(case_metrics)
    return schedules, metrics


def run_experiment(trials=100, num_ships=10, num_berths=NUM_BERTHS, seed=42,
                   target_ship="V5", shift_hours=SHIFT_HOURS, policies=POLICIES,
                   duration_hours=DURATION_HOURS, mean_interarrival_time=3.0,
                   full_study=False):
    """条件確認 → 正常計画 → 条件別比較 → 集計の順に実験する。保存・表示は行わない。

    full_studyでは3需要水準×3対象位置×±2/4/8時間×3方式を比較する。
    実到着は生成した時刻で固定し、調整2方式の前倒し時間は制限しない。
    """
    if trials < 1 or num_berths < 1 or seed < 0 or (duration_hours is None and num_ships < 2):
        raise ValueError("試行数・バース数は1以上、seedは0以上、固定隻数は2以上にしてください。")
    for value, name in ((mean_interarrival_time, "到着間隔"), (duration_hours, "生成期間")):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name}は有限で正の値にしてください。")
    if not np.isfinite(shift_hours) or shift_hours < 0:
        raise ValueError("改ざん幅は有限で0以上にしてください。")
    if not policies or len(set(policies)) != len(policies) or any(p not in POLICIES for p in policies):
        raise ValueError("比較ルールは既知の値から重複なく選んでください。")
    if full_study and (duration_hours is None or tuple(policies) != POLICIES):
        raise ValueError("主実験は期間指定・3方式すべてで実行します。")
    inputs = []
    schedules = []
    metrics = []
    diagnostics = []
    levels = (0.75, 1.0, 1.5) if full_study else (1.0,)
    for demand in levels:
        for trial in range(1, trials + 1):
            trial_seed = seed + trial - 1
            mean_interval = mean_interarrival_time / demand
            data = generate_ship_data(
                num_ships=num_ships,
                num_berths=num_berths,
                seed=trial_seed,
                mean_interarrival_time=mean_interval,
                duration_hours=duration_hours,
            )
            if len(data) < 2 or (not full_study and target_ship not in data["ship"].values):
                raise ValueError(f"seed={trial_seed}：比較対象を含む船を生成できません。期間・間隔・隻数を見直してください。")
            normal_plan = create_plan(data)
            shared = {"demand_multiplier": demand, "trial": trial, "seed": trial_seed}
            inputs.append(data.assign(**shared))
            diagnosis = calculate_plan_diagnostics(data, normal_plan)
            diagnostics.append({**shared, **diagnosis})
            targets = select_targets(data, full_study, target_ship)
            widths = (2.0, 4.0, 8.0) if full_study else (shift_hours,)
            case_schedules, case_metrics = run_comparison_cases(
                data, normal_plan, shared, targets, widths, policies,
            )
            schedules.extend(case_schedules)
            metrics.extend(case_metrics)
    run_metrics = pd.DataFrame(metrics)
    summary = summarize_results(run_metrics)
    return {
        "ship_data": pd.concat(inputs, ignore_index=True),
        "schedules": pd.concat(schedules, ignore_index=True),
        "run_metrics": run_metrics,
        "summary": summary,
        "plan_diagnostics": pd.DataFrame(diagnostics),
    }
