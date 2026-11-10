"""船舶データの生成、到着時刻の改ざん、バース割当、比較実験の計算を行う。"""

import json

import numpy as np
import pandas as pd


# main.pyで使う固定条件。コマンド引数からは変更しない。
NUM_BERTHS = 4
DURATION_HOURS = 168.0
SHIFT_HOURS = 4.0  # 通常実行の申告変更幅（早い・遅いの両方向）
POLICIES = ("fixed", "same_berth", "reassign")
ATTACK_SCALES = ("single", "p10", "p30", "p50")
METRICS = (
    "total_waiting_time",
    "average_waiting_time",
    "max_waiting_time",
    "total_turnaround_time",
    "average_turnaround_time",
    "completion_time",
    "target_average_waiting_time",
    "other_average_waiting_time",
    "other_waiting_ships",
    "other_max_waiting_time",
    "berth_changes",
    "increased_waiting_ships",
    "other_increased_waiting_ships",
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


def apply_attack(data, report_shifts):
    """船名と変更量の辞書を使い、対象船全隻の申告を変えたコピーを返す。

    変更量が負なら早い申告、正なら遅い申告。0未満の申告は0に丸める。
    元データ、実到着時刻、荷役時間、他船の申告時刻は変更しない。
    """
    attacked = data.copy(deep=True) # deep=trueにすることでデータをコピー元と共有しない
    if not set(report_shifts).issubset(set(data["ship"])):
        raise ValueError("改ざん対象に存在しない船が含まれています。")
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


def calculate_metrics(schedule, target_ships, normal=None):
    """全船・対象群・対象外群の指標と、正常より待ちが増えた隻数を返す。"""
    target = schedule["ship"].isin(target_ships)
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
        "target_average_waiting_time": schedule.loc[target, "waiting_time"].mean(),
        "other_average_waiting_time": schedule.loc[~target, "waiting_time"].mean(),
        "other_waiting_ships": int((schedule.loc[~target, "waiting_time"] > 1e-9).sum()),
        "other_max_waiting_time": schedule.loc[~target, "waiting_time"].max(),
        "berth_changes": int((schedule["assigned_berth"] != schedule["planned_berth"]).sum()),
        "increased_waiting_ships": int((changes > 1e-9).sum()),
        "other_increased_waiting_ships": int((changes[~target] > 1e-9).sum()),
    }


def summarize_results(run_metrics):
    """同じ入力・ルールの正常値との差を集計する。幅が違っても正常計画は共有する。"""
    normal_keys = ["demand_multiplier", "trial", "attack_scale", "policy"]
    normal = run_metrics[run_metrics["scenario"] == "normal"].set_index(normal_keys)
    attacks = run_metrics[run_metrics["scenario"] != "normal"].copy()
    joined = attacks.join(normal[list(METRICS)], on=normal_keys, rsuffix="_normal")
    cells = ["experiment_group", "demand_multiplier", "attack_scale", "shift_hours", "scenario", "policy"]
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


def select_targets(data, full_study, target_ship, seed):
    """別の乱数で選択順を作り、包含関係のある対象群と混在の符号を返す。"""
    if not full_study:
        return [("single", [target_ship])], {target_ship: -1}
    rng = np.random.default_rng(seed + 200000)
    ordered = rng.permutation(data["ship"]).tolist()
    first_sign = int(rng.choice([-1, 1]))
    directions = {ship: first_sign if index % 2 == 0 else -first_sign
                  for index, ship in enumerate(ordered)}
    targets = [("single", ordered[:1])]
    for label, fraction in zip(ATTACK_SCALES[1:], (0.1, 0.3, 0.5)):
        count = int(np.ceil(len(data) * fraction))
        targets.append((label, ordered[:count]))
    return targets, directions


def calculate_plan_diagnostics(data, normal_plan, duration_hours):
    """期間内の正常計画のバース使用割合と、到着・完了の範囲をまとめる。"""
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


def run_normal_cases(data, normal_plan, shared, policies):
    """正常運用を3方式で1回ずつ計算し、比較用と図用の表を返す。"""
    normal_schedules = {}
    schedules = []
    for policy in policies:
        normal = execute_policy(data, normal_plan, policy)
        normal_schedules[policy] = normal
        schedules.append(normal.assign(
            **shared, experiment_group="baseline", policy=policy,
            attack_scale="baseline", target_ships="[]", shift_hours=0.0, scenario="normal",
        ))
    return normal_schedules, schedules


def run_attack_case(data, normal_plan, normal_schedules, labels, shifts, policies):
    """全対象の申告を変えて計画を1回更新し、各方式の運用と指標を計算する。"""
    attacked = apply_attack(data, shifts)
    if attacked["reported_arrival_time"].equals(data["reported_arrival_time"]):
        plan = normal_plan.copy(deep=True)
    else:
        plan = create_plan(attacked)
    actual_shifts = (attacked.set_index("ship")["reported_arrival_time"]
                     - attacked.set_index("ship")["actual_arrival_time"])
    metadata = {**labels, "report_shifts": json.dumps(
        {ship: float(actual_shifts[ship]) for ship in shifts}, separators=(",", ":"),
    )}
    schedules = []
    metrics = []
    for policy in policies:
        schedule = execute_policy(attacked, plan, policy)
        schedules.append(schedule.assign(**metadata, policy=policy))
        metrics.append({
            **metadata, "policy": policy, "num_ships": len(data),
            **calculate_metrics(schedule, list(shifts), normal_schedules[policy]),
        })
    return schedules, metrics


def run_comparison_cases(data, normal_plan, shared, targets, directions, widths, policies):
    """正常を共有し、規模・幅・方向ごとに同じ対象集合で比較する。"""
    normal_schedules, schedules = run_normal_cases(data, normal_plan, shared, policies)
    metrics = []
    for scale, ships in targets:
        group = {
            **shared, "attack_scale": scale, "target_count": len(ships),
            "actual_attack_fraction": len(ships) / len(data),
            "target_ships": json.dumps(ships, separators=(",", ":")),
        }
        for policy in policies:
            metrics.append({
                **group, "experiment_group": "baseline", "shift_hours": 0.0,
                "scenario": "normal", "report_shifts": "{}", "policy": policy,
                "num_ships": len(data), **calculate_metrics(normal_schedules[policy], ships),
            })
        scenarios = ("early", "late") if scale == "single" else ("early", "late", "mixed")
        for width in widths:
            for scenario in scenarios:
                shifts = {}
                for ship in ships:
                    sign = directions[ship] if scenario == "mixed" else (-1 if scenario == "early" else 1)
                    shifts[ship] = sign * width
                labels = {
                    **group, "experiment_group": "mixed_supplement" if scenario == "mixed" else "main",
                    "shift_hours": width, "scenario": scenario,
                }
                case_schedules, case_metrics = run_attack_case(
                    data, normal_plan, normal_schedules, labels, shifts, policies,
                )
                schedules.extend(case_schedules)
                metrics.extend(case_metrics)
    return schedules, metrics


def select_example_schedules(schedules, metrics, full_study, width, policy):
    """事前指定セルで正常からの平均待ちの増分が最大の試行だけを図用に保持する。"""
    scale, scenario = ("p30", "mixed") if full_study else ("single", "early")
    rows = pd.DataFrame(metrics)
    selected = rows[(rows.demand_multiplier == 1) & (rows.attack_scale == scale)
                    & (rows.shift_hours == width) & (rows.scenario == scenario) & (rows.policy == policy)]
    if selected.empty:
        return -np.inf, []
    normal = rows[(rows.attack_scale == scale) & (rows.scenario == "normal") & (rows.policy == policy)]
    increase = selected.average_waiting_time.iloc[0] - normal.average_waiting_time.iloc[0]
    frames = [table for table in schedules if table.scenario.iloc[0] == "normal"
              or (table.attack_scale.iloc[0] == scale and table.shift_hours.iloc[0] == width
                  and table.scenario.iloc[0] == scenario)]
    return increase, frames


def run_experiment(trials=100, num_ships=10, num_berths=NUM_BERTHS, seed=42,
                   target_ship="V5", shift_hours=SHIFT_HOURS, policies=POLICIES,
                   duration_hours=DURATION_HOURS, mean_interarrival_time=3.0,
                   full_study=False, keep_schedules=False):
    """条件確認 → 正常計画 → 条件別比較 → 集計の順に実験する。保存・表示は行わない。

    full_studyでは4規模の主実験と、3割合の早い・遅い申告の混在を比較する。
    keep_schedules=Falseでは事前指定セルの説明用1試行だけ船別運用を返す。
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
    best_increase = -np.inf
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
            shared = {"demand_multiplier": demand, "trial": trial, "seed": trial_seed,
                      "selection_seed": trial_seed + 200000 if full_study else np.nan}
            inputs.append(data.assign(**shared))
            diagnosis = calculate_plan_diagnostics(data, normal_plan, duration_hours)
            diagnostics.append({**shared, **diagnosis})
            targets, directions = select_targets(data, full_study, target_ship, trial_seed)
            widths = (2.0, 4.0, 8.0) if full_study else (shift_hours,)
            case_schedules, case_metrics = run_comparison_cases(
                data, normal_plan, shared, targets, directions, widths, policies,
            )
            if keep_schedules:
                schedules.extend(case_schedules)
            else:
                increase, example = select_example_schedules(
                    case_schedules, case_metrics, full_study, 4.0 if full_study else shift_hours, policies[0],
                )
                if increase > best_increase:
                    best_increase, schedules = increase, example
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
