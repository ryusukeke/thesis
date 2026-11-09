"""船舶データの生成、到着時刻の改ざん、バース割当、比較実験の計算を行う。"""

import numpy as np
import pandas as pd


# main.pyで使う実験条件。コマンド引数からは変更しない。
NUM_BERTHS = 4
DURATION_HOURS = 168.0
SHIFT_HOURS = 4.0  # 通常実行の申告変更幅（早い・遅いの両方向）
STUDY_SHIFT_HOURS = (2.0, 4.0, 8.0)  # 主実験で比較する変更幅
EARLY_START_LIMIT = 2.0  # 同一バース内調整・バース変更の前倒し上限

# 時間の単位はすべて「時間」。実測値への当てはめではなく予備実験の仮定。
MEAN_INTERARRIVAL_TIME = 3.0
HANDLING_SHAPE = 2  # shape が整数のガンマ分布は Erlang 分布
HANDLING_SCALE = 2.4  # 基準荷役時間の平均は 2 × 2.4 = 4.8 時間
POLICIES = ("fixed", "same_berth", "reassign", "unrestricted")
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
                       mean_interarrival_time=MEAN_INTERARRIVAL_TIME, duration_hours=None):
    """到着候補、基準荷役時間、荷物の最寄りバースを乱数で作る。

    荷役係数は1 + 0.1 × 最寄りバースからの番号の差。速度や実距離は扱わない。
    実験ではcreate_normal_plan()で到着を正常計画の開始へ調整する。
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
    # 到着候補の隻数が違っても、同じseedなら先頭から同じ船の属性を使う。
    ship_rng = np.random.default_rng(seed + 100000)
    base_times, cargo_berths = [], []
    for _ in range(num_ships):
        base_times.append(ship_rng.gamma(HANDLING_SHAPE, HANDLING_SCALE))
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


def create_normal_plan(data):
    """到着候補から予約を作り、到着を荷役開始に合わせた元データと計画を返す。

    調整は攻撃前に1回だけ行う。速度や燃料は計算せず、到着候補を
    後ろへ調整した時間を記録する。入力と荷役時間は変更しない。
    """
    normal_plan = create_plan(data)
    start_times = normal_plan.set_index("ship")["planned_start_time"]
    coordinated = data.copy(deep=True)
    coordinated["candidate_arrival_time"] = data["actual_arrival_time"]
    coordinated["actual_arrival_time"] = coordinated["ship"].map(start_times)
    coordinated["reported_arrival_time"] = coordinated["actual_arrival_time"]
    coordinated["arrival_adjustment_time"] = (
        coordinated["actual_arrival_time"] - coordinated["candidate_arrival_time"]
    )
    # バース・順番・予約時刻を保持し、到着情報だけを調整後の値にそろえる。
    normal_plan["actual_arrival_time"] = normal_plan["planned_start_time"]
    normal_plan["reported_arrival_time"] = normal_plan["planned_start_time"]
    return coordinated, normal_plan


def execute_fixed_plan(plan):
    """計画のバース・順番・開始時刻を守り、実際の荷役を計算する。"""
    rows = plan.to_dict("records")
    num_berths = max(int(row["assigned_berth"][1:]) for row in rows)

    # 各バースの割当順を守る。バースごとに空く時刻を管理し、別バースは並行して動く。
    actual_available = np.zeros(num_berths) # 指定した数の0をnp.zerosで作る
    for row in rows:
        index = int(row["assigned_berth"][1:]) - 1
        # 計画開始・実到着・同じバースの前船の完了をすべて満たしてから開始する。
        start = max(row["planned_start_time"], row["actual_arrival_time"], actual_available[index])
        completion = start + row["handling_time"]
        actual_available[index] = completion
        row["planned_berth"] = row["assigned_berth"]
        row["earliest_start_time"] = max(row["planned_start_time"], row["actual_arrival_time"])
        row["start_time"] = start
        row["completion_time"] = completion
        row["waiting_time"] = start - row["actual_arrival_time"]
        row["turnaround_time"] = completion - row["actual_arrival_time"]
    return pd.DataFrame(rows)


def execute_adjusted_plan(data, plan, policy="reassign", early_start_limit=EARLY_START_LIMIT):
    """到着済みの待ち船を計画順で選び、空きバースへ入れる。

    same_berthは計画バースを維持し、reassignは他バースへの変更も許す。
    計画開始からearly_start_limit時間まで前倒しできる。Noneなら制限なし。
    到着・開始可能時刻・荷役完了のイベントを進め、荷役中の船は動かさない。
    """
    berth_columns = [c for c in data.columns if c.startswith("handling_time_B")]
    handling_times = data.set_index("ship")
    pending = plan.sort_values("assignment_order").to_dict("records")
    for row in pending:
        earliest = (row["actual_arrival_time"] if early_start_limit is None
                    else max(row["actual_arrival_time"], row["planned_start_time"] - early_start_limit))
        row["earliest_start_time"] = earliest
    actual_available = np.zeros(len(berth_columns))
    rows = []
    current_time = 0.0

    while pending:
        # 同時刻の到着と完了をまとめて扱い、到着済みの船を計画順で確認する。
        for row in pending.copy():
            if row["earliest_start_time"] > current_time:
                continue
            candidates = []
            for index, column in enumerate(berth_columns):
                if actual_available[index] > current_time:
                    continue
                if policy == "same_berth" and row["assigned_berth"] != f"B{index + 1}":
                    continue
                handling = handling_times.loc[row["ship"], column]
                candidates.append((handling, index))
            if not candidates:
                continue

            # この船について最短の荷役時間、同じならバース番号の順で選ぶ。
            handling, index = min(candidates)
            result = row.copy()
            result["planned_berth"] = row["assigned_berth"]
            result["assigned_berth"] = f"B{index + 1}"
            result["handling_time"] = handling
            result["start_time"] = current_time
            result["completion_time"] = current_time + handling
            result["waiting_time"] = current_time - row["actual_arrival_time"]
            result["turnaround_time"] = result["completion_time"] - row["actual_arrival_time"]
            actual_available[index] = result["completion_time"]
            rows.append(result)
            pending.remove(row)

        if pending:
            # 開始可能時刻もイベントに含める。未到着船をバース選択には使わない。
            next_times = [row["earliest_start_time"] for row in pending
                          if row["earliest_start_time"] > current_time]
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
    summaries, comparisons = [], []
    for key, rows in joined.groupby(cells, sort=False):
        labels = dict(zip(cells, key))
        for metric in METRICS:
            values, baseline = rows[metric], rows[f"{metric}_normal"]
            differences = values - baseline
            summaries.append({**labels, "metric": metric, "trials": len(rows),
                              "mean": values.mean(), "std": values.std(ddof=1)})
            comparisons.append({**labels, "metric": metric, "trials": len(rows),
                                "normal_mean": baseline.mean(), "attack_mean": values.mean(),
                                "mean_difference": differences.mean(), "std_difference": differences.std(ddof=1),
                                "min_difference": differences.min(), "max_difference": differences.max(),
                                "increased_trials": int((differences > 1e-9).sum()),
                                "decreased_trials": int((differences < -1e-9).sum()),
                                "unchanged_trials": int((differences.abs() <= 1e-9).sum())})
    comparison = pd.DataFrame(comparisons)
    match = [c for c in cells if c != "policy"] + ["metric"]
    fixed = comparison[comparison["policy"] == "fixed"].set_index(match)["mean_difference"]
    comparison["mean_mitigation_vs_fixed"] = (
        comparison.set_index(match).index.map(fixed) - comparison["mean_difference"].to_numpy()
    )
    return pd.DataFrame(summaries), comparison


def execute_policy(data, plan, policy, early_start_limit):
    """4方式の共通入口。自由調整だけ前倒し制限を外す。"""
    if policy == "fixed":
        return execute_fixed_plan(plan)
    limit = None if policy == "unrestricted" else early_start_limit
    return execute_adjusted_plan(data, plan, policy, limit)


def select_targets(data, full_study, target_ship):
    """到着順の10・50・90%の位置を選ぶ。対象を結果から選ばない。"""
    if not full_study:
        return [("specified", target_ship)]
    ordered = data.sort_values("actual_arrival_time", kind="stable")["ship"].tolist()
    return [(label, ordered[max(0, int(np.ceil(len(ordered) * fraction)) - 1)])
            for label, fraction in (("early", 0.1), ("middle", 0.5), ("late", 0.9))]


def calculate_plan_diagnostics(data, normal_plan, mean_interval, duration_hours):
    """正常計画の到着調整量・予約間隔・期間外到着などをまとめる。"""
    gaps = []
    for _, berth_plan in normal_plan.groupby("assigned_berth"):
        ordered = berth_plan.sort_values("planned_start_time")
        starts = ordered["planned_start_time"].to_numpy()
        completions = ordered["planned_completion_time"].to_numpy()
        gaps.extend(starts[1:] - completions[:-1])

    arrivals_after_period = 0
    if duration_hours is not None:
        arrivals_after_period = int((data["actual_arrival_time"] >= duration_hours).sum())

    return {
        "num_ships": len(data),
        "mean_candidate_interval": mean_interval,
        "mean_arrival_adjustment": data["arrival_adjustment_time"].mean(),
        "max_arrival_adjustment": data["arrival_adjustment_time"].max(),
        "last_actual_arrival": data["actual_arrival_time"].max(),
        "arrivals_after_period": arrivals_after_period,
        "mean_reserved_gap": np.mean(gaps) if gaps else np.nan,
        "zero_reserved_gaps": int((np.array(gaps) <= 1e-9).sum()),
        "reserved_work_hours": normal_plan["handling_time"].sum(),
    }


def recorded_start_limit(policy, limit):
    """結果表に残す上限値。固定は0、無制限は空欄にする。"""
    if policy == "fixed":
        return 0.0
    if policy == "unrestricted":
        return np.nan
    return limit


def run_normal_cases(data, normal_plan, shared, group, limit, policies):
    """各方式の正常運用を1回ずつ計算し、待ち0の確認と結果の記録を行う。"""
    normal_schedules = {}
    schedules = []
    for policy in policies:
        normal = execute_policy(data, normal_plan, policy, limit)
        if (normal["waiting_time"].abs() > 1e-9).any():
            raise ValueError(f"seed={shared['seed']}・{policy}：正常時の待ちが0ではありません。")
        normal_schedules[policy] = normal.set_index("ship")
        schedules.append(normal.assign(
            **shared,
            experiment_group=group,
            policy=policy,
            target_position="baseline",
            target_ship="",
            shift_hours=0.0,
            scenario="normal",
            early_start_limit_hours=recorded_start_limit(policy, limit),
        ))
    return normal_schedules, schedules


def run_attack_case(data, normal_plan, normal_schedules, labels, group, limit, policies):
    """1隻・1方向・1改ざん幅の計画を作り、各方式の運用と正常との差を計算する。"""
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
        schedule = execute_policy(attacked, plan, policy, limit)
        normal = normal_schedules[policy]
        for column in ("waiting_time", "turnaround_time"):
            normal_values = schedule["ship"].map(normal[column])
            schedule[f"{column}_change"] = schedule[column] - normal_values
        schedule["is_target"] = schedule["ship"] == target
        schedule["arrival_shift"] = schedule["reported_arrival_time"] - schedule["actual_arrival_time"]
        actual_limit = recorded_start_limit(policy, limit)
        schedules.append(schedule.assign(
            **labels, experiment_group=group, policy=policy,
            early_start_limit_hours=actual_limit,
        ))
        metrics.append({
            **labels,
            "experiment_group": group,
            "policy": policy,
            "early_start_limit_hours": actual_limit,
            "num_ships": len(data),
            **calculate_metrics(schedule, target),
        })
    return plan.assign(**labels, experiment_group=group), schedules, metrics


def run_experiment_group(data, normal_plan, shared, settings):
    """主実験または補足実験について、正常運用を共有して改ざん条件を比較する。"""
    group = settings["group"]
    limit = settings["limit"]
    policies = settings["policies"]
    targets = settings["targets"]
    widths = settings["widths"]
    normal_schedules, schedules = run_normal_cases(
        data, normal_plan, shared, group=group, limit=limit, policies=policies,
    )
    plans = []
    metrics = []
    for position, target in targets:
        for policy in policies:
            normal = normal_schedules[policy].reset_index()
            metrics.append({
                **shared,
                "experiment_group": group,
                "target_position": position,
                "target_ship": target,
                "shift_hours": 0.0,
                "scenario": "normal",
                "policy": policy,
                "num_ships": len(data),
                "early_start_limit_hours": recorded_start_limit(policy, limit),
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
                plan, case_schedules, case_metrics = run_attack_case(
                    data, normal_plan, normal_schedules, labels,
                    group=group, limit=limit, policies=policies,
                )
                plans.append(plan)
                schedules.extend(case_schedules)
                metrics.extend(case_metrics)
    return plans, schedules, metrics


def run_experiment(trials=10, num_ships=10, num_berths=NUM_BERTHS, seed=42,
                   target_ship="V5", shift_hours=SHIFT_HOURS, policies=POLICIES,
                   duration_hours=DURATION_HOURS, mean_interarrival_time=MEAN_INTERARRIVAL_TIME,
                   early_start_limit=EARLY_START_LIMIT, full_study=False):
    """条件確認 → 正常計画 → 条件別比較 → 集計の順に実験する。保存・表示は行わない。

    full_studyでは3需要水準×3対象位置×±2/4/8時間×4方式を比較する。
    補足実験は中盤・±4時間・バース変更ありで前倒し上限を0にする。
    """
    if trials < 1 or num_berths < 1 or seed < 0 or (duration_hours is None and num_ships < 2):
        raise ValueError("試行数・バース数は1以上、seedは0以上、固定隻数は2以上にしてください。")
    for value, name in ((mean_interarrival_time, "到着候補間隔"), (duration_hours, "生成期間")):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name}は有限で正の値にしてください。")
    if not np.isfinite(shift_hours) or shift_hours < 0 or not np.isfinite(early_start_limit) or early_start_limit < 0:
        raise ValueError("改ざん幅・前倒し上限は有限で0以上にしてください。")
    if not policies or len(set(policies)) != len(policies) or any(p not in POLICIES for p in policies):
        raise ValueError("比較ルールは既知の値から重複なく選んでください。")
    if full_study and (duration_hours is None or tuple(policies) != POLICIES):
        raise ValueError("主実験は期間指定・4方式すべてで実行します。")
    inputs = []
    plans = []
    schedules = []
    metrics = []
    diagnostics = []
    levels = (0.75, 1.0, 1.5) if full_study else (1.0,)
    for demand in levels:
        for trial in range(1, trials + 1):
            trial_seed = seed + trial - 1
            mean_interval = mean_interarrival_time / demand
            raw = generate_ship_data(
                num_ships=num_ships,
                num_berths=num_berths,
                seed=trial_seed,
                mean_interarrival_time=mean_interval,
                duration_hours=duration_hours,
            )
            if len(raw) < 2 or (not full_study and target_ship not in raw["ship"].values):
                raise ValueError(f"seed={trial_seed}：比較対象を含む船を生成できません。期間・間隔・隻数を見直してください。")
            data, normal_plan = create_normal_plan(raw)
            shared = {"demand_multiplier": demand, "trial": trial, "seed": trial_seed}
            inputs.append(data.assign(**shared))
            plans.append(normal_plan.assign(**shared, scenario="normal", target_position="baseline",
                                             target_ship="", shift_hours=0.0))
            diagnosis = calculate_plan_diagnostics(data, normal_plan, mean_interval, duration_hours)
            diagnostics.append({**shared, **diagnosis})
            targets = select_targets(data, full_study, target_ship)
            groups = [{
                "group": "main", "limit": early_start_limit, "targets": targets,
                "widths": STUDY_SHIFT_HOURS if full_study else (shift_hours,),
                "policies": policies,
            }]
            if full_study:
                groups.append({
                    "group": "sensitivity", "limit": 0.0, "targets": [targets[1]],
                    "widths": (4.0,), "policies": ("reassign",),
                })
            for settings in groups:
                group_plans, group_schedules, group_metrics = run_experiment_group(
                    data, normal_plan, shared, settings,
                )
                plans.extend(group_plans)
                schedules.extend(group_schedules)
                metrics.extend(group_metrics)
    run_metrics = pd.DataFrame(metrics)
    summary, comparisons = summarize_results(run_metrics)
    return {
        "ship_data": pd.concat(inputs, ignore_index=True),
        "plans": pd.concat(plans, ignore_index=True),
        "schedules": pd.concat(schedules, ignore_index=True),
        "run_metrics": run_metrics,
        "summary": summary,
        "paired_comparison": comparisons,
        "plan_diagnostics": pd.DataFrame(diagnostics),
    }
