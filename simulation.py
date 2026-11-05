"""船舶データの生成、到着時刻の改ざん、バース割当、比較実験の計算を行う。"""

import numpy as np
import pandas as pd


# 時間の単位はすべて「時間」。実測値への当てはめではなく予備実験の仮定。
MEAN_INTERARRIVAL_TIME = 2.37
HANDLING_SHAPE = 2  # shape が整数のガンマ分布は Erlang 分布
HANDLING_SCALE = 2.4  # 基準荷役時間の平均は 2 × 2.4 = 4.8 時間
SCENARIOS = ("normal", "early", "late")
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
)


def generate_ship_data(num_ships=10, num_berths=3, seed=42,
                       mean_interarrival_time=MEAN_INTERARRIVAL_TIME, duration_hours=None):
    """未調整の到着候補・基準荷役時間・バース別荷役時間を生成する。

    初期の申告到着時刻は実到着時刻と同じにし、係数を船ごとに並べ替えて
    船とバースの相性を表す。実験ではcreate_normal_plan()で到着を調整する。
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
    base_times = rng.gamma(HANDLING_SHAPE, HANDLING_SCALE, num_ships) # 基準となる荷役時間を生成
    # 3バースなら0.9・1.0・1.1。1バースの場合は補正なしの1.0。
    factors = np.linspace(0.9, 1.1, num_berths) if num_berths > 1 else [1.0] # 0.9から1.1までを等間隔に並べる バースが4つなら[0.9, 約0.967, 約1.033, 1.1]
    # 各船が同じ係数の組を持つが、速いバースは船ごとに変わる。
    ship_factors = np.zeros((num_ships, num_berths))  # 行が船、列がバース。0隻でも表の形を保つ。
    for index in range(num_ships):
        ship_factors[index] = rng.permutation(factors)
    data = pd.DataFrame({
        "ship": [f"V{i + 1}" for i in range(num_ships)],
        "actual_arrival_time": arrivals,
        "reported_arrival_time": arrivals.copy(),
        "base_handling_time": base_times,
    })
    for index in range(1, num_berths + 1):
        factor = ship_factors[:, index - 1] # 該当列の全ての行を取り出す
        data[f"berth_factor_B{index}"] = factor
        data[f"handling_time_B{index}"] = base_times * factor
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
        row["start_time"] = start
        row["completion_time"] = completion
        row["waiting_time"] = start - row["actual_arrival_time"]
        row["turnaround_time"] = completion - row["actual_arrival_time"]
    return pd.DataFrame(rows)


def assign_berths(data):
    """申告時刻で計画を作り、固定計画の実行結果を返す。"""
    plan = create_plan(data)
    return execute_fixed_plan(plan)


def execute_adjusted_plan(data, plan, policy="reassign"):
    """到着済みの待ち船を計画順で選び、空きバースへ入れる。

    same_berthは計画バースを維持し、reassignは他バースへの変更も許す。
    到着と荷役完了の時刻を順に進め、荷役中の船は動かさない。
    """
    berth_columns = [c for c in data.columns if c.startswith("handling_time_B")]
    handling_times = data.set_index("ship")
    pending = plan.sort_values("assignment_order").to_dict("records")
    actual_available = np.zeros(len(berth_columns))
    rows = []
    current_time = 0.0

    while pending:
        # 同時刻の到着と完了をまとめて扱い、到着済みの船を計画順で確認する。
        for row in pending.copy():
            if row["actual_arrival_time"] > current_time:
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
            # 次に時計を進める。未来の到着時刻は、バース選択には使わない。
            next_times = [row["actual_arrival_time"] for row in pending
                          if row["actual_arrival_time"] > current_time]
            next_times.extend(time for time in actual_available if time > current_time)
            current_time = min(next_times)

    return pd.DataFrame(rows).sort_values("assignment_order").reset_index(drop=True)


def calculate_metrics(schedule, target_ship):
    """1試行・1条件の全船、対象船、対象船以外の評価指標8個を辞書で返す。

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
    }


def summarize_results(run_metrics):
    """ルール・条件別の集計と、同じルールの正常時との差を返す。

    差は「攻撃時−正常時」。差の平均・標準偏差・最小・最大と増減回数を集計する。
    標準偏差は標本標準偏差で、1試行の場合は計算できずNaNになる。
    """
    summaries = []
    comparisons = []
    for policy in run_metrics["policy"].unique():
        policy_runs = run_metrics[run_metrics["policy"] == policy]
        normal = policy_runs[policy_runs["scenario"] == "normal"].set_index("trial")
        for scenario in SCENARIOS:
            runs = policy_runs[policy_runs["scenario"] == scenario].set_index("trial")
            for metric in METRICS:
                values = runs[metric]
                summaries.append({
                    "policy": policy, "scenario": scenario, "metric": metric,
                    "trials": len(values), "mean": values.mean(), "std": values.std(ddof=1),
                })
                if scenario == "normal":
                    continue
                # 同じルール・同じ試行の正常値を引く。
                differences = values - normal[metric]
                comparisons.append({
                    "policy": policy, "scenario": scenario, "metric": metric,
                    "trials": len(values),
                    "normal_mean": normal[metric].mean(), "attack_mean": values.mean(),
                    "mean_difference": differences.mean(),
                    "std_difference": differences.std(ddof=1),
                    "min_difference": differences.min(), "max_difference": differences.max(),
                    "increased_trials": int((differences > 1e-9).sum()),
                    "decreased_trials": int((differences < -1e-9).sum()),
                    "unchanged_trials": int((differences.abs() <= 1e-9).sum()),
                })
    # 正常時からの増加量を、固定計画と各調整ルールで比べる。
    fixed_effects = {(row["scenario"], row["metric"]): row["mean_difference"]
                     for row in comparisons if row["policy"] == "fixed"}
    for row in comparisons:
        fixed_effect = fixed_effects.get((row["scenario"], row["metric"]), np.nan)
        row["mean_mitigation_vs_fixed"] = fixed_effect - row["mean_difference"]
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def run_experiment(trials=100, num_ships=10, num_berths=3, seed=42,
                   target_ship="V5", shift_hours=4.0, policies=("fixed",),
                   duration_hours=None, mean_interarrival_time=MEAN_INTERARRIVAL_TIME):
    """待ち0の正常計画を基準に、同じ元データで攻撃条件・ルールを比較する。

    到着候補から正常計画を確定してから申告だけを改ざんし、全船の計画を更新する。
    各条件の計画は全ルールで共有する。保存・表示は呼び出し元に任せる。
    """
    if trials < 1 or num_berths < 1 or seed < 0 or (duration_hours is None and num_ships < 2):
        raise ValueError("試行回数は1以上、船数は2以上、バース数は1以上、seedは0以上にしてください。")
    if duration_hours is None and target_ship not in [f"V{i + 1}" for i in range(num_ships)]:
        raise ValueError(f"対象船は V1 から V{num_ships} の中から選んでください。")
    if not np.isfinite(shift_hours) or shift_hours < 0:
        raise ValueError("改ざん幅は0以上の有限の値にしてください。")
    if not np.isfinite(mean_interarrival_time) or mean_interarrival_time <= 0:
        raise ValueError("平均到着間隔は0より大きい有限の値にしてください。")
    if duration_hours is not None and (not np.isfinite(duration_hours) or duration_hours <= 0):
        raise ValueError("対象期間は0より大きい有限の値にしてください。")
    if not policies or len(set(policies)) != len(policies) or any(p not in POLICIES for p in policies):
        raise ValueError("運用ルールは fixed・same_berth・reassign から重複なく指定してください。")

    inputs, plans, schedules, metrics = [], [], [], []
    for trial in range(1, trials + 1):
        trial_seed = seed + trial - 1
        data = generate_ship_data(num_ships, num_berths, trial_seed,
                                  mean_interarrival_time, duration_hours)
        if len(data) < 2 or target_ship not in data["ship"].values:
            raise ValueError(f"試行{trial}（seed={trial_seed}）は{len(data)}隻で、比較対象{target_ship}を含められません。期間・到着間隔を見直してください。")
        data, normal_plan = create_normal_plan(data)
        inputs.append(data.assign(trial=trial, seed=trial_seed))
        normal_schedules = {}
        for scenario, shift in (("normal", 0.0), ("early", -shift_hours), ("late", shift_hours)):
            scenario_data = apply_attack(data, target_ship, shift)
            # 変更幅0（または0への丸めで情報が変わらない場合）も同じ予約を使う。
            if scenario_data["reported_arrival_time"].equals(data["reported_arrival_time"]):
                plan = normal_plan.copy(deep=True)
            else:
                plan = create_plan(scenario_data)
            plans.append(plan.assign(trial=trial, seed=trial_seed, scenario=scenario))
            for policy in policies:
                if policy == "fixed":
                    schedule = execute_fixed_plan(plan)
                    schedule["planned_berth"] = schedule["assigned_berth"]
                else:
                    schedule = execute_adjusted_plan(scenario_data, plan, policy)
                if scenario == "normal":
                    # 失敗した試行を選び直したり、計算済みの待ちを0に置き換えたりしない。
                    if (schedule["waiting_time"].abs() > 1e-9).any():
                        raise ValueError(f"試行{trial}・{policy}：正常計画の待ち時間が0ではありません。")
                    normal_schedules[policy] = schedule.set_index("ship")
                # 船番号で、そのルールの正常時と対応づける。
                for column in ("waiting_time", "turnaround_time"):
                    baseline = schedule["ship"].map(normal_schedules[policy][column])
                    schedule[f"{column}_change"] = schedule[column] - baseline
                schedule["is_target"] = schedule["ship"] == target_ship
                schedule["arrival_shift"] = schedule["reported_arrival_time"] - schedule["actual_arrival_time"]
                schedules.append(schedule.assign(trial=trial, seed=trial_seed, scenario=scenario, policy=policy))
                metrics.append({
                    "trial": trial, "seed": trial_seed, "scenario": scenario, "policy": policy,
                    "num_ships": len(data),
                    "waiting_ships": int((schedule["waiting_time"] > 1e-9).sum()),
                    **calculate_metrics(schedule, target_ship),
                })
    run_metrics = pd.DataFrame(metrics)
    summary, comparison = summarize_results(run_metrics)
    return {
        "ship_data": pd.concat(inputs, ignore_index=True),
        "plans": pd.concat(plans, ignore_index=True),
        "schedules": pd.concat(schedules, ignore_index=True),
        "run_metrics": run_metrics, "summary": summary, "paired_comparison": comparison,
    }
