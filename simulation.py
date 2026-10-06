"""船舶データの生成、到着時刻の改ざん、バース割当、比較実験の計算を行う。"""

import numpy as np
import pandas as pd


# 時間の単位はすべて「時間」。実測値への当てはめではなく予備実験の仮定。
MEAN_INTERARRIVAL_TIME = 2.37
HANDLING_SHAPE = 2  # shape が整数のガンマ分布は Erlang 分布
HANDLING_SCALE = 2.4  # 基準荷役時間の平均は 2 × 2.4 = 4.8 時間
HANDLING_NOISE = 0.05  # 船とバースごとに、基準時間×係数を±5%補正する
SCENARIOS = ("normal", "early", "late")
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


def generate_ship_data(num_ships=10, num_berths=3, seed=42):
    """各船の到着時刻・基準荷役時間・バース係数・バース別荷役時間を生成する。

    初期の申告到着時刻は実到着時刻と同じにし、係数を船ごとに並べ替えて
    船とバースの相性を表す。生成した船舶データをDataFrameで返す。
    """
    rng = np.random.default_rng(seed)
    # 到着間隔を累積して到着時刻にする。同じseedなら同じデータを再現できる。
    intervals = rng.exponential(MEAN_INTERARRIVAL_TIME, num_ships)
    arrivals = np.cumsum(intervals)
    arrivals -= arrivals[0]  # 最初の船の到着を時刻0にする
    base_times = rng.gamma(HANDLING_SHAPE, HANDLING_SCALE, num_ships) # 基準となる荷役時間を生成
    # 3バースなら0.9・1.0・1.1。1バースの場合は補正なしの1.0。
    factors = np.linspace(0.9, 1.1, num_berths) if num_berths > 1 else [1.0] # 0.9から1.1までを等間隔に並べる バースが4つなら[0.9, 約0.967, 約1.033, 1.1]
    # 各船が同じ係数の組を持つが、速いバースは船ごとに変わる。
    ship_factors = np.array([rng.permutation(factors) for _ in range(num_ships)])
    data = pd.DataFrame({
        "ship": [f"V{i + 1}" for i in range(num_ships)],
        "actual_arrival_time": arrivals,
        "reported_arrival_time": arrivals.copy(),
        "base_handling_time": base_times,
    })
    for index in range(1, num_berths + 1):
        factor = ship_factors[:, index - 1] # 該当列の全ての行を取り出す
        noise = rng.uniform(-HANDLING_NOISE, HANDLING_NOISE, num_ships)
        data[f"berth_factor_B{index}"] = factor
        data[f"handling_time_B{index}"] = base_times * factor * (1 + noise)
    return data


def apply_attack(data, target_ship, shift_hours):
    """対象船1隻の申告時刻を実到着時刻からずらしたコピーを返す。

    shift_hoursが負なら早い申告、正なら遅い申告。0未満の申告は0に丸める。
    元データ、実到着時刻、荷役時間、他船の申告時刻は変更しない。
    """
    attacked = data.copy(deep=True)
    target = attacked["ship"] == target_ship
    if target.sum() != 1:
        raise ValueError(f"対象船 {target_ship} が一意に見つかりません。")
    original = attacked.loc[target, "actual_arrival_time"]
    # シミュレーション開始前の時刻を申告しないものとする。
    attacked.loc[target, "reported_arrival_time"] = (original + shift_hours).clip(lower=0)
    return attacked


def assign_berths(data):
    """申告時刻でFCFS計画を作り、その固定計画の実行結果を表で返す。

    計画と実行の開始・完了時刻、割当バース、待ち時間・在港時間を記録する。
    未到着船を飛ばす順序変更や、別バースへの再割当は行わない。
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

    # 各バースの割当順を守る。バースごとに空く時刻を管理し、別バースは並行して動く。
    actual_available = np.zeros(len(berth_columns))
    for row in plan:
        index = int(row["assigned_berth"][1:]) - 1
        # 計画開始・実到着・同じバースの前船の完了をすべて満たしてから開始する。
        start = max(row["planned_start_time"], row["actual_arrival_time"], actual_available[index])
        completion = start + row["handling_time"]
        actual_available[index] = completion
        row["start_time"] = start
        row["completion_time"] = completion
        row["waiting_time"] = start - row["actual_arrival_time"]
        row["turnaround_time"] = completion - row["actual_arrival_time"]
    return pd.DataFrame(plan)


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
    """条件別の集計表と、試行を対応づけた正常時との差の比較表を返す。

    差は「攻撃時−正常時」。差の平均・標準偏差・最小・最大と増減回数を集計する。
    標準偏差は標本標準偏差で、1試行の場合は計算できずNaNになる。
    """
    summaries = []
    comparisons = []
    normal = run_metrics[run_metrics["scenario"] == "normal"].set_index("trial")
    for scenario in SCENARIOS:
        runs = run_metrics[run_metrics["scenario"] == scenario].set_index("trial")
        for metric in METRICS:
            values = runs[metric]
            summaries.append({
                "scenario": scenario, "metric": metric, "trials": len(values),
                "mean": values.mean(), "std": values.std(ddof=1),
            })
            if scenario == "normal":
                continue
            # trialを索引にして同じ元データの結果同士を引き算する。
            differences = values - normal[metric]
            # 小数の計算誤差を増減と数えないよう、±1e-9時間以内は変化なしとする。
            comparisons.append({
                "scenario": scenario, "metric": metric, "trials": len(values),
                "normal_mean": normal[metric].mean(), "attack_mean": values.mean(),
                "mean_difference": differences.mean(),
                "std_difference": differences.std(ddof=1),
                "min_difference": differences.min(), "max_difference": differences.max(),
                "increased_trials": int((differences > 1e-9).sum()),
                "decreased_trials": int((differences < -1e-9).sum()),
                "unchanged_trials": int((differences.abs() <= 1e-9).sum()),
            })
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def run_experiment(trials=100, num_ships=10, num_berths=3, seed=42,
                   target_ship="V5", shift_hours=4.0):
    """試行ごとにseedを1増やし、正常・早い申告・遅い申告の固定計画を比較する。

    同じ試行では実到着と荷役時間を共有し、対象船の申告時刻だけを変える。
    元データ、船別スケジュール、試行別指標、集計、正常時との差の5表を辞書で返す。
    ファイル保存と画面表示は呼び出し元に任せる。
    """
    if trials < 1 or num_ships < 2 or num_berths < 1 or seed < 0:
        raise ValueError("試行回数は1以上、船数は2以上、バース数は1以上、seedは0以上にしてください。")
    if target_ship not in [f"V{i + 1}" for i in range(num_ships)]:
        raise ValueError(f"対象船は V1 から V{num_ships} の中から選んでください。")
    if not np.isfinite(shift_hours) or shift_hours < 0:
        raise ValueError("改ざん幅は0以上の有限の値にしてください。")

    inputs, schedules, metrics = [], [], []
    for trial in range(1, trials + 1):
        trial_seed = seed + trial - 1
        data = generate_ship_data(num_ships, num_berths, trial_seed)
        inputs.append(data.assign(trial=trial, seed=trial_seed))
        normal = assign_berths(data).set_index("ship")
        for scenario, shift in (("normal", 0.0), ("early", -shift_hours), ("late", shift_hours)):
            scenario_data = apply_attack(data, target_ship, shift)
            schedule = assign_berths(scenario_data)
            # 船番号で対応づける。改ざん後の行番号同士を比較しない。
            for column in ("waiting_time", "turnaround_time"):
                baseline = schedule["ship"].map(normal[column])
                schedule[f"{column}_change"] = schedule[column] - baseline
            schedule["is_target"] = schedule["ship"] == target_ship
            schedule["arrival_shift"] = schedule["reported_arrival_time"] - schedule["actual_arrival_time"]
            schedules.append(schedule.assign(trial=trial, seed=trial_seed, scenario=scenario))
            metrics.append({
                "trial": trial, "seed": trial_seed, "scenario": scenario,
                **calculate_metrics(schedule, target_ship),
            })
    run_metrics = pd.DataFrame(metrics)
    summary, comparison = summarize_results(run_metrics)
    return {
        "ship_data": pd.concat(inputs, ignore_index=True),
        "schedules": pd.concat(schedules, ignore_index=True),
        "run_metrics": run_metrics, "summary": summary, "paired_comparison": comparison,
    }
