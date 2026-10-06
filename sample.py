"""到着時刻の改ざんがバース割当へ与える影響を、同じ船舶データで比較する。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# 時間の単位はすべて「時間」。実測値への当てはめではなく予備実験の仮定。
MEAN_INTERARRIVAL_TIME = 2.37
HANDLING_SHAPE = 2  # shape が整数のガンマ分布は Erlang 分布
HANDLING_SCALE = 2.4  # 基準荷役時間の平均は 2 × 2.4 = 4.8 時間
HANDLING_NOISE = 0.05  # バースごとに基準時間の ±5% の小さな揺らぎ
SCENARIO_NAMES = {"normal": "正常時", "early": "早く申告", "late": "遅く申告"}
METRIC_NAMES = {
    "total_waiting_time": "総待ち時間",
    "average_waiting_time": "平均待ち時間",
    "max_waiting_time": "最大待ち時間",
    "total_turnaround_time": "総在港時間",
    "average_turnaround_time": "平均在港時間",
    "completion_time": "全船の荷役完了時刻",
    "target_waiting_time": "対象船の待ち時間",
    "other_average_waiting_time": "対象船以外の平均待ち時間",
}


def generate_ship_data(num_ships=10, num_berths=3, seed=42):
    """船ごとに一つの基準荷役時間を作り、全バースで共有する。"""
    rng = np.random.default_rng(seed)
    intervals = rng.exponential(MEAN_INTERARRIVAL_TIME, num_ships)
    arrivals = np.cumsum(intervals)
    arrivals -= arrivals[0]  # 最初の船の到着を時刻0にする
    base_times = rng.gamma(HANDLING_SHAPE, HANDLING_SCALE, num_ships)
    factors = np.linspace(0.9, 1.1, num_berths) if num_berths > 1 else [1.0]

    data = pd.DataFrame({
        "ship": [f"V{i + 1}" for i in range(num_ships)],
        "actual_arrival_time": arrivals,
        "reported_arrival_time": arrivals.copy(),
        "base_handling_time": base_times,
    })
    for index, factor in enumerate(factors, 1):
        noise = rng.uniform(-HANDLING_NOISE, HANDLING_NOISE, num_ships)
        data[f"handling_time_B{index}"] = base_times * factor * (1 + noise)
    return data


def apply_attack(data, target_ship, shift_hours):
    """申告時刻だけを変更する。実到着時刻と荷役時間は変更しない。"""
    attacked = data.copy(deep=True)
    target = attacked["ship"] == target_ship
    if target.sum() != 1:
        raise ValueError(f"対象船 {target_ship} が一意に見つかりません。")
    original = attacked.loc[target, "actual_arrival_time"]
    # シミュレーション開始前の時刻を申告しないものとする。
    attacked.loc[target, "reported_arrival_time"] = (original + shift_hours).clip(lower=0)
    return attacked


def assign_berths(data):
    """申告時刻でFCFS計画を作り、実到着時刻でその計画を実行する。"""
    berth_columns = [c for c in data.columns if c.startswith("handling_time_B")]
    planned_available = np.zeros(len(berth_columns))
    plan = []

    # 同時刻なら元データの船番号順を維持する。
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

    # 割当バースと順番は固定。早く申告しても実到着前には荷役できない。
    actual_available = np.zeros(len(berth_columns))
    for row in plan:
        index = int(row["assigned_berth"][1:]) - 1
        start = max(row["planned_start_time"], row["actual_arrival_time"], actual_available[index])
        completion = start + row["handling_time"]
        actual_available[index] = completion
        row["start_time"] = start
        row["completion_time"] = completion
        row["waiting_time"] = start - row["actual_arrival_time"]
        row["turnaround_time"] = completion - row["actual_arrival_time"]
    return pd.DataFrame(plan)


def calculate_metrics(schedule, target_ship):
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
    """各条件の平均・標準偏差と、同じseedでの正常時との差を集計する。"""
    summaries = []
    comparisons = []
    normal = run_metrics[run_metrics["scenario"] == "normal"].set_index("trial")
    for scenario in SCENARIO_NAMES:
        runs = run_metrics[run_metrics["scenario"] == scenario].set_index("trial")
        for metric in METRIC_NAMES:
            values = runs[metric]
            summaries.append({
                "scenario": scenario, "metric": metric, "trials": len(values),
                "mean": values.mean(), "std": values.std(ddof=1),
            })
            if scenario == "normal":
                continue
            differences = values - normal[metric]
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


def save_plot(summary, output_dir, target_ship, shift_hours):
    """卒論用の簡単な棒グラフ。日本語フォントへの依存を避けて英語表記にする。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#54788b", "#b68045", "#986060"]
    for axis, metric, title in zip(
        axes,
        ["average_waiting_time", "other_average_waiting_time"],
        ["All ships", "Ships other than the target"],
    ):
        rows = summary[summary["metric"] == metric].set_index("scenario").loc[list(SCENARIO_NAMES)]
        bars = axis.bar(["Normal", "Early report", "Late report"], rows["mean"], color=colors)
        axis.bar_label(bars, fmt="%.3f", padding=3)
        axis.set_title(title)
        axis.set_ylabel("Average waiting time (hours)")
        axis.set_ylim(0, max(rows["mean"].max() * 1.2, 0.1))
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle(f"Mean waiting time: {target_ship} report shifted by +/-{shift_hours:g} h ({rows['trials'].iloc[0]} paired trials)")
    fig.tight_layout()
    fig.savefig(output_dir / "waiting_time_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100, help="試行回数（初期値100）")
    parser.add_argument("--ships", type=int, default=10, help="船数（初期値10）")
    parser.add_argument("--berths", type=int, default=3, help="バース数（初期値3）")
    parser.add_argument("--seed", type=int, default=42, help="最初の乱数seed（初期値42）")
    parser.add_argument("--attack-ship", default="V5", help="改ざん対象船（初期値V5）")
    parser.add_argument("--shift", type=float, default=4.0, help="早める・遅らせる時間（初期値4）")
    parser.add_argument("--output", type=Path, default=Path("results"), help="保存先（初期値results）")
    parser.add_argument("--no-plot", action="store_true", help="CSVのみ出力する")
    args = parser.parse_args()
    try:
        results = run_experiment(args.trials, args.ships, args.berths, args.seed, args.attack_ship, args.shift)
    except ValueError as error:
        parser.error(str(error))

    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(args.output / f"{name}.csv", index=False, encoding="utf-8-sig")
    config = {
        "trials": args.trials, "num_ships": args.ships, "num_berths": args.berths,
        "first_seed": args.seed, "last_seed": args.seed + args.trials - 1,
        "target_ship": args.attack_ship, "shift_hours": args.shift,
        "mean_interarrival_time": MEAN_INTERARRIVAL_TIME,
        "handling_shape": HANDLING_SHAPE, "handling_scale": HANDLING_SCALE,
        "handling_noise": HANDLING_NOISE,
        "berth_factors": np.linspace(0.9, 1.1, args.berths).tolist() if args.berths > 1 else [1.0],
        "time_unit": "hours", "execution_policy": "fixed_plan_no_rescheduling",
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
    }
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 過去のグラフが新しいCSVと混在しないようにする。
    (args.output / "waiting_time_comparison.png").unlink(missing_ok=True)
    if not args.no_plot:
        try:
            save_plot(results["summary"], args.output, args.attack_ship, args.shift)
        except ModuleNotFoundError as error:
            if error.name != "matplotlib":
                raise
            print("グラフの作成には matplotlib が必要です。CSVの保存は完了しました。")

    print(f"\n{args.trials}回の比較：船{args.ships}隻、バース{args.berths}、対象{args.attack_ship}、改ざん幅±{args.shift:g}時間")
    print("評価は実到着時刻から計算。各試行で3条件に同じ船舶データを使用。\n")
    means = results["summary"].pivot(index="scenario", columns="metric", values="mean")
    display_metrics = ["average_waiting_time", "average_turnaround_time", "target_waiting_time", "other_average_waiting_time"]
    print(means.loc[list(SCENARIO_NAMES), display_metrics].rename(index=SCENARIO_NAMES, columns=METRIC_NAMES).round(3).to_string())
    print("\n正常時からの平均待ち時間の変化（正なら悪化、負なら改善）")
    comparison = results["paired_comparison"]
    for _, row in comparison[comparison["metric"] == "average_waiting_time"].iterrows():
        print(f"{SCENARIO_NAMES[row['scenario']]}: {row['mean_difference']:+.3f}時間 / 悪化{row['increased_trials']}回、改善{row['decreased_trials']}回、変化なし{row['unchanged_trials']}回")
    print(f"\n保存先: {args.output.resolve()}")


if __name__ == "__main__":
    main()
