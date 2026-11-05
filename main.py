"""到着時刻改ざんの比較実験を実行し、結果を表示・保存する。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulation import (
    HANDLING_SCALE,
    HANDLING_SHAPE,
    MEAN_INTERARRIVAL_TIME,
    POLICIES,
    run_experiment,
)


SCENARIO_NAMES = {"normal": "正常時", "early": "早く申告", "late": "遅く申告"}
POLICY_NAMES = {"fixed": "固定計画", "same_berth": "同じバース内で調整", "reassign": "バース変更あり"}
PLOT_POLICY_NAMES = {"fixed": "Fixed plan", "same_berth": "Same berth", "reassign": "Berth changes"}
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


def save_plot(summary, output_dir, target_ship, shift_hours):
    """全船と対象船以外の平均待ち時間を、条件・ルール別のPNGにする。

    集計済みの試行平均を使う。日本語フォントへの依存を避けて英語表記にする。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#54788b", "#b68045", "#986060"]
    policies = summary["policy"].unique()
    positions = np.arange(3)
    width = 0.8 / len(policies)
    for axis, metric, title in zip(
        axes,
        ["average_waiting_time", "other_average_waiting_time"],
        ["All ships", "Ships other than the target"],
    ):
        for index, policy in enumerate(policies):
            rows = summary[(summary["metric"] == metric) & (summary["policy"] == policy)]
            rows = rows.set_index("scenario").loc[list(SCENARIO_NAMES)]
            offset = (index - (len(policies) - 1) / 2) * width
            bars = axis.bar(positions + offset, rows["mean"], width,
                            color=colors[index], label=PLOT_POLICY_NAMES[policy])
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        axis.set_xticks(positions, ["Normal", "Early report", "Late report"])
        axis.set_title(title)
        axis.set_ylabel("Average waiting time (hours)")
        axis.set_ylim(0, max(summary.loc[summary["metric"] == metric, "mean"].max() * 1.25, 0.1))
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle(f"Coordinated baseline: {target_ship} report shifted by +/-{shift_hours:g} h ({rows['trials'].iloc[0]} paired trials)")
    fig.tight_layout()
    fig.savefig(output_dir / "waiting_time_comparison.png", dpi=180)
    plt.close(fig)


def main():
    """コマンド引数で実験を実行し、集計結果を表示する。通常は結果も保存する。

    --no-saveでは全保存処理を省略し、--no-plotではグラフ保存だけを省略する。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10, help="試行回数（初期値10）")
    period = parser.add_mutually_exclusive_group()
    period.add_argument("--ships", type=int, help="期間の代わりに固定隻数を指定する")
    period.add_argument("--duration", type=float, help="到着候補を生成する期間（時間、初期値168）。調整後の到着は期間外も含む")
    parser.add_argument("--berths", type=int, default=3, help="バース数（初期値3）")
    parser.add_argument("--seed", type=int, default=42, help="最初の乱数seed（初期値42）")
    parser.add_argument("--shift", type=float, default=4.0, help="早める・遅らせる時間（初期値4）")
    parser.add_argument("--interval", type=float, default=MEAN_INTERARRIVAL_TIME,
                        help="調整前の到着候補の平均間隔（時間、初期値2.37は暫定値）")
    parser.add_argument("--policy", choices=("all", *POLICIES), default="all",
                        help="比較するルール（初期値all：3ルール）")
    parser.add_argument("--output", type=Path, default=Path("results_coordinated"), help="保存先（初期値results_coordinated）")
    parser.add_argument("--no-plot", action="store_true", help="グラフを作らずCSVとJSONを保存する")
    parser.add_argument("--no-save", action="store_true", help="ファイルを保存せずターミナルに結果だけ表示する")
    args = parser.parse_args()
    # 固定隻数なら期間は使わず、指定がなければ1週間にする。
    duration = args.duration
    num_ships = 10
    if args.ships is not None:
        duration = None
        num_ships = args.ships
    elif duration is None:
        duration = 168.0
    if args.policy == "all":
        policies = POLICIES
    else:
        policies = (args.policy,)
    try:
        results = run_experiment(trials=args.trials, num_ships=num_ships,
                                 num_berths=args.berths, seed=args.seed, shift_hours=args.shift,
                                 policies=policies, duration_hours=duration,
                                 mean_interarrival_time=args.interval)
    except ValueError as error:
        parser.error(str(error))

    # 保存省略時はフォルダ作成も既存グラフの削除も行わない。
    if not args.no_save:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, table in results.items():
            table.to_csv(args.output / f"{name}.csv", index=False, encoding="utf-8-sig")
        ship_counts = results["run_metrics"].groupby("trial")["num_ships"].first()
        ship_data = results["ship_data"]
        after_period = (ship_data["actual_arrival_time"] >= duration if duration is not None
                        else pd.Series(False, index=ship_data.index))
        config = {
            "model_version": "coordinated_arrivals_v1",
            "trials": args.trials, "num_ships": args.ships, "num_berths": args.berths,
            "duration_hours": duration, "ships_per_trial": ship_counts.to_dict(),
            "duration_applies_to": "candidate_arrival_time",
            "arrival_adjustment": "align_actual_and_reported_arrivals_with_normal_plan_start",
            "arrival_adjustment_limit_hours": None,
            "arrival_adjustment_total_hours_per_trial": ship_data.groupby("trial")["arrival_adjustment_time"].sum().to_dict(),
            "last_actual_arrival_hours_per_trial": ship_data.groupby("trial")["actual_arrival_time"].max().to_dict(),
            "adjusted_arrivals_after_period_per_trial": after_period.groupby(ship_data["trial"]).sum().to_dict(),
            "first_seed": args.seed, "last_seed": args.seed + args.trials - 1,
            "target_ship": "V5", "shift_hours": args.shift,
            "mean_interarrival_time": args.interval,
            "handling_shape": HANDLING_SHAPE, "handling_scale": HANDLING_SCALE,
            "handling_time_model": "base_time_times_berth_factor",
            "berth_factors": np.linspace(0.9, 1.1, args.berths).tolist() if args.berths > 1 else [1.0],
            "berth_factor_assignment": "random_permutation_per_ship",
            "time_unit": "hours", "policies": list(policies),
            "plan_update": "keep_normal_reservations_replan_if_report_changes_before_operations",
            "normal_waiting_time": "zero_required_for_all_policies",
            "reassignment_rule": "shortest_handling_time_among_free_berths_in_plan_order",
            "initial_berth_state": "all_empty",
            "first_arrival": "first_interarrival_after_zero" if duration is not None else "time_zero",
            "simultaneous_events": "arrivals_and_completions_together",
            "completion_after_period_end": "retain_and_finish_all_candidate_ships_including_adjusted_arrivals",
            "numpy_version": np.__version__, "pandas_version": pd.__version__,
        }
        (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # 過去のグラフが新しいCSVと混在しないようにする。
        (args.output / "waiting_time_comparison.png").unlink(missing_ok=True)
        if not args.no_plot:
            try:
                save_plot(results["summary"], args.output, "V5", args.shift)
            except ModuleNotFoundError as error:
                if error.name != "matplotlib":
                    raise
                print("グラフの作成には matplotlib が必要です。CSVの保存は完了しました。")

    ship_counts = results["run_metrics"].groupby("trial")["num_ships"].first()
    period_text = f"到着候補の生成期間{duration:g}時間" if duration is not None else f"固定{args.ships}隻"
    print(f"\n{args.trials}回の比較：{period_text}、船{ship_counts.min()}〜{ship_counts.max()}隻、バース{args.berths}、対象V5、改ざん幅±{args.shift:g}時間")
    print("評価は実到着時刻から計算。各試行で全条件・全ルールに同じ船舶データを使用。")
    print("正常計画の荷役開始に到着を合わせてから申告を改ざんします。正常時は全船の待ち0を確認済みです。")
    print("到着候補の間隔・荷役係数は仮定値です。速度・燃料は計算せず、到着調整に上限は設けていません。")
    adjustments = results["ship_data"]["arrival_adjustment_time"]
    print(f"到着候補から後ろへ調整した時間：全船平均{adjustments.mean():.3f}時間、最大{adjustments.max():.3f}時間（待ち時間とは別に記録）。")
    if duration is not None:
        beyond = int((results["ship_data"]["actual_arrival_time"] >= duration).sum())
        print(f"調整後の到着が期間末以降となった船：全試行合計{beyond}隻。除外せず荷役完了まで計算します。")
    print()
    # 集計表を「行＝条件、列＝指標」に並べ替え、4指標の試行平均を表示する。
    display_metrics = ["average_waiting_time", "average_turnaround_time", "target_waiting_time", "other_average_waiting_time"]
    for policy in policies:
        print(f"【{POLICY_NAMES[policy]}】")
        policy_summary = results["summary"][results["summary"]["policy"] == policy]
        means = policy_summary.pivot(index="scenario", columns="metric", values="mean")
        print(means.loc[list(SCENARIO_NAMES), display_metrics].rename(index=SCENARIO_NAMES, columns=METRIC_NAMES).round(3).to_string())
        normal_runs = results["run_metrics"]
        normal_runs = normal_runs[(normal_runs["policy"] == policy) & (normal_runs["scenario"] == "normal")]
        print(f"正常時に待ち時間が出た試行：{int((normal_runs['waiting_ships'] > 0).sum())}/{args.trials}回\n")
    print("\n正常時からの平均待ち時間の変化（正なら悪化、負なら改善）")
    comparison = results["paired_comparison"]
    for _, row in comparison[comparison["metric"] == "average_waiting_time"].iterrows():
        print(f"{POLICY_NAMES[row['policy']]}・{SCENARIO_NAMES[row['scenario']]}: {row['mean_difference']:+.3f}時間 / 悪化{row['increased_trials']}回、改善{row['decreased_trials']}回、変化なし{row['unchanged_trials']}回")
    if "fixed" in policies and len(policies) > 1:
        print("\n対象船以外への攻撃影響の軽減量（固定計画に比べて正なら軽減）")
        for _, row in comparison[comparison["metric"] == "other_average_waiting_time"].iterrows():
            if row["policy"] != "fixed" and np.isfinite(row["mean_mitigation_vs_fixed"]):
                print(f"{POLICY_NAMES[row['policy']]}・{SCENARIO_NAMES[row['scenario']]}: {row['mean_mitigation_vs_fixed']:+.3f}時間")
    if args.no_save:
        print("\nファイルは保存していません。")
    else:
        print(f"\n保存先: {args.output.resolve()}")


if __name__ == "__main__":
    main()
