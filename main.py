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
    run_experiment,
)


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


def save_plot(summary, output_dir, target_ship, shift_hours):
    """全船と対象船以外の平均待ち時間を、3条件で比較するPNGを保存する。

    集計済みの試行平均を使う。日本語フォントへの依存を避けて英語表記にする。
    """
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
    """コマンド引数で実験を実行し、集計結果を表示する。通常は結果も保存する。

    --no-saveでは全保存処理を省略し、--no-plotではグラフ保存だけを省略する。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100, help="試行回数（初期値100）")
    parser.add_argument("--ships", type=int, default=10, help="船数（初期値10）")
    parser.add_argument("--berths", type=int, default=3, help="バース数（初期値3）")
    parser.add_argument("--seed", type=int, default=42, help="最初の乱数seed（初期値42）")
    parser.add_argument("--attack-ship", default="V5", help="改ざん対象船（初期値V5）")
    parser.add_argument("--shift", type=float, default=4.0, help="早める・遅らせる時間（初期値4）")
    parser.add_argument("--output", type=Path, default=Path("results"), help="保存先（初期値results）")
    parser.add_argument("--no-plot", action="store_true", help="グラフを作らずCSVとJSONを保存する")
    parser.add_argument("--no-save", action="store_true", help="ファイルを保存せずターミナルに結果だけ表示する")
    args = parser.parse_args()
    try:
        results = run_experiment(args.trials, args.ships, args.berths, args.seed, args.attack_ship, args.shift)
    except ValueError as error:
        parser.error(str(error))

    # 保存省略時はフォルダ作成も既存グラフの削除も行わない。
    if not args.no_save:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, table in results.items():
            table.to_csv(args.output / f"{name}.csv", index=False, encoding="utf-8-sig")
        config = {
            "trials": args.trials, "num_ships": args.ships, "num_berths": args.berths,
            "first_seed": args.seed, "last_seed": args.seed + args.trials - 1,
            "target_ship": args.attack_ship, "shift_hours": args.shift,
            "mean_interarrival_time": MEAN_INTERARRIVAL_TIME,
            "handling_shape": HANDLING_SHAPE, "handling_scale": HANDLING_SCALE,
            "handling_time_model": "base_time_times_berth_factor",
            "berth_factors": np.linspace(0.9, 1.1, args.berths).tolist() if args.berths > 1 else [1.0],
            "berth_factor_assignment": "random_permutation_per_ship",
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
    # 集計表を「行＝条件、列＝指標」に並べ替え、4指標の試行平均を表示する。
    means = results["summary"].pivot(index="scenario", columns="metric", values="mean")
    display_metrics = ["average_waiting_time", "average_turnaround_time", "target_waiting_time", "other_average_waiting_time"]
    print(means.loc[list(SCENARIO_NAMES), display_metrics].rename(index=SCENARIO_NAMES, columns=METRIC_NAMES).round(3).to_string())
    print("\n正常時からの平均待ち時間の変化（正なら悪化、負なら改善）")
    comparison = results["paired_comparison"]
    for _, row in comparison[comparison["metric"] == "average_waiting_time"].iterrows():
        print(f"{SCENARIO_NAMES[row['scenario']]}: {row['mean_difference']:+.3f}時間 / 悪化{row['increased_trials']}回、改善{row['decreased_trials']}回、変化なし{row['unchanged_trials']}回")
    if args.no_save:
        print("\nファイルは保存していません。")
    else:
        print(f"\n保存先: {args.output.resolve()}")


if __name__ == "__main__":
    main()
