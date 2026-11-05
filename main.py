"""4バースの到着情報改ざん実験を実行し、設定・CSV・論文用グラフを保存する。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from port_collection.analyze_port_data import analyze_forecasts, save_reference
from simulation import (HANDLING_SCALE, HANDLING_SHAPE, MEAN_INTERARRIVAL_TIME,
                        POLICIES, run_experiment)

ROOT = Path(__file__).resolve().parent
POLICY_NAMES = {"fixed": "A: Fixed plan", "same_berth": "B: Same berth",
                "reassign": "C: Berth changes", "unrestricted": "D: Unrestricted"}
COLORS = {"fixed": "#365c78", "same_berth": "#c57c28", "reassign": "#34836c", "unrestricted": "#97577c"}


def plot_results(results, output, full_study):
    """集計条件を図に明示し、PNGとPDFを保存する。誤差棒の上側は試行の標本標準偏差。下側は描かない。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = results["summary"]
    main = summary[(summary.experiment_group == "main") & (summary.metric == "other_average_waiting_time")]
    position = "middle" if full_study else "specified"
    panels = [("01_attack_width", "shift_hours", main[(main.demand_multiplier == 1) & (main.target_position == position)],
               "Report shift magnitude (hours)", "Baseline demand; middle target" if full_study else "Single setting")]
    if full_study:
        panels.extend([
            ("02_demand", "demand_multiplier", main[(main.target_position == "middle") & (main.shift_hours == 4)],
             "Candidate demand / observed forecast reference", "Middle target; report shift +/-4 h"),
            ("03_target_position", "target_position", main[(main.demand_multiplier == 1) & (main.shift_hours == 4)],
             "Target position in actual arrival order", "Baseline demand; report shift +/-4 h"),
        ])
    for filename, x, data, xlabel, subtitle in panels:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
        positions = ["early", "middle", "late"] if x == "target_position" else sorted(data[x].unique())
        for axis, scenario, title in zip(axes, ["early", "late"], ["Early report", "Late report"]):
            for policy in data.policy.unique():
                rows = data[(data.scenario == scenario) & (data.policy == policy)].set_index(x).reindex(positions)
                xvalues = range(len(positions)) if x == "target_position" else positions
                deviations = rows["std"].fillna(0).to_numpy()
                axis.errorbar(xvalues, rows["mean"], yerr=np.array([np.zeros(len(rows)), deviations]),
                              marker="o", capsize=3, color=COLORS[policy], label=POLICY_NAMES[policy])
            axis.set_xticks(range(len(positions)) if x == "target_position" else positions, positions)
            axis.set_xlabel(xlabel)
            axis.set_title(title)
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("Other ships: average waiting time (hours)")
        axes[1].legend(fontsize=8)
        fig.suptitle(subtitle + "; whiskers: +1 sample SD across paired input plans", fontsize=11)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{filename}.{suffix}", dpi=180)
        plt.close(fig)
    if not full_study:
        return

    # 同じ中盤・4時間の条件で、前倒し0/2/無制限を比べる。0だけ補足実験から取る。
    subset = summary[(summary.demand_multiplier == 1) & (summary.target_position == "middle")
                     & (summary.shift_hours == 4) & (summary.metric == "other_average_waiting_time")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, scenario in zip(axes, ("early", "late")):
        values, errors = [], []
        for group, policy in (("sensitivity", "reassign"), ("main", "reassign"), ("main", "unrestricted")):
            row = subset[(subset.experiment_group == group) & (subset.policy == policy) & (subset.scenario == scenario)].iloc[0]
            values.append(row["mean"])
            errors.append(0 if pd.isna(row["std"]) else row["std"])
        axis.bar(["0 h", "2 h", "Unlimited"], values, yerr=np.array([np.zeros(len(errors)), errors]), capsize=3,
                 color=["#365c78", COLORS["reassign"], COLORS["unrestricted"]])
        axis.set_title(f"{scenario.capitalize()} report")
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Permitted start before updated planned time")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Other ships: average waiting time (hours)")
    fig.suptitle("Berth changes allowed; baseline demand; middle target; +/-4 h; +1 sample SD", fontsize=10)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"04_advance_limit.{suffix}", dpi=180)
    plt.close(fig)

    # 平均を代表する例ではない。事前指定セル内でAの他船待ちが最大の例を説明用に選ぶ。
    runs = results["run_metrics"]
    examples = runs[(runs.experiment_group == "main") & (runs.demand_multiplier == 1)
                    & (runs.target_position == "middle") & (runs.shift_hours == 4)
                    & (runs.scenario == "early") & (runs.policy == "fixed")]
    example = examples.sort_values(["other_average_waiting_time", "trial"], ascending=[False, True]).iloc[0]
    schedules = results["schedules"]
    schedules = schedules[(schedules.experiment_group == "main") & (schedules.demand_multiplier == 1)
                          & (schedules.trial == example.trial)]
    attacked = schedules[(schedules.target_position == "middle") & (schedules.shift_hours == 4)
                         & (schedules.scenario == "early")]
    target = example.target_ship
    target_arrival = attacked.loc[attacked.ship == target, "actual_arrival_time"].iloc[0]
    start, end = max(0, target_arrival - 8), target_arrival + 24
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    normal = schedules[(schedules.scenario == "normal") & (schedules.policy == "fixed")]
    for axis, (label, table) in zip(axes, [("Normal", normal)] + [(POLICY_NAMES[p], attacked[attacked.policy == p]) for p in POLICIES]):
        for _, row in table.iterrows():
            if row.completion_time <= start or row.start_time >= end:
                continue
            berth = int(row.assigned_berth[1:])
            axis.barh(berth, row.handling_time, left=row.start_time, height=0.65,
                      color="#b84e45" if row.ship == target else "#6f95ac", edgecolor="white")
            x = max(start, row.start_time)
            if min(end, row.completion_time) - x > 1:
                axis.text(x + 0.1, berth, row.ship, va="center", fontsize=7, clip_on=True)
        axis.axvline(target_arrival, color="#555555", linestyle="--", linewidth=1)
        axis.set_yticks([1, 2, 3, 4], ["B1", "B2", "B3", "B4"])
        axis.set_title(label, loc="left", fontsize=10)
        axis.grid(axis="x", alpha=0.15)
    axes[-1].set_xlim(start, end)
    axes[-1].set_xlabel("Hours since beginning; dashed line: target actual arrival")
    fig.suptitle(f"Illustrative maximum in predefined cell: seed {int(example.seed)}, {target}, early report -4 h", fontsize=11)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"05_schedule_example.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", action="store_true", help="神戸港の予定規模を参照した主実験と補足実験を実行")
    parser.add_argument("--trials", type=int, default=10)
    period = parser.add_mutually_exclusive_group()
    period.add_argument("--ships", type=int, help="単一設定の確認用の固定隻数")
    period.add_argument("--duration", type=float, default=None)
    parser.add_argument("--berths", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shift", type=float, default=4.0, help="単一設定での変更幅。主実験は2・4・8時間")
    parser.add_argument("--interval", type=float, default=None, help="候補の平均間隔。主実験の省略時は神戸港予定の集計値")
    parser.add_argument("--advance", type=float, default=2.0, help="同じバース内調整・バース変更の前倒し上限")
    parser.add_argument("--policy", choices=("all", *POLICIES), default="all")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/kobe_port/raw")
    parser.add_argument("--through", default="2026-11-04", help="参照するCSVの取得日上限")
    parser.add_argument("--output", type=Path, default=Path("results/final_20261105"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    if args.study and (args.ships is not None or args.policy != "all" or args.berths != 4
                       or args.advance != 2 or args.duration not in (None, 168.0)):
        parser.error("主実験は168時間・4バース・全方式・前倒し上限2時間で実行します。")
    if not args.no_save and args.output.exists() and any(args.output.iterdir()):
        parser.error("保存先にファイルがあります。既存結果を残すため、別の--outputを指定してください。")
    duration = None if args.ships is not None else (args.duration if args.duration is not None else 168.0)
    policies = POLICIES if args.policy == "all" else (args.policy,)
    try:
        reference = None
        if args.study:
            reference, daily, selected = analyze_forecasts(args.raw, args.through)
        interval = args.interval if args.interval is not None else (reference["mean_candidate_interval_hours"] if reference else MEAN_INTERARRIVAL_TIME)
        results = run_experiment(trials=args.trials, num_ships=args.ships if args.ships is not None else 10, num_berths=args.berths,
                                 seed=args.seed, shift_hours=args.shift, policies=policies,
                                 duration_hours=duration, mean_interarrival_time=interval,
                                 early_start_limit=args.advance, full_study=args.study)
    except ValueError as error:
        parser.error(str(error))
    if not args.no_save:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, table in results.items():
            table.to_csv(args.output / f"{name}.csv", index=False, encoding="utf-8-sig")
        if reference:
            save_reference(args.output, reference, daily, selected)
        config = {
            "model_version": "discrete_4berth_bounded_advance_v2", "full_study": args.study,
            "trials_per_cell": args.trials, "first_seed": args.seed, "last_seed": args.seed + args.trials - 1,
            "num_berths": args.berths, "num_ships": args.ships, "candidate_duration_hours": duration,
            "base_mean_candidate_interval_hours": interval, "interval_source": "explicit_argument" if args.interval is not None else ("kobe_day_ahead_forecasts" if reference else "rounded_assumption_3h"),
            "demand_multipliers": [0.75, 1.0, 1.5] if args.study else [1.0],
            "target_positions": [0.1, 0.5, 0.9] if args.study else ["V5"],
            "shift_magnitudes_hours": [2.0, 4.0, 8.0] if args.study else [args.shift],
            "main_early_start_limit_hours": args.advance, "unrestricted_early_start_limit_hours": None,
            "sensitivity": "reassign, middle target, +/-4h, limit0" if args.study else None,
            "policies": list(policies), "handling_shape": HANDLING_SHAPE, "handling_scale": HANDLING_SCALE,
            "handling_model": "base_times_1_plus_0.1_times_berth_index_distance",
            "cargo_nearest_berth": "uniform_integer_from_1_to_berths",
            "attribute_rng_seed": "trial_seed+100000; per-ship gamma then cargo position",
            "normal_arrival_adjustment": "normal_reserved_start; no upper limit; no speed or fuel model",
            "plan_update": "keep_normal_reservations; replan_once_if_report_changes",
            "initial_berth_state": "all_empty", "period_end": "finish_all_candidate_ships_including_adjusted_arrivals_after_period",
            "same_time_order": "arrivals_and_releases_and_completions_together; stable_input_order",
            "distinct_normal_plans": len(results["plan_diagnostics"]),
            "main_attack_executions": int(((results["run_metrics"].experiment_group == "main") & (results["run_metrics"].scenario != "normal")).sum()),
            "sensitivity_attack_executions": int(((results["run_metrics"].experiment_group == "sensitivity") & (results["run_metrics"].scenario != "normal")).sum()),
            "normal_metrics": "shared across shifts; repeated target-specific metrics are not independent plans",
            "numpy_version": np.__version__, "pandas_version": pd.__version__,
            "code_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                            for name in ("simulation.py", "main.py", "port_collection/analyze_port_data.py")},
            "sources": {"berth_count": "https://www.mol.co.jp/info/article/2026/0819.html",
                        "rescheduling": "https://transp-or.epfl.ch/documents/technicalReports/NitBierEra2013.pdf",
                        "information_manipulation": "https://doi.org/10.1088/1742-6596/2867/1/012047"},
        }
        (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        if not args.no_plot:
            plot_results(results, args.output, args.study)
    runs = results["run_metrics"]
    normal = runs[runs.scenario == "normal"]
    attacks = runs[runs.scenario != "normal"]
    print(f"正常計画 {len(results['plan_diagnostics'])}組、主実験の改ざん運用 {len(attacks[attacks.experiment_group == 'main'])}ケース、補足 {len(attacks[attacks.experiment_group == 'sensitivity'])}ケース")
    print(f"正常時の最大待ち時間：{normal.max_waiting_time.max():.9f}時間。到着候補の基準平均間隔：{interval:.6f}時間。")
    print("下表は全主実験条件を同じ重みで平均した概要です。条件別の比較はsummary.csvを参照してください。")
    print(attacks[attacks.experiment_group == "main"].groupby(["policy", "scenario"])[["other_average_waiting_time", "other_waiting_ships", "berth_changes"]].mean().round(4).to_string())
    print("実測の荷役時間・実待ち時間による検証ではありません。到着調整量はplan_diagnostics.csvに記録します。")
    print("保存なし" if args.no_save else f"保存先：{args.output.resolve()}")


if __name__ == "__main__":
    main()
