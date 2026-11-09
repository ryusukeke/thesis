"""実験結果から比較グラフと船のスケジュール図を作り、PNG・PDFを保存する。"""

import matplotlib
import numpy as np
import pandas as pd

from simulation import POLICIES

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICY_NAMES = {
    "fixed": "A: Fixed plan",
    "same_berth": "B: Same berth",
    "reassign": "C: Berth changes",
    "unrestricted": "D: Unrestricted",
}
COLORS = {
    "fixed": "#365c78",
    "same_berth": "#c57c28",
    "reassign": "#34836c",
    "unrestricted": "#97577c",
}


def plot_results(results, output, full_study):
    """単一設定は比較図1つ、主実験は比較図・上限比較・スケジュールの5図を保存する。"""
    plot_comparisons(results["summary"], output, full_study)
    if full_study:
        plot_advance_limits(results["summary"], output)
        plot_schedule_example(results, output)


def plot_comparisons(summary, output, full_study):
    """条件別の平均と、上側だけ標本標準偏差1個分のひげを描く。"""
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


def plot_advance_limits(summary, output):
    """前倒し上限0時間・2時間・無制限を同じ条件で比較する。"""
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


def plot_schedule_example(results, output):
    """事前指定した条件内で、固定方式の他船待ちが最大の例を描く。"""
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
