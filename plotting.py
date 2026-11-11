"""規模・変更幅の比較と、待ちの負担・時刻別の待ち船数を描く。"""

import json

import matplotlib
import numpy as np
import pandas as pd

from simulation import ATTACK_SCALES, POLICIES

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICY_NAMES = {"fixed": "A: Fixed plan", "same_berth": "B: Same berth", "reassign": "C: Berth changes"}
SCENARIO_NAMES = {"early": "Early reports", "late": "Late reports", "mixed": "Mixed reports"}
SCALE_NAMES = {"single": "1 ship", "p10": "10%", "p30": "30%", "p50": "50%"}
COLORS = {policy: f"C{index}" for index, policy in enumerate(POLICIES)}
PLOT_STYLE = {"figure.titlesize": 11, "axes.titlesize": 10, "legend.fontsize": 8,
              "errorbar.capsize": 3, "grid.alpha": 0.2, "savefig.dpi": 180}


def plot_results(results, output, full_study):
    """主実験は6図、単一設定は概要・待ち船数・スケジュールの3図を保存する。"""
    with plt.rc_context(PLOT_STYLE):
        if full_study:
            plot_comparisons(results["summary"], output)
            plot_group_burden(results["summary"], output)
        plot_waiting_overview(results, output, full_study)
        plot_queue_example(results["schedules"], output)
        plot_schedule_example(results["schedules"], output)


def save_figure(fig, output, filename):
    """配置を整え、PNGとPDFで保存する。"""
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{filename}.{suffix}")
    plt.close(fig)


def setup_change_axis(axis, title, xlabel):
    """正常差の0基準、軸名とグリッドをそろえる。負の差も表示する。"""
    axis.set(title=title, xlabel=xlabel, axisbelow=True)
    axis.axhline(0, color="0.45", linewidth=0.8)
    axis.grid(axis="y")


def plot_comparisons(summary, output):
    """他条件を固定して規模・幅の断面を描く。"""
    waiting = summary[summary.metric == "average_waiting_time"]
    plot_waiting_comparison(
        waiting[(waiting.shift_hours == 4)],
        output, "01_attack_fraction", "attack_scale", "Tampered ships / all ships",
        "Report shift magnitude 4 h",
    )
    plot_waiting_comparison(
        waiting[(waiting.attack_scale == "p30")],
        output, "02_attack_width", "shift_hours", "Report shift magnitude (hours)",
        "30% tampered",
    )


def plot_waiting_comparison(data, output, filename, x_column, xlabel, subtitle):
    """各セルの対応差の平均と、上側だけ標本標準偏差1個分を描く。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    positions = list(ATTACK_SCALES) if x_column == "attack_scale" else sorted(data[x_column].unique())
    xvalues = np.arange(len(positions)) if x_column == "attack_scale" else np.array(positions)
    ticklabels = [SCALE_NAMES[value] for value in positions] if x_column == "attack_scale" else positions
    for axis, scenario in zip(axes, SCENARIO_NAMES):
        for policy in POLICIES:
            rows = data[(data.scenario == scenario) & (data.policy == policy)].set_index(x_column).reindex(positions)
            present = rows["mean_difference"].notna().to_numpy()
            values = rows.loc[present, "mean_difference"]
            deviations = rows.loc[present, "std_difference"].fillna(0)
            if rows.loc[present, "trials"].max() == 1:
                axis.plot(xvalues[present], values, marker="o", color=COLORS[policy], label=POLICY_NAMES[policy])
            else:
                axis.errorbar(xvalues[present], values, yerr=[np.zeros(len(values)), deviations],
                              marker="o", color=COLORS[policy], label=POLICY_NAMES[policy])
        setup_change_axis(axis, SCENARIO_NAMES[scenario], xlabel)
        axis.set_xticks(xvalues, ticklabels)
        axis.legend()
    axes[0].set_ylabel("All ships: change in mean waiting (hours)")
    note = "; 1 trial per condition" if data.trials.max() == 1 else "; whiskers: +1 sample SD of paired changes"
    fig.suptitle(subtitle + note)
    save_figure(fig, output, filename)


def select_overview_rows(summary):
    """概要と負担先は同じ断面にそろえる。"""
    scale = "p30" if "p30" in summary.attack_scale.values else "single"
    return summary[(summary.attack_scale == scale)
                   & (summary.shift_hours == 4)]


def plot_group_burden(summary, output):
    """対象群・対象外群・全船で待ちの増減を比べる。"""
    data = select_overview_rows(summary)
    groups = {"target_average_waiting_time": "Tampered ships", "other_average_waiting_time": "Other ships",
              "average_waiting_time": "All ships"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, scenario in zip(axes, SCENARIO_NAMES):
        for metric, label in groups.items():
            rows = data[(data.scenario == scenario) & (data.metric == metric)].set_index("policy").reindex(POLICIES)
            axis.plot(range(len(POLICIES)), rows.mean_difference, marker="o", label=label)
        setup_change_axis(axis, SCENARIO_NAMES[scenario], "Operating policy")
        axis.set_xticks(range(len(POLICIES)), ["A", "B", "C"])
        axis.legend()
    axes[0].set_ylabel("Change in group mean waiting (hours)")
    fig.suptitle("30% tampered; report shift magnitude 4 h")
    save_figure(fig, output, "03_group_burden")


def plot_waiting_overview(results, output, full_study):
    """正常時と攻撃時の絶対値を、同じ需要・規模・幅で比較する。"""
    data = select_overview_rows(results["summary"])
    data = data[data.metric == "average_waiting_time"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    baseline = data[data.scenario == "early"].set_index("policy").reindex(POLICIES)
    axes[0].plot(range(len(POLICIES)), baseline.normal_mean, marker="o", label="Normal")
    for scenario in data.scenario.unique():
        rows = data[data.scenario == scenario].set_index("policy").reindex(POLICIES)
        axes[0].plot(range(len(POLICIES)), rows["mean"], marker="o", label=SCENARIO_NAMES[scenario])
        axes[1].plot(range(len(POLICIES)), rows.mean_difference, marker="o", label=SCENARIO_NAMES[scenario])
    for axis in axes:
        axis.set_xticks(range(len(POLICIES)), ["A: Fixed", "B: Same berth", "C: Berth changes"])
        axis.grid(axis="y")
        axis.legend()
    axes[0].set(title="All ships: normal and tampered", ylabel="Mean waiting time (hours)", ylim=(0, None))
    setup_change_axis(axes[1], "Tampered minus same-policy normal", "Operating policy")
    axes[1].set_ylabel("Change in mean waiting (hours)")
    fig.suptitle( ("30% tampered" if full_study else "V5 tampered") + "; shift magnitude 4 h")
    save_figure(fig, output, "04_waiting_overview")


def queue_profile(schedule, end):
    """到着で+1、荷役開始で-1とし、同時刻をまとめた待ち船数を返す。"""
    times = np.r_[0.0, schedule.actual_arrival_time, schedule.start_time, end]
    counts = np.r_[0, np.ones(len(schedule)), -np.ones(len(schedule)), 0]
    changes = pd.Series(counts, index=times).groupby(level=0).sum().sort_index()
    return changes.index.to_numpy(), changes.cumsum().to_numpy()


def example_description(schedules):
    """保持した説明用試行の条件を見出しにする。平均的な例とは扱わない。"""
    row = schedules[schedules.scenario != "normal"].iloc[0]
    return (f"Illustrative case in predefined cell: seed {int(row.seed)}, "
            f"{SCALE_NAMES[row.attack_scale]}, {row.scenario}, shift {row.shift_hours:g} h")


def plot_queue_example(schedules, output):
    """同じ試行の正常と攻撃時の待ち船数を3方式で並べる。"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)
    end = schedules.completion_time.max()
    for axis, policy in zip(axes, POLICIES):
        for scenario, label, color in [("normal", "Normal", "0.45"), ("attack", "Tampered", COLORS[policy])]:
            rows = schedules[(schedules.policy == policy) & ((schedules.scenario == "normal") if scenario == "normal"
                                                          else (schedules.scenario != "normal"))]
            times, counts = queue_profile(rows, end)
            axis.step(times, counts, where="post", label=label, color=color,
                      linestyle="--" if scenario == "normal" else "-")
        axis.set(title=POLICY_NAMES[policy], ylabel="Ships waiting", ylim=(0, None))
        axis.grid(axis="y")
        axis.legend()
    axes[-1].set(xlabel="Hours since beginning", xlim=(0, end))
    fig.suptitle(example_description(schedules))
    save_figure(fig, output, "05_queue_example")


def draw_ship_schedule(axis, schedule, targets, start, end):
    """範囲と重なる荷役を描き、改ざん対象群を赤く示す。"""
    for row in schedule.itertuples():
        if row.completion_time <= start or row.start_time >= end:
            continue
        berth = int(row.assigned_berth[1:])
        axis.barh(berth, row.handling_time, left=row.start_time, height=0.65,
                  color="C3" if row.ship in targets else "C0", edgecolor="white")
        visible_start = max(start, row.start_time)
        if min(end, row.completion_time) - visible_start > 1:
            axis.text(visible_start + 0.1, berth, row.ship, va="center", fontsize=7, clip_on=True)


def plot_schedule_example(schedules, output):
    """固定方式の船別待ち増分が最大の船の周辺を6段で描く。"""
    attacked = schedules[(schedules.policy == "fixed") & (schedules.scenario != "normal")].set_index("ship")
    normal = schedules[(schedules.policy == "fixed") & (schedules.scenario == "normal")].set_index("ship")
    changes = attacked.waiting_time - normal.waiting_time
    center = attacked.loc[changes.idxmax(), "actual_arrival_time"]
    start, end = max(0, center - 12), center + 36
    targets = set(json.loads(attacked.target_ships.iloc[0]))
    fig, axes = plt.subplots(6, 1, figsize=(12, 11), sharex=True)
    for axis, (policy, scenario) in zip(axes, [(p, s) for p in POLICIES for s in ("normal", "attack")]):
        rows = schedules[(schedules.policy == policy) & ((schedules.scenario == "normal") if scenario == "normal"
                                                       else (schedules.scenario != "normal"))]
        draw_ship_schedule(axis, rows, targets, start, end)
        axis.set(title=POLICY_NAMES[policy] + (": Normal" if scenario == "normal" else ": Tampered"))
        axis.set_yticks([1, 2, 3, 4], ["B1", "B2", "B3", "B4"])
        axis.grid(axis="x")
    axes[-1].set(xlim=(start, end), xlabel="Hours since beginning; red: tampered ships")
    fig.suptitle(example_description(schedules))
    save_figure(fig, output, "06_schedule_example")
