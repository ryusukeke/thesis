"""1隻の予定ずれの方式比較と、説明用試行の運用を描く。"""

import matplotlib
import numpy as np
import pandas as pd

from simulation import POLICIES

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICY_NAMES = {"fixed": "A: Fixed plan", "same_berth": "B: Same berth", "reassign": "C: Berth changes"}
COLORS = {policy: f"C{index}" for index, policy in enumerate(POLICIES)}
PLOT_STYLE = {"figure.titlesize": 11, "axes.titlesize": 10, "legend.fontsize": 8,
              "grid.alpha": 0.2, "savefig.dpi": 180}
FIGURE_CAPTIONS = {
    "01_waiting_comparison": "Figure 1. Change in total waiting time by arrival frequency and ETA delay",
    "02_queue_example": "Figure 2. Numbers of ships waiting over time",
    "03_schedule_example": "Figure 3. Matched and shifted handling schedules",
}


def plot_results(results, output):
    """ずれなし比較・待ち船数・荷役スケジュールの3図をPNGで保存する。"""
    with plt.rc_context(PLOT_STYLE):
        plot_waiting_comparison(results, output)
        plot_queue_example(results["schedules"], output)
        plot_schedule_example(results["schedules"], output)


def save_figure(fig, output, filename, note):
    """図番号・タイトル・条件を図の下へ置き、PNGで保存する。"""
    caption = FIGURE_CAPTIONS[filename] + "\n" + note
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=11)
    fig.tight_layout(rect=(0, 0.65 / fig.get_figheight(), 1, 1))
    fig.savefig(output / f"{filename}.png")
    plt.close(fig)


def plot_waiting_comparison(results, output):
    """到着頻度ごとに、全船の総待ち時間の対応差を2・8時間で比較する。"""
    data = results["summary"].query("metric == 'total_waiting_time'")
    demands = sorted(data.demand_multiplier.unique())
    shifts = sorted(data.shift_hours.unique())
    fig, panels = plt.subplots(1, len(demands), figsize=(4.2 * len(demands), 4.5),
                               sharey=True, squeeze=False)
    axes = panels[0]
    for axis, demand in zip(axes, demands):
        for index, policy in enumerate(POLICIES):
            rows = data[(data.demand_multiplier == demand) & (data.policy == policy)]
            positions = np.arange(len(shifts)) + (index - 1) * 0.24
            differences = rows.set_index("shift_hours").reindex(shifts).mean_difference
            axis.bar(positions, differences, width=0.24, color=COLORS[policy], label=POLICY_NAMES[policy])
            unchanged = differences.abs() <= 1e-9
            zero_markers = axis.scatter(positions[unchanged], np.zeros(unchanged.sum()),
                                        color=COLORS[policy], s=16, clip_on=False, zorder=3)
            zero_markers.set_in_layout(False)
        axis.axhline(0, color="0.45", linewidth=0.8)
        axis.set(title=f"Arrival frequency ×{demand:g}", xlabel="ETA delay (hours)")
        axis.set_xticks(range(len(shifts)), shifts)
        axis.set_axisbelow(True)
        axis.grid(axis="y")
    axes[0].set_ylabel("Change in total waiting time (hours)")
    axes[-1].legend()
    trials = int(data.trials.max())
    trial_label = "1 trial per condition" if trials == 1 else f"Mean across {trials} trials per condition"
    save_figure(fig, output, "01_waiting_comparison",
                f"Compared with matched ETA. Dots mark zero change. {trial_label}.")


def queue_profile(schedule, end):
    """到着で+1、荷役開始で-1とし、同時刻をまとめた待ち船数を返す。"""
    times = np.r_[0.0, schedule.actual_arrival_time, schedule.start_time, end]
    counts = np.r_[0, np.ones(len(schedule)), -np.ones(len(schedule)), 0]
    changes = pd.Series(counts, index=times).groupby(level=0).sum().sort_index()
    return changes.index.to_numpy(), changes.cumsum().to_numpy()


def example_description(schedules):
    """図に使った試行のシード、到着頻度、遅れ幅、対象船を示す。"""
    first = schedules.iloc[0]
    return (f"Illustrative seed {int(first.seed)}, frequency ×{first.demand_multiplier:g}, "
            f"ETA +{schedules.shift_hours.max():g} h for {first.target_ship}")


def plot_queue_example(schedules, output):
    """同じ試行のずれなしと予定ずれあり条件の待ち船数を3方式で並べる。"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)
    end = schedules.completion_time.max()
    for axis, policy in zip(axes, POLICIES):
        for scenario, label, color in [("normal", "Matched ETA", "0.45"), ("attack", "Shifted ETA", COLORS[policy])]:
            rows = schedules[(schedules.policy == policy) & ((schedules.scenario == "normal") if scenario == "normal"
                                                          else (schedules.scenario != "normal"))]
            times, counts = queue_profile(rows, end)
            axis.step(times, counts, where="post", label=label, color=color,
                      linestyle="--" if scenario == "normal" else "-")
        axis.set(title=POLICY_NAMES[policy], ylabel="Ships waiting", ylim=(0, None))
        axis.grid(axis="y")
        axis.legend()
    axes[-1].set(xlabel="Hours since beginning", xlim=(0, end))
    save_figure(fig, output, "02_queue_example", example_description(schedules))


def draw_ship_schedule(axis, schedule, target_ship, start, end):
    """範囲と重なる荷役を描き、予定をずらす対象船を青で示す。"""
    for row in schedule.itertuples():
        if row.completion_time <= start or row.start_time >= end:
            continue
        berth = int(row.assigned_berth[1:])
        axis.barh(berth, row.handling_time, left=row.start_time, height=0.65,
                  color="C0" if row.ship == target_ship else "0.72", edgecolor="white")
        visible_start = max(start, row.start_time)
        if min(end, row.completion_time) - visible_start > 1:
            axis.text(visible_start + 0.1, berth, row.ship, va="center", fontsize=7, clip_on=True)


def plot_schedule_example(schedules, output):
    """対象船の実到着周辺の荷役を6段で描く。"""
    target_ship = schedules.target_ship.iloc[0]
    center = schedules.loc[schedules.ship == target_ship, "actual_arrival_time"].iloc[0]
    start, end = max(0, center - 12), center + 36
    fig, axes = plt.subplots(6, 1, figsize=(12, 11), sharex=True)
    for axis, (policy, scenario) in zip(axes, [(p, s) for p in POLICIES for s in ("normal", "attack")]):
        rows = schedules[(schedules.policy == policy) & ((schedules.scenario == "normal") if scenario == "normal"
                                                       else (schedules.scenario != "normal"))]
        draw_ship_schedule(axis, rows, target_ship, start, end)
        axis.set(title=POLICY_NAMES[policy] + (": Matched ETA" if scenario == "normal" else ": Shifted ETA"))
        axis.set_yticks([1, 2, 3, 4], ["B1", "B2", "B3", "B4"])
        axis.grid(axis="x")
    axes[-1].set(xlim=(start, end), xlabel="Hours since beginning; blue marks the same target ship in every panel")
    save_figure(fig, output, "03_schedule_example", example_description(schedules))
