"""1隻の予定ずれの方式比較と、説明用試行の運用を描く。"""

import matplotlib
import numpy as np

from simulation import POLICIES

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICY_NAMES = {"fixed": "A: Fixed plan", "same_berth": "B: Same berth", "reassign": "C: Berth changes"}
COLORS = {policy: f"C{index}" for index, policy in enumerate(POLICIES)}
PLOT_STYLE = {"figure.titlesize": 11, "axes.titlesize": 10, "legend.fontsize": 8,
              "grid.alpha": 0.2, "savefig.dpi": 180}


def plot_results(results, output):
    """総待ち時間と到着頻度ごとの荷役スケジュールをPNGで保存する。"""
    with plt.rc_context(PLOT_STYLE):
        plot_waiting_comparison(results, output)
        by_demand = results["schedules"].groupby("demand_multiplier", sort=True)
        for figure_number, (_, schedules) in enumerate(by_demand, start=2):
            plot_schedule_example(schedules, output, figure_number)


def save_figure(fig, output, filename, title, note):
    """図番号・タイトル・条件を図の下へ置き、PNGで保存する。"""
    caption = f"{title}\n{note}"
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=11)
    fig.tight_layout(rect=(0, 0.65 / fig.get_figheight(), 1, 1))
    fig.savefig(output / f"{filename}.png")
    plt.close(fig)


def plot_waiting_comparison(results, output):
    """到着頻度ごとに、全船の総待ち時間を2・8時間で比較する。"""
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
            total_wait = rows.set_index("shift_hours").reindex(shifts)["mean"]
            axis.bar(positions, total_wait, width=0.24, color=COLORS[policy], label=POLICY_NAMES[policy])
            no_wait = total_wait.abs() <= 1e-9
            zero_markers = axis.scatter(positions[no_wait], np.zeros(no_wait.sum()),
                                        color=COLORS[policy], s=16, clip_on=False, zorder=3)
            zero_markers.set_in_layout(False)
        axis.axhline(0, color="0.45", linewidth=0.8)
        axis.set(title=f"Arrival frequency ×{demand:g}", xlabel="ETA delay (hours)")
        axis.set_xticks(range(len(shifts)), shifts)
        axis.set_axisbelow(True)
        axis.grid(axis="y")
    axes[0].set_ylabel("Total waiting time (hours)")
    axes[-1].legend()
    trials = int(data.trials.max())
    trial_label = "1 trial per condition" if trials == 1 else f"Mean across {trials} trials per condition"
    save_figure(fig, output, "01_waiting_comparison",
                "Figure 1. Total waiting time by arrival frequency and ETA delay",
                f"Dots mark zero waiting time. {trial_label}.")


def example_description(schedules):
    """図に使った試行のシード、到着頻度、遅れ幅、対象船を示す。"""
    first = schedules.iloc[0]
    return (f"Illustrative seed {int(first.seed)}, frequency ×{first.demand_multiplier:g}, "
            f"ETA +{schedules.shift_hours.max():g} h for {first.target_ship}")


def draw_ship_schedule(axis, schedule, target_ship, end):
    """荷役の棒と船IDを描き、修正後の計画からずれた船を赤くする。"""
    label_min_hours = end / 100
    for row in schedule.itertuples():
        berth = int(row.assigned_berth[1:])
        is_target = row.ship == target_ship
        differs_from_plan = (
            row.assigned_berth != row.planned_berth
            or abs(row.start_time - row.planned_start_time) > 1e-9
            or abs(row.completion_time - row.planned_completion_time) > 1e-9
        )
        color, label_color = ("C0", "C0") if is_target else ("0.72", "0.25")
        if differs_from_plan:
            color, label_color = "#e35d5d", "#a4262c"
        axis.barh(berth, row.handling_time, left=row.start_time, height=0.50,
                  color=color, edgecolor="none" if is_target else "white", linewidth=0.5)
        center = row.start_time + row.handling_time / 2
        if row.handling_time >= label_min_hours:
            axis.text(center, berth, row.ship, ha="center", va="center", fontsize=7)
        else:
            label_x = np.clip(center, label_min_hours / 2, end - label_min_hours / 2)
            axis.text(label_x, berth + 0.33, row.ship, ha="center", va="bottom",
                      fontsize=7, color=label_color)


def plot_schedule_example(schedules, output, figure_number):
    """予定ずれを反映した計画と各方式の実行を、最後の荷役完了まで描く。"""
    target_ship = schedules.target_ship.iloc[0]
    example = schedules.iloc[0]
    # 実行結果には、3方式で共通の修正後の計画が船ごとに記録されている。
    plan = schedules.drop_duplicates("ship").assign(
        assigned_berth=lambda rows: rows.planned_berth,
        start_time=lambda rows: rows.planned_start_time,
        completion_time=lambda rows: rows.planned_completion_time,
        handling_time=lambda rows: rows.completion_time - rows.start_time,
    )
    end = max(plan.completion_time.max(), schedules.completion_time.max())
    panels = [("Updated plan (with delayed ETA)", plan)]
    for policy in POLICIES:
        rows = schedules.loc[schedules.policy.eq(policy)]
        panels.append((POLICY_NAMES[policy], rows))
    fig, axes = plt.subplots(len(panels), 1, figsize=(22, 10), sharex=True)
    for axis, (title, rows) in zip(axes, panels):
        draw_ship_schedule(axis, rows, target_ship, end)
        axis.set(title=title, ylim=(0.6, 4.65))
        axis.set_yticks([1, 2, 3, 4], ["B1", "B2", "B3", "B4"])
        axis.grid(axis="x")
    axes[-1].set(xlim=(0, end),
                 xlabel="Hours since beginning  |  Red = handling differs from updated plan  |  Blue = target ship when on plan")
    filename = f"{figure_number:02d}_schedule_{example.demand_multiplier:g}x"
    title = f"Figure {figure_number}. Updated plan and actual handling with delayed ETA"
    save_figure(fig, output, filename, title, example_description(schedules))
