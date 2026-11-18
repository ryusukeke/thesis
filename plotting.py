"""全船の予定ずれの方式比較と、説明用試行の運用を描く。"""

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
    "01_waiting_comparison": "Figure 1. Mean waiting with matched and shifted ETAs",
    "02_queue_example": "Figure 2. Numbers of ships waiting over time",
    "03_schedule_example": "Figure 3. Matched and shifted handling schedules",
}


def plot_results(results, output):
    """ずれなし比較・待ち船数・荷役スケジュールの3図をPNGとPDFで保存する。"""
    with plt.rc_context(PLOT_STYLE):
        plot_waiting_comparison(results, output)
        plot_queue_example(results["schedules"], output)
        plot_schedule_example(results["schedules"], output)


def save_figure(fig, output, filename, note):
    """図番号・タイトル・条件を図の下へ置き、PNGとPDFで保存する。"""
    caption = FIGURE_CAPTIONS[filename] + "\n" + note
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=11)
    fig.tight_layout(rect=(0, 0.65 / fig.get_figheight(), 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{filename}.{suffix}")
    plt.close(fig)


def plot_waiting_comparison(results, output):
    """平均待ちとずれなしとの差を並べ、右図には各試行の対応差も点で示す。"""
    data = results["summary"]
    data = data[data.metric == "average_waiting_time"].set_index("policy").reindex(POLICIES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    positions = np.arange(len(POLICIES))
    axes[0].bar(positions - 0.18, data.normal_mean, width=0.36, label="Matched ETA", color="0.65")
    axes[0].bar(positions + 0.18, data["mean"], width=0.36, label="Shifted ETA", color="C0")
    axes[0].set(title="Trial means", ylabel="All ships: mean waiting (hours)", ylim=(0, None))
    axes[0].legend()
    runs = results["run_metrics"]
    for index, policy in enumerate(POLICIES):
        rows = runs[runs.policy == policy].pivot(index="trial", columns="scenario", values="average_waiting_time")
        differences = rows.attack - rows.normal
        # 横方向のずらしは表示用。別の乱数を使わず、試行番号順に並べる。
        offsets = np.linspace(-0.12, 0.12, len(differences))
        axes[1].scatter(index + offsets, differences, color=COLORS[policy], alpha=0.65,
                        label="Individual trials" if index == 0 else None)
        axes[1].plot(index, differences.mean(), marker="_", markersize=28, color="black",
                     markeredgewidth=2, label="Mean change" if index == 0 else None)
    axes[1].axhline(0, color="0.45", linewidth=0.8)
    axes[1].set(title="Paired changes: shifted ETA minus matched ETA", ylabel="Change in mean waiting (hours)")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(positions, ["A: Fixed", "B: Same berth", "C: Berth changes"])
        axis.set_xlabel("Operating policy")
        axis.set_axisbelow(True)
        axis.grid(axis="y")
    save_figure(fig, output, "01_waiting_comparison",
                f"All planned arrivals shifted; independent uniform shifts in {{-8, -4, -2, +2, +4, +8}} h; {int(data.trials.max())} trials")


def queue_profile(schedule, end):
    """到着で+1、荷役開始で-1とし、同時刻をまとめた待ち船数を返す。"""
    times = np.r_[0.0, schedule.actual_arrival_time, schedule.start_time, end]
    counts = np.r_[0, np.ones(len(schedule)), -np.ones(len(schedule)), 0]
    changes = pd.Series(counts, index=times).groupby(level=0).sum().sort_index()
    return changes.index.to_numpy(), changes.cumsum().to_numpy()


def example_description(schedules):
    """説明用試行の選定方法を示す。平均的な試行とは説明しない。"""
    return (f"Illustrative seed {int(schedules.seed.iloc[0])}: largest fixed-policy mean-wait increase; "
            "all planned arrivals shifted")


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


def draw_ship_schedule(axis, schedule, shifts, start, end):
    """範囲と重なる荷役を描き、実際より早い予定情報を赤、実際より遅い予定情報を青で示す。"""
    for row in schedule.itertuples():
        if row.completion_time <= start or row.start_time >= end:
            continue
        berth = int(row.assigned_berth[1:])
        axis.barh(berth, row.handling_time, left=row.start_time, height=0.65,
                  color="C3" if shifts[row.ship] < 0 else "C0", edgecolor="white")
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
    shifts = attacked.reported_arrival_time - attacked.actual_arrival_time
    fig, axes = plt.subplots(6, 1, figsize=(12, 11), sharex=True)
    for axis, (policy, scenario) in zip(axes, [(p, s) for p in POLICIES for s in ("normal", "attack")]):
        rows = schedules[(schedules.policy == policy) & ((schedules.scenario == "normal") if scenario == "normal"
                                                       else (schedules.scenario != "normal"))]
        draw_ship_schedule(axis, rows, shifts, start, end)
        axis.set(title=POLICY_NAMES[policy] + (": Matched ETA" if scenario == "normal" else ": Shifted ETA"))
        axis.set_yticks([1, 2, 3, 4], ["B1", "B2", "B3", "B4"])
        axis.grid(axis="x")
    axes[-1].set(xlim=(start, end), xlabel="Hours since beginning; colors show ETA shift direction (also in matched panels): red = earlier, blue = later")
    save_figure(fig, output, "03_schedule_example", example_description(schedules))
