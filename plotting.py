"""実験結果から比較グラフと船のスケジュール図を作り、PNG・PDFを保存する。"""

import matplotlib
import numpy as np

from simulation import POLICIES

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICY_NAMES = {
    "fixed": "A: Fixed plan",
    "same_berth": "B: Same berth",
    "reassign": "C: Berth changes",
}
# 表示設定。C0、C1、…はMatplotlib標準の色で、方式と色の対応を全図で揃える。
COLORS = {policy: f"C{index}" for index, policy in enumerate(POLICIES)}
SCHEDULE_COLORS = {"target": "C3", "other": "C0", "arrival": "0.35"}
PLOT_STYLE = {
    "figure.titlesize": 11,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "errorbar.capsize": 3,
    "grid.alpha": 0.2,
    "savefig.dpi": 180,
}


def plot_results(results, output, full_study):
    """単一設定は2図、主実験は条件別比較・スケジュール・概要の5図を保存する。"""
    with plt.rc_context(PLOT_STYLE):  # この描画中だけ共通設定を使う。
        plot_comparisons(results["summary"], output, full_study)
        if full_study:
            plot_schedule_example(results, output)
        plot_waiting_overview(results, output)


def save_figure(fig, output, filename):
    """配置を整え、同じ図をPNGとPDFで保存する。"""
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{filename}.{suffix}")
    plt.close(fig)


def setup_waiting_axis(axis, scenario, xlabel, *, xticks=None, xticklabels=None):
    """待ち時間の比較図に共通する軸ラベル・範囲・グリッドを設定する。"""
    axis.set(title=f"{scenario.capitalize()} report", xlabel=xlabel, axisbelow=True)
    axis.axhline(0, color="0.45", linewidth=0.8)
    axis.grid(axis="y")
    if xticks is not None:
        axis.set_xticks(xticks, xticklabels)


def setup_schedule_axes(axes, target_arrival, start, end):
    """スケジュール図の見出し・バース目盛り・到着線・表示範囲を設定する。"""
    labels = [f"{POLICY_NAMES[policy]}: {scenario}"
              for policy in POLICIES for scenario in ("Normal", "Tampered")]
    for axis, label in zip(axes, labels):
        axis.axvline(target_arrival, color=SCHEDULE_COLORS["arrival"], linestyle="--", linewidth=1)
        axis.set_yticks([1, 2, 3, 4], ["B1", "B2", "B3", "B4"])
        axis.set_title(label, loc="left")
        axis.grid(axis="x")
    axes[-1].set(xlim=(start, end), xlabel="Hours since beginning; dashed line: target actual arrival")


def plot_comparisons(summary, output, full_study):
    """改ざん幅・需要・対象位置の比較に、共通の描画処理を使う。"""
    main = summary[
        (summary.experiment_group == "main")
        & (summary.metric == "other_average_waiting_time")
    ]
    position = "middle" if full_study else "specified"
    baseline_demand = main[main.demand_multiplier == 1]
    plot_waiting_comparison(
        baseline_demand[baseline_demand.target_position == position], output, "01_attack_width",
        x_column="shift_hours", xlabel="Report shift magnitude (hours)",
        subtitle="Baseline demand; middle target" if full_study else "Single setting",
    )
    if not full_study:
        return

    middle_target = main[main.target_position == "middle"]
    plot_waiting_comparison(
        middle_target[middle_target.shift_hours == 4], output, "02_demand",
        x_column="demand_multiplier", xlabel="Arrival demand / forecast reference",
        subtitle="Middle target; report shift +/-4 h",
    )
    plot_waiting_comparison(
        baseline_demand[baseline_demand.shift_hours == 4], output, "03_target_position",
        x_column="target_position", xlabel="Target position in actual arrival order",
        subtitle="Baseline demand; report shift +/-4 h",
    )


def plot_waiting_comparison(data, output, filename, *, x_column, xlabel, subtitle):
    """正常時からの増減の平均と、上側だけ対応差の標本標準偏差1個分を描く。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    if x_column == "target_position":
        positions = ["early", "middle", "late"]
        xvalues = range(len(positions))
    else:
        positions = sorted(data[x_column].unique())
        xvalues = positions
    for axis, scenario in zip(axes, ("early", "late")):
        for policy in data.policy.unique():
            selected = data[(data.scenario == scenario) & (data.policy == policy)]
            rows = selected.set_index(x_column).reindex(positions)
            deviations = rows["std_difference"].fillna(0).to_numpy()
            axis.errorbar(xvalues, rows["mean_difference"], yerr=np.array([np.zeros(len(rows)), deviations]),
                          marker="o", color=COLORS[policy], label=POLICY_NAMES[policy])
        setup_waiting_axis(axis, scenario, xlabel, xticks=xvalues, xticklabels=positions)
    axes[0].set_ylabel("Other ships: change from normal waiting (hours)")
    axes[1].legend()
    fig.suptitle(subtitle + "; whiskers: +1 sample SD of paired changes")
    save_figure(fig, output, filename)


def select_schedule_example(results):
    """説明用の試行を選び、その正常運用と改ざん後の運用を取り出す。"""
    # 平均を代表する例ではない。事前指定セル内でAの他船待ちの増分が最大の例を選ぶ。
    runs = results["run_metrics"]
    examples = runs[
        (runs.experiment_group == "main")
        & (runs.demand_multiplier == 1)
        & (runs.target_position == "middle")
        & (runs.shift_hours == 4)
        & (runs.scenario == "early")
        & (runs.policy == "fixed")
    ].copy()
    normal_runs = runs[
        (runs.demand_multiplier == 1) & (runs.target_position == "middle")
        & (runs.scenario == "normal") & (runs.policy == "fixed")
    ].set_index("trial")
    examples["waiting_increase"] = (
        examples.other_average_waiting_time
        - examples.trial.map(normal_runs.other_average_waiting_time)
    )
    example = examples.sort_values(["waiting_increase", "trial"], ascending=[False, True]).iloc[0]
    schedules = results["schedules"]
    schedules = schedules[
        (schedules.experiment_group == "main")
        & (schedules.demand_multiplier == 1)
        & (schedules.trial == example.trial)
    ]
    normal = schedules[schedules.scenario == "normal"]
    attacked = schedules[
        (schedules.target_position == "middle")
        & (schedules.shift_hours == 4)
        & (schedules.scenario == "early")
    ]
    return example, normal, attacked


def draw_ship_schedule(axis, schedule, target, start, end):
    """1つの軸に、表示範囲と重なる船の荷役区間と船名を描く。"""
    for _, row in schedule.iterrows():
        if row.completion_time <= start or row.start_time >= end:
            continue
        berth = int(row.assigned_berth[1:])
        axis.barh(berth, row.handling_time, left=row.start_time, height=0.65,
                  color=SCHEDULE_COLORS["target" if row.ship == target else "other"], edgecolor="white")
        label_start = max(start, row.start_time)
        visible_duration = min(end, row.completion_time) - label_start
        if visible_duration > 1:
            axis.text(label_start + 0.1, berth, row.ship, va="center", fontsize=7, clip_on=True)


def plot_schedule_example(results, output):
    """事前指定した条件内で、固定方式の他船待ちの増分が最大の例を描く。"""
    example, normal, attacked = select_schedule_example(results)
    target = example.target_ship
    target_arrival = attacked.loc[attacked.ship == target, "actual_arrival_time"].iloc[0]
    start, end = max(0, target_arrival - 8), target_arrival + 24
    fig, axes = plt.subplots(len(POLICIES) * 2, 1, figsize=(12, 11), sharex=True)
    tables = [table[table.policy == policy]
              for policy in POLICIES for table in (normal, attacked)]
    for axis, table in zip(axes, tables):
        draw_ship_schedule(axis, table, target, start, end)
    setup_schedule_axes(axes, target_arrival, start, end)
    fig.suptitle(f"Illustrative maximum in predefined cell: seed {int(example.seed)}, {target}, early report -4 h")
    save_figure(fig, output, "04_schedule_example")


def plot_waiting_overview(results, output):
    """全条件の平均で、正常・改ざん時の絶対値と他船への増減を並べる。"""
    runs = results["run_metrics"]
    normal = runs[runs.scenario == "normal"].drop_duplicates(
        ["demand_multiplier", "trial", "policy"]
    )
    attacks = runs[runs.scenario != "normal"]
    other = results["summary"]
    other = other[other.metric == "other_average_waiting_time"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for scenario, label in (("normal", "Normal"), ("early", "Early report"), ("late", "Late report")):
        rows = normal if scenario == "normal" else attacks[attacks.scenario == scenario]
        means = rows.groupby("policy").average_waiting_time.mean().reindex(POLICIES)
        axes[0].plot(range(len(POLICIES)), means, marker="o", label=label)
        if scenario != "normal":
            changes = other[other.scenario == scenario].groupby("policy").mean_difference.mean().reindex(POLICIES)
            axes[1].plot(range(len(POLICIES)), changes, marker="o", label=label)
    for axis in axes:
        axis.set_xticks(range(len(POLICIES)), list(POLICY_NAMES.values()))
        axis.grid(axis="y")
        axis.legend()
    axes[0].set(title="All ships: normal and tampered operations", ylabel="Average waiting time (hours)", ylim=(0, None))
    axes[1].set(title="Other ships: tampered minus same-policy normal", ylabel="Change in average waiting time (hours)")
    axes[1].axhline(0, color="0.45", linewidth=0.8)
    fig.suptitle("Overview: equal weight per input trial and experimental condition")
    save_figure(fig, output, "05_waiting_overview")
