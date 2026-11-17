"""主実験・補助実験の比較図と、試行1の実行過程をPNG/PDFへ保存する。"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from simulation import METHOD_LABELS

COLORS = {"normal": "#777777", "wait": "#b54a45", "fifo": "#2675ac", "spt": "#df8a17"}


def configure_fonts():
    names = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic", "Yu Gothic"):
        if name in names:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False, "pdf.fonttype": 42})


def save_figure(fig, folder, filename, caption):
    folder.mkdir(parents=True, exist_ok=True)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{filename}.{extension}", dpi=160)
    plt.close(fig)


def format_axis(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)


def draw_duration_comparison(summary, folder):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    durations = [6, 12, 24, 48]
    for ax, load in zip(axes, sorted(summary.load.unique())):
        for method in METHOD_LABELS:
            group = summary[(summary.load == load) & (summary.method == method)].sort_values("duration")
            if method == "normal":
                values = np.repeat(group.iloc[0].mean_wait, len(durations))
                deviations = np.repeat(group.iloc[0].mean_wait_sd, len(durations))
            else:
                values, deviations = group.mean_wait, group.mean_wait_sd
            ax.errorbar(durations, values, yerr=np.nan_to_num(deviations), marker="o",
                        capsize=3, label=METHOD_LABELS[method], color=COLORS[method])
        ax.set_title(f"到着頻度 {load:g}倍")
        ax.set_xticks(durations)
        ax.set_ylim(bottom=0)
        format_axis(ax, "停止時間 D（時間）", "全船平均待ち（時間）")
    axes[0].legend(fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.2, wspace=0.24)
    save_figure(fig, folder, "図1_停止時間と全船平均待ち",
                "図1 停止時間と全船平均待ち（準備2時間・手動上限2隻）\n点は試行平均、誤差棒は試行間標本SD。停止なしは同じ値を各停止時間に表示。")


def draw_paired_differences(summary, folder):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, load in zip(axes, sorted(summary.load.unique())):
        group = summary[(summary.load == load) & (summary.method == "spt")].sort_values("duration")
        for metric, label, color in (
                ("fifo_delta_wait", "全船", "#2675ac"),
                ("fifo_delta_long_wait", "長時間船（入力上位25％）", "#b54a45")):
            ax.errorbar(group.duration, group[metric], yerr=np.nan_to_num(group[f"{metric}_sd"]),
                        marker="o", capsize=3, label=label, color=color)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"到着頻度 {load:g}倍")
        ax.set_xticks([6, 12, 24, 48])
        format_axis(ax, "停止時間 D（時間）", "短い荷役優先 − 手動FIFO（時間）")
    axes[0].legend(fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.2, wspace=0.28)
    save_figure(fig, folder, "図2_手動方式の平均待ちの差",
                "図2 同じ入力・同じ能力での手動方式の差（準備2時間・上限2隻）\n点は試行ごとの対応差の平均、誤差棒は対応差の標本SD。負は短い荷役優先の待ちが小さい。")


def draw_example(results, folder, outage_start):
    schedules, events = results["schedules"], results["events"]
    chosen = schedules[(schedules.trial == 1) & (schedules.load == schedules.load.max())]
    chosen = chosen[(chosen.method == "normal") | ((chosen.duration == 24)
                    & ((chosen.method == "wait") | ((chosen.preparation == 2) & (chosen.capacity == 2))))]
    example_events = events[(events.trial == 1) & (events.load == chosen.load.max())
                            & events.condition.isin(chosen.condition)]
    waiting_limit = example_events.observed_waiting.max() + 1
    fig, axes = plt.subplots(4, 2, figsize=(15, 11))
    for row, method in enumerate(METHOD_LABELS):
        group = chosen[chosen.method == method]
        condition = group.iloc[0].condition
        audit = events[(events.trial == 1) & (events.load == chosen.load.max())
                       & (events.condition == condition)]
        queue_ax, berth_ax = axes[row]
        queue_ax.step(audit.time, audit.observed_waiting, where="post", color=COLORS[method])
        for ship in group.itertuples():
            color = COLORS[method] if ship.phase == "手動" else "#73869a"
            berth_ax.broken_barh([(ship.start, ship.handling)], (ship.berth - 0.35, 0.7),
                                facecolors=color, edgecolors="white", linewidth=0.4)
        for ax in (queue_ax, berth_ax):
            ax.axvspan(outage_start, outage_start + 24, color="#b54a45", alpha=0.1)
            ax.axvline(outage_start + 2, color="#777777", linestyle=":", linewidth=0.8)
            ax.set_xlim(0, max(168, chosen.completion.max()))
        queue_ax.set_title(METHOD_LABELS[method])
        berth_ax.set_title(METHOD_LABELS[method])
        queue_ax.set_ylim(0, waiting_limit)
        berth_ax.set_yticks([1, 2, 3, 4], ["B1", "B2", "B3", "B4"])
        format_axis(queue_ax, "時刻（時間）", "開始処理後の待機船（隻）")
        format_axis(berth_ax, "時刻（時間）", "荷役中のバース")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.11, hspace=0.7, wspace=0.18)
    save_figure(fig, folder, "図3_代表試行の待機と荷役スケジュール",
                "図3 試行1（seed42）・到着頻度2倍・停止24時間の実行過程\n淡赤は停止48〜72時間、点線は準備完了50時間。右の色付き区間は手動中に開始した荷役。")


def draw_auxiliary_comparison(summary, folder, variable, number, filename):
    """準備又は能力を一つだけ変え、全船・長時間船の待ちを並べる。"""
    if variable == "preparation":
        selected = summary[(summary.capacity == 2) | summary.method.isin(["normal", "wait"])]
        xlabel, fixed = "準備時間 s（時間）", "手動上限2隻"
        values = [0, 2, 6]
    else:
        selected = summary[(summary.preparation == 2) | summary.method.isin(["normal", "wait"])]
        xlabel, fixed = "手動同時上限 K（隻）", "準備2時間"
        values = [1, 2, 3]
    metrics = [("mean_wait", "全船平均待ち"), ("long_mean_wait", "長時間船平均待ち")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, load in enumerate(sorted(selected.load.unique())):
        for column, (metric, label) in enumerate(metrics):
            ax = axes[row, column]
            for method in ("fifo", "spt"):
                group = selected[(selected.load == load) & (selected.method == method)].sort_values(variable)
                ax.errorbar(group[variable], group[metric], yerr=np.nan_to_num(group[f"{metric}_sd"]),
                            marker="o", capsize=3, color=COLORS[method], label=METHOD_LABELS[method])
            for method, style in (("normal", ":"), ("wait", "--")):
                reference = selected[(selected.load == load) & (selected.method == method)].iloc[0]
                ax.axhline(reference[metric], color=COLORS[method], linestyle=style,
                           label=METHOD_LABELS[method])
            ax.set_title(f"到着頻度{load:g}倍・{label}")
            ax.set_xticks(values)
            ax.set_ylim(bottom=0)
            format_axis(ax, xlabel, label + "（時間）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.14, hspace=0.4, wspace=0.23)
    save_figure(fig, folder, filename,
                f"図{number} {xlabel.split(' ')[0]}の影響（停止24時間・{fixed}）\n点は試行平均、誤差棒は試行間標本SD。水平線は停止なし・復旧待ちの平均。")


def draw_auxiliary_differences(summary, folder):
    """5組の対応差を、準備の確認・能力の確認に分けて示す。"""
    selected = summary[summary.method == "spt"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, load in enumerate(sorted(selected.load.unique())):
        for column, variable in enumerate(("preparation", "capacity")):
            ax = axes[row, column]
            if variable == "preparation":
                group = selected[(selected.load == load) & (selected.capacity == 2)]
                xlabel, fixed, values = "準備時間 s（時間）", "上限2隻固定", [0, 2, 6]
            else:
                group = selected[(selected.load == load) & (selected.preparation == 2)]
                xlabel, fixed, values = "手動同時上限 K（隻）", "準備2時間固定", [1, 2, 3]
            group = group.sort_values(variable)
            for metric, label, color in (
                    ("fifo_delta_wait", "全船", "#2675ac"),
                    ("fifo_delta_long_wait", "長時間船（入力上位25％）", "#b54a45")):
                ax.errorbar(group[variable], group[metric], yerr=np.nan_to_num(group[f"{metric}_sd"]),
                            marker="o", capsize=3, color=color, label=label)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"到着頻度{load:g}倍・{fixed}")
            ax.set_xticks(values)
            format_axis(ax, xlabel, "短い荷役優先 − 手動FIFO（時間）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.14, hspace=0.4, wspace=0.25)
    save_figure(fig, folder, "図6_補助実験の手動方式の差",
                "図6 準備・能力を一つずつ変えた手動方式の対応差（停止24時間）\n点は対応差の試行平均、誤差棒は対応差の標本SD。負は短い荷役優先の待ちが小さい。")


def plot_results(results, output, settings):
    configure_fonts()
    summary = results["summary"]
    if settings["experiment"] != "補助実験":
        main = summary[summary.include_main]
        folder = Path(output) / "主実験"
        draw_duration_comparison(main, folder)
        draw_paired_differences(main, folder)
        draw_example(results, folder, settings["outage_start"])
    if settings["experiment"] != "主実験":
        auxiliary = summary[summary.include_auxiliary]
        folder = Path(output) / "補助実験"
        draw_auxiliary_comparison(auxiliary, folder, "preparation", 4, "図4_準備時間と待ち時間")
        draw_auxiliary_comparison(auxiliary, folder, "capacity", 5, "図5_手動同時上限と待ち時間")
        draw_auxiliary_differences(auxiliary, folder)
