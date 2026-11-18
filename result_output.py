"""実験条件の記録、計算結果のExcel・JSON保存、画面への概要表示を担当する。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import __version__ as openpyxl_version
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from simulation import DURATION_HOURS, NUM_BERTHS, REPORT_SHIFT_HOURS, TARGET_POSITION, WAIT_REASONS

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "data/kobe_reference_20261105"
REFERENCE_FILES = ("kobe_reference.json", "kobe_daily_forecasts.csv", "kobe_reference_rows.csv")
CODE_FILES = ("simulation.py", "main.py", "cli.py", "result_output.py", "plotting.py")
EXCEL_SHEETS = {"summary": "条件別集計", "run_metrics": "試行別結果", "ship_data": "船別データ",
                "waiting_details": "待ち理由"}
WAIT_REASON_LABELS = {
    "all_berths_busy_wait": "全バース使用中の待ち（時間）",
    "assigned_berth_wait": "指定バース使用中・別バース空きの待ち（時間）",
    "planned_order_wait": "計画順番待ち（時間）",
    "planned_start_wait": "計画開始待ち（時間）",
}
METRIC_LABELS = {
    "total_waiting_time": "全船の総待ち時間（時間）",
    "average_waiting_time": "全船の平均待ち時間（時間）",
    "target_waiting_time": "対象船の待ち時間（時間）",
    "max_waiting_time": "全船の最大待ち時間（時間）",
    "total_turnaround_time": "全船の総在港時間（時間）",
    "average_turnaround_time": "全船の平均在港時間（時間）",
    "completion_time": "最終荷役完了時刻（開始からの時間）",
    "berth_changes": "計画からバースを変更した船数（隻）",
    "increased_waiting_ships": "ずれなし条件より待ちが増えた全船数（隻）",
}
EXCEL_COLUMNS = {
    "scenario": "予定情報の条件", "policy": "運用方式", "metric": "評価指標", "trials": "試行数",
    "mean": "予定ずれあり条件の指標値（試行平均）", "std": "予定ずれあり条件の標準偏差",
    "normal_mean": "ずれなし条件の指標値（試行平均）", "mean_difference": "ずれなし条件との差（試行平均）",
    "std_difference": "ずれなし条件との差の標準偏差",
    "min_difference": "ずれなし条件との差の最小値", "max_difference": "ずれなし条件との差の最大値",
    "increased_trials": "ずれなし条件より増加した試行数", "decreased_trials": "ずれなし条件より減少した試行数",
    "unchanged_trials": "ずれなし条件と同じ試行数",
    "trial": "試行番号", "seed": "乱数シード",
    "demand_multiplier": "到着頻度倍率", "shift_hours": "予定の遅れ（時間）",
    "target_ship": "対象船ID", "target_rank": "対象船の実到着順", "is_target": "予定をずらす船",
    "num_ships": "全船数（隻）", **METRIC_LABELS,
    "ship": "船ID", "actual_arrival_time": "実到着時刻（時間）",
    "reported_arrival_time": "ずれなし条件の予定到着時刻（時間）",
    "planned_berth": "計画バース", "assigned_berth": "実行バース",
    "planned_start_time": "計画荷役開始時刻（時間）", "start_time": "実荷役開始時刻（時間）",
    "waiting_time": "待ち時間（時間）",
    **WAIT_REASON_LABELS,
    **{f"average_{name}": label.replace("（時間）", "（時間／隻）")
       for name, label in WAIT_REASON_LABELS.items()},
    "competition_events": "開始枠の競合回数（回）",
    "competition_participations": "開始枠の競合参加回数（回）",
    "base_handling_time": "基準荷役時間（時間）", "cargo_nearest_berth": "荷物の最寄りバース番号",
    **{f"berth_factor_B{berth}": f"B{berth}の荷役係数" for berth in range(1, NUM_BERTHS + 1)},
    **{f"handling_time_B{berth}": f"B{berth}の荷役時間（時間）" for berth in range(1, NUM_BERTHS + 1)},
}
EXCEL_VALUES = {
    "scenario": {"normal": "ずれなし", "attack": "予定ずれあり"},
    "policy": {"fixed": "固定", "same_berth": "同じバース内調整", "reassign": "バース変更"},
    "metric": METRIC_LABELS,
    "is_target": {True: "対象", False: "対象外"},
}
SUMMARY_OVERVIEW_COLUMNS = {
    "demand_multiplier": EXCEL_COLUMNS["demand_multiplier"], "shift_hours": EXCEL_COLUMNS["shift_hours"],
    "policy": "運用方式", "normal_mean": "ずれなし条件の平均待ち（時間）",
    "mean": "予定ずれありの平均待ち（時間）", "mean_difference": "待ち時間の増減（時間）",
}


def calculate_file_hashes(directory, filenames):
    """再現確認に使うファイル名とSHA-256の対応を作る。"""
    hashes = {}
    for name in filenames:
        content = (directory / name).read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
    return hashes


def create_config(args, results, interval):
    """実行条件・乱数規則・モデルの仮定と、再現に必要な版を記録する。"""
    runs = results["run_metrics"]
    normal = runs[runs.scenario == "normal"]
    normal_waiting = normal.groupby(["demand_multiplier", "policy"], as_index=False).agg(
        trials=("trial", "count"), mean_waiting_hours=("average_waiting_time", "mean"),
        max_waiting_hours=("max_waiting_time", "max"),
    )
    return {
        "study_scope": "arrival schedule deviation; cyber tampering is one possible cause; causes are not simulated separately",
        "model_version": "discrete_4berth_one_ship_late_v9",
        "trials_per_cell": args.trials, "first_seed": args.seed,
        "last_seed": args.seed + args.trials - 1,
        "num_berths": NUM_BERTHS, "arrival_duration_hours": DURATION_HOURS,
        "base_mean_interarrival_hours": interval, "interval_source": "kobe_day_ahead_forecasts",
        "demand_multipliers": sorted(runs.demand_multiplier.unique().tolist()),
        "demand_rule": "mean_interarrival_time = base_mean_interarrival_hours / demand_multiplier",
        "target_ship_count": 1,
        "target_position": TARGET_POSITION,
        "target_selection": "ceil(ship_count * target_position), one-based actual arrival order; ties follow input order",
        "report_shift_hours": list(REPORT_SHIFT_HOURS),
        "shift_rule": "target ETA = actual arrival + shift_hours; other ETAs = actual arrivals",
        "primary_metric": "average_waiting_time; shifted_minus_matched within trial, demand and policy",
        "policies": runs.policy.unique().tolist(),
        "adjusted_start_rule": "arrived_and_berth_free; no advance time limit",
        "fixed_start_rule": "keep_planned_start_berth_and_order; wait_for_actual_arrival",
        "handling_shape": 2, "handling_scale": 2.4,
        "handling_model": "base_times_1_plus_0.1_times_berth_index_distance",
        "cargo_nearest_berth": "uniform_integer_from_1_to_berths",
        "attribute_rng_seed": "trial_seed+100000; per-ship gamma then cargo position",
        "normal_arrival_adjustment": "none; actual arrivals remain at generated times",
        "plan_update": "change_one_report_then_replan_once_before_operations; actual_arrivals_unchanged",
        "comparison_baseline": "same_input_trial_demand_and_policy; shifted_minus_matched",
        "initial_berth_state": "all_empty",
        "period_end": "finish_all_ships_arriving_before_period_end_including_later_completions",
        "same_time_order": "arrivals_and_completions_together; eligible_ships_in_updated_plan_order",
        "distinct_normal_plans": len(results["plan_diagnostics"]),
        "shifted_executions": int((runs.scenario == "attack").sum()),
        "normal_executions": len(normal),
        "normal_waiting_summary": normal_waiting.to_dict("records"),
        "normal_occupancy": results["plan_diagnostics"]["normal_occupancy"].mean(),
        "schedule_storage": "illustrative trial with largest mean-wait increase for the first policy at the baseline demand and +8 hours",
        "example_demand_multiplier": float(results["schedules"].demand_multiplier.iloc[0]),
        "example_shift_hours": int(results["schedules"].shift_hours.max()),
        "example_seed": int(results["schedules"].seed.iloc[0]),
        "normal_plan_diagnostics": {
            "last_actual_arrival_hours": results["plan_diagnostics"]["last_actual_arrival"].max(),
            "last_planned_completion_hours": results["plan_diagnostics"]["last_planned_completion"].max(),
        },
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "openpyxl_version": openpyxl_version,
        "result_format": "results.xlsx; 条件別集計・試行別結果・船別データ・待ち理由; 日本語表示",
        "waiting_diagnostics": {
            "classification_priority": list(WAIT_REASONS),
            "classification": "exclusive states of observed waiting intervals; not a causal decomposition",
            "competition_rule": "reassign only; at each unique arrival/completion before starts: waiting_ships > free_berths > 0",
            "participations": "each candidate ship counts once per competition event; not additive to event count",
            "detail_rows": len(results["waiting_details"]),
        },
        "excel_number_format": {
            "significant_digits_for_fractional_values": 3,
            "integer_values": "unchanged; no decimals", "trailing_zeros": "omitted",
            "near_zero_display_threshold": 1e-12,
        },
        "figure_caption_position": "bottom",
        "summary_layout": {"overview_rows": len(results["summary"]) // len(METRIC_LABELS),
                           "overview_header_row": 7,
                           "detail_header_row": len(results["summary"]) // len(METRIC_LABELS) + 14,
                           "detail_rows": len(results["summary"])},
        "code_sha256": calculate_file_hashes(ROOT, CODE_FILES),
        "reference_sha256": calculate_file_hashes(REFERENCE_DIR, REFERENCE_FILES),
        "sources": {
            "berth_count": "https://www.mol.co.jp/info/article/2026/0819.html",
            "rescheduling": "https://transp-or.epfl.ch/documents/technicalReports/NitBierEra2013.pdf",
            "information_manipulation": "https://doi.org/10.1088/1742-6596/2867/1/012047",
        },
    }


def save_excel(results, path):
    """集計・試行別指標・船別入力・待ち理由を日本語の1冊へ保存する。"""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name in EXCEL_SHEETS:
            table = results[name]
            if name == "summary":
                save_summary_sheet(writer, table)
                continue
            if name == "waiting_details":
                save_waiting_reason_sheet(writer, results)
                continue
            display = translate_excel_table(table)
            sheet_name = EXCEL_SHEETS[name]
            display.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet = writer.sheets[sheet_name]
            format_result_sheet(sheet, table)


def create_waiting_overview(run_metrics):
    """各試行の船平均を平均し、競合回数は1週あたりで集計する。"""
    columns = ["average_waiting_time", *[f"average_{name}" for name in WAIT_REASONS], "competition_events"]
    keys = ["demand_multiplier", "shift_hours", "policy", "scenario"]
    return run_metrics.groupby(keys, sort=False)[columns].mean().reset_index()


def save_waiting_reason_sheet(writer, results):
    """1枚の上段に方式別一覧、下段に全試行の船別明細を置く。"""
    name = EXCEL_SHEETS["waiting_details"]
    overview = create_waiting_overview(results["run_metrics"])
    details = results["waiting_details"]
    identifiers = ["trial", "demand_multiplier", "shift_hours", "policy", "ship"]
    details = details[identifiers + [column for column in details if column not in identifiers]]
    labels = {**EXCEL_COLUMNS, "average_waiting_time": "平均待ち時間（時間／隻）",
              "reported_arrival_time": "その条件の申告到着時刻（時間）",
              "completion_time": "荷役完了時刻（時間）", "competition_events": "開始枠の競合（回／週）"}
    translate_excel_table(overview, labels).to_excel(writer, sheet_name=name, startrow=6, index=False)
    detail_header = len(overview) + 12
    translate_excel_table(details, labels).to_excel(writer, sheet_name=name, startrow=detail_header - 1, index=False)
    sheet = writer.sheets[name]
    format_result_sheet(sheet, overview, header_row=7)
    format_result_sheet(sheet, details, header_row=detail_header)
    trials = results["run_metrics"].trial.nunique()
    trial_label = "1試行の結果" if trials == 1 else f"{trials}試行の平均"
    notes = {
        1: "待ち時間の内訳と開始枠の競合",
        2: f"上段は{trial_label}。4種類の時間の合計が平均待ち時間と一致します。",
        3: "分類の優先順：全バース使用中、指定バース使用中、計画順番待ち、計画開始待ち。重複して数えません。",
        4: "競合＝到着・完了時点で、開始前の到着済み待ち隻数＞空きバース数＞0。バース変更のみ集計。",
        5: "他2方式の競合欄は対象外で空欄。時間の内訳はその時間帯の状態で、原因の寄与率ではありません。",
        6: "表1 ずれなし・予定ずれありの待ち時間の内訳",
        detail_header - 3: "計画順番待ち＝指定先は空きだが先行船が未完了。計画開始待ち＝先行船も完了し、計画開始前。",
        detail_header - 2: "表2 船ごとの待ち時間の内訳",
        detail_header - 1: "1行は1試行・到着頻度・予定の遅れ・方式・船の結果。競合参加回数は同じ競合に参加した各船へ数えます。",
    }
    for row, value in notes.items():
        cell = sheet.cell(row, 1, value)
        cell.font = Font(name="Arial", size=14 if row == 1 else 10,
                         bold=row in (1, 6, detail_header - 2))
        cell.alignment = Alignment(vertical="center")
        sheet.row_dimensions[row].height = 26 if row == 1 else 22
    for row, text, destination in [(6, "船別明細へ", detail_header - 2),
                                    (detail_header - 2, "一覧へ戻る", 1)]:
        cell = sheet.cell(row, 8, text)
        cell.hyperlink = f"#'{name}'!A{destination}"
        cell.font = Font(name="Arial", size=10, color="0563C1", underline="single")
    for table_name, header, table in [("WaitingReasonsOverview", 7, overview),
                                      ("WaitingReasonsDetails", detail_header, details)]:
        native = Table(displayName=table_name,
                       ref=f"A{header}:{get_column_letter(len(table.columns))}{header + len(table)}")
        native.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
        sheet.add_table(native)
    sheet.freeze_panes = "E2"
    sheet.sheet_view.zoomScale = 85


def create_summary_overview(summary):
    """各到着頻度・変更幅・方式の平均待ちと対応差を一覧にする。"""
    overview = summary.loc[summary.metric == "average_waiting_time", list(SUMMARY_OVERVIEW_COLUMNS)]
    return overview.reset_index(drop=True)


def save_summary_sheet(writer, summary):
    """上段は平均待ちの比較、下段は全指標の詳細を残す。"""
    overview = create_summary_overview(summary)
    name = EXCEL_SHEETS["summary"]
    translate_excel_table(overview, SUMMARY_OVERVIEW_COLUMNS).to_excel(
        writer, sheet_name=name, startrow=6, index=False
    )
    detail_header = len(overview) + 14
    translate_excel_table(summary).to_excel(writer, sheet_name=name, startrow=detail_header - 1, index=False)
    sheet = writer.sheets[name]
    format_result_sheet(sheet, overview, header_row=7)
    format_result_sheet(sheet, summary, header_row=detail_header)
    format_summary_sheet(sheet, overview, summary, detail_header)


def format_summary_sheet(sheet, overview, summary, detail_header):
    """条件と読み方を表の上へ置き、待ちの増減を色で見分けやすくする。"""
    trials = int(summary.trials.max())
    trial_label = "1試行の結果" if trials == 1 else f"{trials}試行の平均"
    deviation_note = "1試行のため標準偏差は空欄です。" if trials == 1 else "標準偏差は試行間のばらつきです。"
    notes = {
        1: "1隻の予定ずれと船の待ち時間",
        2: "実到着順の10%地点の1隻を対象に、予定到着を2時間または8時間遅らせた比較。",
        3: "増減＝予定ずれあり−同じ方式のずれなし条件。＋は待ちが増加、−は減少。",
        4: f"{trial_label}。時間は船1隻あたり。実到着・荷役条件はずれなし条件と同じ。",
        6: "表1 全船の平均待ち時間の比較",
        detail_header - 3: "表2 全指標の詳細集計",
        detail_header - 2: f"{deviation_note} 差は同じ試行・到着頻度・方式のずれなしから算出。",
    }
    for row, text in notes.items():
        cell = sheet.cell(row, 1, text)
        cell.font = Font(name="Arial", size=14 if row == 1 else 10,
                         bold=row in (1, 6, detail_header - 3), color="000000")
        cell.alignment = Alignment(vertical="center")
        sheet.row_dimensions[row].height = 26 if row == 1 else 22
    for row, label, target in [(6, "詳細指標へ", detail_header - 3),
                               (detail_header - 3, "一覧へ戻る", 1)]:
        link = sheet.cell(row, 4, label)
        link.hyperlink = f"#'条件別集計'!A{target}"
        link.font = Font(name="Arial", size=10, color="0563C1", underline="single")
    sheet.freeze_panes = "B8"
    sheet.sheet_view.zoomScale = 90
    for column in "ABCDEF":
        sheet.column_dimensions[column].width = 30
    sheet.row_dimensions[7].height = 36
    table = Table(displayName="WaitingOverview", ref=f"A7:F{7 + len(overview)}")
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
    sheet.add_table(table)
    for index in range(8, 8 + len(overview)):
        difference = sheet.cell(index, 6)
        if abs(difference.value) >= 1e-12:
            base = excel_number_format(difference.value)
            difference.number_format = f"+{base};-{base};0"
    changes = f"F8:F{7 + len(overview)}"
    for operator, limit, fill, color in [("greaterThan", 1e-9, "FCE8E6", "9C0006"),
                                         ("lessThan", -1e-9, "E2F0D9", "275A2B")]:
        sheet.conditional_formatting.add(changes, CellIsRule(
            operator=operator, formula=[str(limit)], fill=PatternFill("solid", fgColor=fill),
            font=Font(color=color),
        ))


def translate_excel_table(table, columns=EXCEL_COLUMNS):
    """計算に使う英語の識別子は保ち、Excelへ渡す表だけ日本語にする。"""
    display = table.copy(deep=True)
    for column, labels in EXCEL_VALUES.items():
        if column in display:
            display[column] = display[column].replace(labels)
    return display.rename(columns=columns)


def excel_number_format(value, percentage=False):
    """値は変えず、小数を有効数字3桁以内で表示する書式を返す。"""
    display_value = float(value) * (100 if percentage else 1)
    suffix = "%" if percentage else ""
    # 計算誤差程度の残差は、これまでと同じく0と表示する。
    if display_value.is_integer() or abs(display_value) < 1e-12:
        return "0" + suffix
    rounded = float(f"{display_value:.3g}")
    if abs(rounded) >= 1000:
        return "0.##E+00" + suffix
    decimal_places = max(0, 2 - int(np.floor(np.log10(abs(rounded)))))
    decimals = "." + "#" * decimal_places if decimal_places else ""
    return "0" + decimals + suffix


def format_result_sheet(sheet, table, header_row=1):
    """保存した表の見出し、列幅、数値表示を整える。"""
    sheet.freeze_panes = "B2"
    last_row = header_row + len(table)
    sheet.auto_filter.ref = f"A{header_row}:{get_column_letter(len(table.columns))}{last_row}"
    sheet.sheet_view.showGridLines = False
    sheet.row_dimensions[header_row].height = 60
    for index, column in enumerate(table.columns, 1):
        header = sheet.cell(header_row, index)
        header.fill = PatternFill("solid", fgColor="243B53")
        header.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        header.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        width = 26
        sheet.column_dimensions[get_column_letter(index)].width = width
        for cells in sheet.iter_rows(min_row=header_row + 1, max_row=last_row, min_col=index, max_col=index):
            cell = cells[0]
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            if isinstance(cell.value, (int, float)):
                cell.number_format = excel_number_format(cell.value)
    for cells in sheet.iter_rows(min_row=header_row + 1, max_row=last_row, max_col=len(table.columns)):
        lines = 1
        for cell in cells:
            if isinstance(cell.value, str):
                text_width = sum(2 if ord(char) > 255 else 1 for char in cell.value)
                width = sheet.column_dimensions[cell.column_letter].width - 2
                lines = max(lines, int(np.ceil(text_width / width)))
        sheet.row_dimensions[cells[0].row].height = max(22, lines * 15)


def save_results(results, config, output):
    """集計・試行別結果・船別入力・待ち理由の4表と実験条件を保存する。"""
    output.mkdir(parents=True, exist_ok=True)
    save_excel(results, output / "results.xlsx")
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def print_summary(results, interval, output):
    """実行した条件と平均待ちを表示する。"""
    runs = results["run_metrics"]
    print(f"{runs.trial.nunique()}試行、ずれなし運用{int((runs.scenario == 'normal').sum())}ケース、"
          f"1隻の予定ずれ運用{int((runs.scenario == 'attack').sum())}ケース。")
    print("実到着順の10%地点の1隻だけ、予定到着を2時間または8時間遅らせます。")
    print(f"実到着の基準平均間隔：{interval:.6f}時間。実到着・荷役条件はずれなし条件と共通です。")
    print(translate_excel_table(create_summary_overview(results["summary"]), SUMMARY_OVERVIEW_COLUMNS)
          .round(4).to_string(index=False))
    print("保存なし" if output is None else f"保存先：{output.resolve()}")
