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

from simulation import DURATION_HOURS, NUM_BERTHS, POLICIES, SHIFT_HOURS

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "data/kobe_reference_20261105"
REFERENCE_FILES = ("kobe_reference.json", "kobe_daily_forecasts.csv", "kobe_reference_rows.csv")
CODE_FILES = ("simulation.py", "main.py", "cli.py", "result_output.py", "plotting.py")
EXCEL_SHEETS = {"summary": "条件別集計", "run_metrics": "試行別結果", "ship_data": "船別データ"}
METRIC_LABELS = {
    "total_waiting_time": "全船の総待ち時間（時間）",
    "average_waiting_time": "全船の平均待ち時間（時間）",
    "max_waiting_time": "全船の最大待ち時間（時間）",
    "total_turnaround_time": "全船の総在港時間（時間）",
    "average_turnaround_time": "全船の平均在港時間（時間）",
    "completion_time": "最終荷役完了時刻（開始からの時間）",
    "target_average_waiting_time": "対象船群の平均待ち時間（時間）",
    "other_average_waiting_time": "対象外船群の平均待ち時間（時間）",
    "other_waiting_ships": "待ちが発生した対象外船数（隻）",
    "other_max_waiting_time": "対象外船群の最大待ち時間（時間）",
    "berth_changes": "計画からバースを変更した船数（隻）",
    "increased_waiting_ships": "正常時より待ちが増えた全船数（隻）",
    "other_increased_waiting_ships": "正常時より待ちが増えた対象外船数（隻）",
}
EXCEL_COLUMNS = {
    "experiment_group": "実験区分", "attack_scale": "改ざん対象の規模",
    "shift_hours": "申告変更幅（時間）", "scenario": "申告条件", "policy": "運用方式",
    "metric": "評価指標", "trials": "試行数",
    "mean": "改ざん時の指標値（試行平均）", "std": "改ざん時の標準偏差",
    "normal_mean": "正常時の指標値（試行平均）",
    "mean_difference": "正常時との差（試行平均）",
    "std_difference": "正常時との差の標準偏差",
    "min_difference": "正常時との差の最小値", "max_difference": "正常時との差の最大値",
    "increased_trials": "正常時より増加した試行数",
    "decreased_trials": "正常時より減少した試行数",
    "unchanged_trials": "正常時と同じ試行数",
    "mean_mitigation_vs_fixed": "固定方式に対する軽減量（試行平均）",
    "trial": "試行番号", "seed": "乱数シード", "selection_seed": "対象選択用シード",
    "target_count": "改ざん対象隻数", "actual_attack_fraction": "実際の改ざん対象割合",
    "target_ships": "改ざん対象船ID", "report_shifts": "各対象船の申告変更量（時間）",
    "num_ships": "全船数（隻）", **METRIC_LABELS,
    "ship": "船ID", "actual_arrival_time": "実到着時刻（時間）",
    "reported_arrival_time": "正常時の申告到着時刻（時間）",
    "base_handling_time": "基準荷役時間（時間）",
    "cargo_nearest_berth": "荷物の最寄りバース番号",
    **{f"berth_factor_B{berth}": f"B{berth}の荷役係数" for berth in range(1, NUM_BERTHS + 1)},
    **{f"handling_time_B{berth}": f"B{berth}の荷役時間（時間）" for berth in range(1, NUM_BERTHS + 1)},
}
EXCEL_VALUES = {
    "experiment_group": {"baseline": "正常比較用", "main": "主実験", "mixed_supplement": "混在の補足"},
    "attack_scale": {"single": "1隻", "p10": "全船の10%", "p30": "全船の30%", "p50": "全船の50%"},
    "scenario": {"normal": "正常", "early": "早い申告", "late": "遅い申告", "mixed": "早い・遅い申告の混在"},
    "policy": {"fixed": "固定", "same_berth": "同じバース内調整", "reassign": "バース変更"},
    "metric": METRIC_LABELS,
}
SUMMARY_OVERVIEW_COLUMNS = {
    "attack_scale": "改ざんする船", "shift_hours": "申告変更幅（時間）",
    "scenario": "申告の方向", "policy": "運用方式",
    "normal_mean": "正常時の平均待ち（時間）",
    "mean": "改ざん後の平均待ち（時間）",
    "mean_difference": "待ち時間の増減（時間）",
}


def calculate_file_hashes(directory, filenames):
    """再現確認に使うファイル名とSHA-256の対応を作る。"""
    hashes = {}
    for name in filenames:
        content = (directory / name).read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
    return hashes


def create_config(args, results, interval):
    """実行した条件と、再現に必要なコード・ライブラリの情報をまとめる。"""
    attacks = results["run_metrics"]
    normal = attacks[attacks.scenario == "normal"].drop_duplicates(
        ["trial", "policy"]
    )
    normal_waiting = normal.groupby(["policy"], as_index=False).agg(
        trials=("trial", "count"),
        mean_waiting_hours=("average_waiting_time", "mean"),
        max_waiting_hours=("max_waiting_time", "max"),
    )
    attacks = attacks[attacks.scenario != "normal"]
    return {
        # 実験の対象条件
        "model_version": "discrete_4berth_multi_ship_v6",
        "full_study": args.study,
        "trials_per_cell": args.trials,
        "first_seed": args.seed,
        "last_seed": args.seed + args.trials - 1,
        "num_berths": NUM_BERTHS,
        "num_ships": None,  # 固定隻数を指定せず、期間内の実到着をすべて生成する。
        "arrival_duration_hours": DURATION_HOURS,
        "base_mean_interarrival_hours": interval,
        "interval_source": "kobe_day_ahead_forecasts",
        "attack_scales": ["single", "p10", "p30", "p50"] if args.study else ["single"],
        "target_selection": "permutation_prefix; fractions_ceil" if args.study else "specified_V5",
        "selection_rng_seed": "trial_seed+200000; permutation then first mixed sign",
        "mixed_rule": "alternating_signs_in_selection_order; random_first_sign; no_single_ship_mixed",
        "primary_metric": "average_waiting_time; attack_minus_same_policy_normal",
        "shift_magnitudes_hours": [2.0, 4.0, 8.0] if args.study else [SHIFT_HOURS],
        "adjusted_start_rule": "arrived_and_berth_free; no advance time limit",
        "fixed_start_rule": "keep_planned_start_berth_and_order; wait_for_actual_arrival",
        "policies": list(POLICIES),
        # 計算モデルの前提
        "handling_shape": 2,
        "handling_scale": 2.4,
        "handling_model": "base_times_1_plus_0.1_times_berth_index_distance",
        "cargo_nearest_berth": "uniform_integer_from_1_to_berths",
        "attribute_rng_seed": "trial_seed+100000; per-ship gamma then cargo position",
        "normal_arrival_adjustment": "none; actual arrivals remain at generated times",
        "plan_update": "change_all_selected_reports_then_replan_once_before_operations; actual_arrivals_unchanged",
        "comparison_baseline": "same_input_trial_and_policy; attack_minus_normal",
        "initial_berth_state": "all_empty",
        "period_end": "finish_all_ships_arriving_before_period_end_including_later_completions",
        "same_time_order": "arrivals_and_completions_together; eligible_ships_in_updated_plan_order",
        # 実際に計算した件数
        "distinct_normal_plans": len(results["plan_diagnostics"]),
        "main_attack_executions": int((attacks.experiment_group == "main").sum()),
        "mixed_attack_executions": int((attacks.experiment_group == "mixed_supplement").sum()),
        "normal_metrics": "shared across shifts; repeated target-specific metrics are not independent plans",
        "normal_waiting_summary": normal_waiting.to_dict("records"),
        "normal_occupancy": results["plan_diagnostics"]["normal_occupancy"].mean(),
        "schedule_storage": "one illustrative trial; maximum fixed mean-wait increase in predefined cell",
        "example_cell": "p30_mixed_4h" if args.study else "single_early_4h",
        "normal_plan_diagnostics": {
            "last_actual_arrival_hours": results["plan_diagnostics"]["last_actual_arrival"].max(),
            "last_planned_completion_hours": results["plan_diagnostics"]["last_planned_completion"].max(),
        },
        # 再現確認に必要な版・ファイル・資料
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "openpyxl_version": openpyxl_version,
        "result_format": "results.xlsx; 条件別集計は平均待ちの一覧と13指標の詳細; 試行別結果・船別データ; 日本語表示",
        "code_sha256": calculate_file_hashes(ROOT, CODE_FILES),
        "reference_sha256": calculate_file_hashes(REFERENCE_DIR, REFERENCE_FILES),
        "sources": {
            "berth_count": "https://www.mol.co.jp/info/article/2026/0819.html",
            "rescheduling": "https://transp-or.epfl.ch/documents/technicalReports/NitBierEra2013.pdf",
            "information_manipulation": "https://doi.org/10.1088/1742-6596/2867/1/012047",
        },
    }


def save_excel(results, path):
    """集計・試行別指標・改ざん前の船別入力を日本語の1冊へ保存する。"""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name in EXCEL_SHEETS:
            table = results[name]
            if name == "summary":
                save_summary_sheet(writer, table)
                continue
            display = translate_excel_table(table)
            sheet_name = EXCEL_SHEETS[name]
            display.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet = writer.sheets[sheet_name]
            format_result_sheet(sheet, table)


def create_summary_overview(summary):
    """全船平均待ちだけを取り出し、同じ条件の3方式を並べる。"""
    overview = summary.loc[summary.metric == "average_waiting_time", list(SUMMARY_OVERVIEW_COLUMNS)].copy()
    orders = {"attack_scale": ["single", "p10", "p30", "p50"],
              "scenario": ["early", "late", "mixed"], "policy": list(POLICIES)}
    for column, values in orders.items():
        overview[column] = pd.Categorical(overview[column], categories=values, ordered=True)
    overview = overview.sort_values(["attack_scale", "shift_hours", "scenario", "policy"])
    for column in orders:
        overview[column] = overview[column].astype(object)
    return overview.reset_index(drop=True)


def save_summary_sheet(writer, summary):
    """上段は平均待ちの比較、下段は元の13指標をすべて残す。"""
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
    """読み方・表題を添え、条件の区切りと増減を見分けやすくする。"""
    notes = {
        1: "条件別集計　全船の平均待ち時間",
        2: "1行は「改ざん条件と運用方式」の結果。同じ条件の3方式を続けて表示。",
        3: "増減＝改ざん後−正常時。＋は待ちが増加、−は減少、0は正常時と同じ。",
        4: f"各条件{int(summary.trials.max())}試行。時間は船1隻あたりの平均。0でも待ち時間がないとは限りません。",
        6: "表1 全船の平均待ち時間の比較",
        detail_header - 3: "表2 全13指標の詳細集計",
        detail_header - 2: "対象船群＝改ざんした船、対象外船群＝それ以外の船。1試行の標準偏差は空欄。",
    }
    for row, text in notes.items():
        cell = sheet.cell(row, 1, text)
        cell.font = Font(name="Arial", size=14 if row == 1 else 10,
                         bold=row in (1, 6, detail_header - 3), color="000000")
        cell.alignment = Alignment(vertical="center")
        sheet.row_dimensions[row].height = 26 if row == 1 else 22
    for row, label, target in [(6, "詳細13指標へ", detail_header - 3),
                               (detail_header - 3, "一覧へ戻る", 1)]:
        link = sheet.cell(row, 7, label)
        link.hyperlink = f"#'条件別集計'!A{target}"
        link.font = Font(name="Arial", size=10, color="0563C1", underline="single")
    sheet.freeze_panes = "E8"
    sheet.sheet_view.zoomScale = 90
    for column, width in zip("ABCDEFG", [22, 17, 28, 24, 24, 24, 24]):
        sheet.column_dimensions[column].width = width
    sheet.row_dimensions[7].height = 36
    table = Table(displayName="WaitingOverview", ref=f"A7:G{7 + len(overview)}")
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=False)
    sheet.add_table(table)
    groups = overview.groupby(["attack_scale", "shift_hours", "scenario"], sort=False).ngroup()
    for index, group in enumerate(groups, 8):
        for cell in sheet[index][:7]:
            cell.fill = PatternFill("solid", fgColor="F0F4F8" if group % 2 else "FFFFFF")
        sheet.cell(index, 2).alignment = Alignment(horizontal="center", vertical="center")
        difference = sheet.cell(index, 7)
        if abs(difference.value) >= 1e-12:
            base = excel_number_format(difference.value)
            difference.number_format = f"+{base};-{base};0"
    changes = f"G8:G{7 + len(overview)}"
    for operator, limit, fill, color in [("greaterThan", 1e-9, "FCE8E6", "9C0006"),
                                         ("lessThan", -1e-9, "E2F0D9", "275A2B")]:
        sheet.conditional_formatting.add(changes, CellIsRule(
            operator=operator, formula=[str(limit)], fill=PatternFill("solid", fgColor=fill),
            font=Font(color=color)
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
        width = 64 if column in ("target_ships", "report_shifts") else 26
        sheet.column_dimensions[get_column_letter(index)].width = width
        for cells in sheet.iter_rows(min_row=header_row + 1, max_row=last_row, min_col=index, max_col=index):
            cell = cells[0]
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            if isinstance(cell.value, (int, float)):
                cell.number_format = excel_number_format(
                    cell.value, percentage=column == "actual_attack_fraction"
                )
    for cells in sheet.iter_rows(min_row=header_row + 1, max_row=last_row, max_col=len(table.columns)):
        lines = 1
        for cell in cells:
            if isinstance(cell.value, str):
                text_width = sum(2 if ord(char) > 255 else 1 for char in cell.value)
                width = sheet.column_dimensions[cell.column_letter].width - 2
                lines = max(lines, int(np.ceil(text_width / width)))
        sheet.row_dimensions[cells[0].row].height = max(22, lines * 15)


def save_results(results, config, output):
    """集計・試行別結果・船別入力の3表と実験条件を保存する。"""
    output.mkdir(parents=True, exist_ok=True)
    save_excel(results, output / "results.xlsx")
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def print_summary(results, interval, output):
    """実行件数と方向別に利用可能な条件の平均を表示する。outputがNoneなら保存なし。"""
    runs = results["run_metrics"]
    normal = runs[runs.scenario == "normal"]
    attacks = runs[runs.scenario != "normal"]
    print(
        f"正常計画 {len(results['plan_diagnostics'])}組、"
        f"3方式の改ざん運用 {len(attacks)}ケース"
    )
    print(
        f"正常時の最大待ち時間：{normal.max_waiting_time.max():.9f}時間。"
        f"実到着の基準平均間隔：{interval:.6f}時間。"
    )
    print("下表は方向ごとに利用可能な条件を同じ重みでまとめた概要です。条件別の比較はresults.xlsxの「条件別集計」シートを参照してください。")
    if (attacks.scenario == "mixed").any():
        print("早い・遅い申告には1隻条件を含み、混在は10%・30%・50%のみです。")
    if results["run_metrics"]["trial"].nunique() == 1:
        print("各条件1試行の結果です。異なる乱数での平均・ばらつきは評価していません。")
    summary = results["summary"]
    all_wait = summary[summary.metric == "average_waiting_time"]
    print("全船の平均待ち時間（時間）。mean_differenceは同じ方式の正常時からの増減です。")
    print(all_wait.groupby(["policy", "scenario"])[
        ["normal_mean", "mean", "mean_difference"]
    ].mean().round(4).to_string())
    print("実到着は固定し、正常時にも待ちを認めます。実測の荷役時間・実待ち時間による検証ではありません。")
    print("保存なし" if output is None else f"保存先：{output.resolve()}")
