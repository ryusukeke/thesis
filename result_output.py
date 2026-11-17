"""実験条件・日本語Excelの保存と結果概要の表示。"""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from simulation import CONDITION_COLUMNS, METHOD_LABELS, METRICS

ROOT = Path(__file__).resolve().parent
LABELS = {
    "trial": "試行", "seed": "試行シード", "load": "到着頻度倍率",
    "condition": "条件ID", "method": "運用方式", "duration": "停止時間（時間）",
    "preparation": "準備時間（時間）", "capacity": "手動同時上限（隻）",
    "trials": "試行数", "ships": "船数（隻）",
    "mean_wait": "全船平均待ち（時間）", "max_wait": "最大待ち（時間）",
    "long_mean_wait": "長時間船平均待ち（時間）",
    "delta_wait": "停止なしとの差・全船平均待ち（時間）",
    "fifo_delta_wait": "手動FIFOとの差・全船平均待ち（時間）",
    "fifo_delta_long_wait": "手動FIFOとの差・長時間船平均待ち（時間）",
    "fifo_worse_ships": "手動FIFOより待ちが増えた船（隻）",
    "cohort_ships": "停止中到着群の船数（隻）",
    "cohort_mean_wait": "停止中到着群の平均待ち（時間）",
    "cohort_normal_mean_wait": "同じ到着群の停止なし平均待ち（時間）",
    "cohort_clearance": "停止中到着群の全船開始までの復旧後時間（時間）",
    "manual_starts": "手動中の新規開始（隻）",
    "manual_berth_hours": "手動時間帯の延べバース稼働（隻・時間）",
    "normal_increases": "停止なし比・平均待ち増加試行数",
    "normal_decreases": "停止なし比・平均待ち減少試行数",
    "normal_unchanged": "停止なし比・平均待ち不変試行数",
    "fifo_increases": "手動FIFO比・平均待ち増加試行数",
    "fifo_decreases": "手動FIFO比・平均待ち減少試行数",
    "fifo_unchanged": "手動FIFO比・平均待ち不変試行数",
    "ship": "船ID", "arrival": "到着（時間）", "handling": "荷役時間（時間）",
    "long_handling": "長時間船（入力上位25％）", "berth": "バース番号",
    "start": "荷役開始（時間）", "completion": "荷役完了（時間）",
    "waiting": "待ち（時間）", "turnaround": "在港時間（時間）",
    "phase": "開始時または判断時の状態", "time": "判断イベント時刻（時間）",
    "start_limit": "新規開始判断の同時上限（隻）", "busy_before": "開始処理前の荷役中（隻）",
    "starts": "同時刻の新規開始（隻）", "busy_after": "開始処理後の荷役中（隻）",
    "observed_waiting": "開始処理後の待機船（隻）",
}
for metric in METRICS:
    LABELS[f"{metric}_sd"] = LABELS[metric] + "・試行間標本SD"


def make_config(settings, results):
    """全条件と計算コード・参照JSONのハッシュを保存する。"""
    filenames = ["simulation.py", "main.py", "result_output.py", "plotting.py",
                 "data/kobe_reference_20261105/kobe_reference.json"]
    runs = results["runs"]
    conditions = runs[["load", *CONDITION_COLUMNS]].drop_duplicates()
    return {
        "モデル": "TOS停止時の限定的手動運用・離散4バース",
        "計算設定": settings,
        "条件数": len(conditions), "実行運用数": len(runs),
        "条件一覧": translated_table(conditions).to_dict("records"),
        "入力": "到着間隔は指数分布、到着は0以上168時間未満。全船を完了まで処理",
        "荷役": "Gamma(2,2.4)、平均4.8時間。全バース・全運用で同じ。実測値ではない",
        "乱数系列": {"到着": "seed", "荷役": "seed+100000"},
        "通常と復旧後": "到着順FIFO、4バース。最適解とは扱わない",
        "停止中": "新規開始を見合わせる。準備後の手動は総荷役中隻数がK未満なら開始可能",
        "開始済み荷役": "停止・準備・復旧の境界でも中断・移動しない",
        "手動選船": "到着済みだけを対象にFIFO又は荷役時間順。同順位は到着・船ID順",
        "情報の仮定": "到着と完了を即時観測。到着済みの荷役見積りが既知で実時間と等しい",
        "処理順": "同時刻の完了・到着を観測→状態判定→新規開始。手動区間[u+s,u+D)",
        "復旧": "通常業務へ戻れる時刻を指定。記録の再入力・照合による追加遅延は扱わない",
        "長時間船": "入力荷役の上位25％、端数切上げ、同時間は船ID順。運用間で固定",
        "停止中到着群": "a∈[u,u+D)で固定。全船開始までの復旧後時間はmax(0,最終開始-u-D)、空群0",
        "手動稼働指標": "手動時間帯と全船荷役区間の重なりの総和。人員・人件費の計測ではない",
        "集計": "試行を等重みで平均。標本SDはddof=1。1試行ならSDは空欄。対応差は同じ船入力",
        "比較": "手動方式同士が中心。停止なしもFIFOのため、そこからの差は純粋な停止損害とは限らない",
        "基準の再利用": "停止なしは試行・頻度ごとに1回だけ計算する",
        "固定条件": "手動方式は準備2時間・同時上限2隻。実測値ではなく研究上の仮定",
        "イベント監査": "代表試行イベントは試行1だけ。船別実行は基準を含む全運用",
        "神戸港参照": "保存済み翌日予定件数から生成頻度の規模を設定。実績寄港数・実荷役時間ではない",
        "依存版": {"python": platform.python_version(),
                   **{name: importlib.metadata.version(name)
                      for name in ("numpy", "pandas", "openpyxl", "matplotlib")}},
        "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in filenames},
    }


def translated_table(table):
    """計算用の列を変えず、保存直前だけ日本語へ変換する。"""
    result = table.copy()
    if "method" in result:
        result["method"] = result.method.map(METHOD_LABELS)
    for column in ("long_handling",):
        if column in result:
            result[column] = result[column].map({True: "はい", False: "いいえ"})
    missing = set(result) - set(LABELS)
    if missing:
        raise ValueError(f"日本語見出し未定義: {missing}")
    front = [c for c in ("trial", "seed", "load", *CONDITION_COLUMNS) if c in result]
    columns = front + [c for c in result if c not in front]
    return result[columns].rename(columns=LABELS)


def write_result_sheet(workbook, name, table):
    """船別明細も逐次保存する。数値は丸めず、Excelの標準表示で不要な末尾の0を付けない。"""
    sheet = workbook.create_sheet(name)
    sheet.freeze_panes = "D2"
    sheet.sheet_view.showGridLines = False
    sheet.row_dimensions[1].height = 95
    headers = []
    for index, label in enumerate(table.columns, 1):
        sheet.column_dimensions[get_column_letter(index)].width = 27
        cell = WriteOnlyCell(sheet, label)
        cell.font = Font(name="Yu Gothic", bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="243B53")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        headers.append(cell)
    sheet.append(headers)
    for row in table.itertuples(index=False, name=None):
        cells = []
        for value in row:
            cell = WriteOnlyCell(sheet, None if pd.isna(value) else value)
            cells.append(cell)
        sheet.append(cells)
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(table.columns))}{len(table) + 1}"


def save_results(results, config, output):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("保存先が空ではありません。別の--outputを指定してください。")
    output.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.Workbook(write_only=True)
    for key, name in (("summary", "条件別集計"), ("runs", "試行別結果"),
                      ("inputs", "船別入力"), ("schedules", "船別実行"), ("events", "代表試行イベント")):
        write_result_sheet(workbook, name, translated_table(results[key]))
    workbook.save(output / "実験結果.xlsx")
    (output / "実験条件.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(results, output):
    table = results["summary"]
    selected = table[table.method == "spt"]
    columns = ["load", "duration", "mean_wait", "fifo_delta_wait", "fifo_delta_long_wait", "fifo_worse_ships"]
    print(translated_table(selected[columns]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"計算した運用数: {len(results['runs'])}、保存先: {output}")
