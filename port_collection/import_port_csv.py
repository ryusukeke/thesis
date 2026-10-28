"""神戸港EDIのCSV原本を保存し、重複をまとめたExcelを更新する。"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path


PROGRAM_DIR = Path(__file__).resolve().parent
ROOT = PROGRAM_DIR.parent
JST = timezone(timedelta(hours=9))
TABLES = {
    "入港船予定表": "https://edi.port.city.kobe.jp/WW01020101",
    "出港船表": "https://edi.port.city.kobe.jp/WW01020104",
    "在港船表": "https://edi.port.city.kobe.jp/WW01020102",
    "移動船表": "https://edi.port.city.kobe.jp/WW01020103",
    "航路通航船情報（中央）": "https://edi.port.city.kobe.jp/WW01020201?data=04",
    "航路通航船情報（神戸西）": "https://edi.port.city.kobe.jp/WW01020201?data=01",
    "航路通航船情報（六甲東）": "https://edi.port.city.kobe.jp/WW01020201?data=06",
}
RUNTIME = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"


def read_csv(path):
    """出力ファイル名・見出し・列数を確認してから読み込む。"""
    match = re.fullmatch(r"(\d{16})_(.+)\.csv", path.name)
    if not match:
        raise ValueError(f"神戸港EDIのCSV出力名ではありません: {path.name}")
    stamp, table = match.groups()
    if table not in TABLES:
        raise ValueError(f"収集対象の表ではありません: {table}")
    captured = datetime.strptime(stamp[:14], "%Y%m%d%H%M%S").replace(
        microsecond=int(stamp[14:]) * 10000, tzinfo=JST
    )
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows or not {"コールサイン", "船名"}.issubset(rows[0]):
        raise ValueError(f"船舶表の見出しがありません: {path.name}")
    headers = rows[0]
    records = [row for row in rows[1:] if any(row)]
    if any(len(row) != len(headers) for row in records):
        raise ValueError(f"見出しとデータの列数が一致しません: {path.name}")
    return table, captured, headers, records


def duplicate_key(headers, row, captured):
    """全列一致を基本とし、年の違う同じ月日の記録は分ける。"""
    years = []
    for header, value in zip(headers, row):
        if not header.endswith(("日時", "時刻")) or not re.fullmatch(r"\d{2}/\d{2} \d{2}:\d{2}", value):
            continue
        candidates = []
        for year in range(captured.year - 1, captured.year + 2):
            try:
                date = datetime.strptime(f"{year}/{value}", "%Y/%m/%d %H:%M").replace(tzinfo=JST)
                candidates.append(date)
            except ValueError:
                continue
        if not candidates:
            raise ValueError(f"日時を読み取れません: {value}")
        years.append(min(candidates, key=lambda date: abs(date - captured)).year)
    return (tuple(row), tuple(years) if years else (captured.year,))


def accumulate(output):
    """保存済みの原本を毎回読み、同一行の初回・最終確認をまとめる。"""
    tables = {name: {"headers": [], "records": {}} for name in TABLES}
    history = []
    files = sorted((output / "raw").glob("*/*.csv"), key=lambda path: path.name)
    for path in files:
        name, captured, headers, rows = read_csv(path)
        table = tables[name]
        if table["headers"] and table["headers"] != headers:
            raise ValueError(f"CSVの列構成が変わっています。確認が必要です: {path.name}")
        table["headers"] = headers
        new_count = 0
        seen = set()
        for row in rows:
            key = duplicate_key(headers, row, captured)
            if key in seen:
                continue
            seen.add(key)
            stamp = captured.isoformat()
            relative = str(path.relative_to(output))
            if key not in table["records"]:
                table["records"][key] = {
                    "values": row, "firstSeen": stamp, "lastSeen": stamp,
                    "count": 1, "firstFile": relative, "lastFile": relative,
                }
                new_count += 1
            else:
                record = table["records"][key]
                record["lastSeen"] = stamp
                record["lastFile"] = relative
                record["count"] += 1
        history.append([
            captured.isoformat(), name, len(rows), len(seen), new_count,
            len(seen) - new_count, len(rows) - len(seen), str(path.relative_to(output)), TABLES[name],
        ])
    for table in tables.values():
        table["records"] = list(table["records"].values())
    return {"tables": tables, "history": history, "sources": TABLES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*", type=Path, help="サイトから取得したCSV。省略すると原本から再作成")
    parser.add_argument("--output", type=Path, default=ROOT / "data/kobe_port")
    parser.add_argument("--preview-dir", type=Path, help="指定時は各シートのプレビューも保存")
    args = parser.parse_args()
    output = args.output.resolve()

    # 読み取りと上書きの確認を済ませてから、原本を追加する。
    additions = []
    for source in args.csv:
        _, captured, _, _ = read_csv(source)
        destination = output / "raw" / captured.strftime("%Y-%m-%d") / source.name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError(f"同名の異なるCSVがあります: {destination}")
        additions.append((source, destination))
    for source, destination in additions:
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    data = accumulate(output)
    if not data["history"]:
        raise ValueError("保存済みCSVがありません。取得したCSVを指定してください。")
    node = RUNTIME / "bin/node"
    modules = RUNTIME / "node_modules"
    if not node.exists() or not modules.exists():
        raise FileNotFoundError("Codex付属のNode.jsが見つかりません。workspace dependenciesを確認してください。")

    # 依存先は一時フォルダだけに接続し、Excelは完成してから置き換える。
    with tempfile.TemporaryDirectory(prefix="kobe_port_") as temporary:
        work = Path(temporary)
        (work / "node_modules").symlink_to(modules, target_is_directory=True)
        builder = work / "build_port_workbook.mjs"
        shutil.copy2(PROGRAM_DIR / "build_port_workbook.mjs", builder)
        payload = work / "data.json"
        payload.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temporary_xlsx = work / "船舶動静_蓄積.xlsx"
        command = [str(node), str(builder), str(payload), str(temporary_xlsx)]
        if args.preview_dir:
            command.append(str(args.preview_dir.resolve()))
        subprocess.run(command, check=True)
        pending = output / "船舶動静_蓄積.pending.xlsx"
        shutil.copy2(temporary_xlsx, pending)
        pending.replace(output / "船舶動静_蓄積.xlsx")
    for name, table in data["tables"].items():
        if not table["headers"]:
            continue
        print(f"{name}: {len(table['records'])}件（重複整理後の履歴）")
    print(f"保存先: {output / '船舶動静_蓄積.xlsx'}")


if __name__ == "__main__":
    main()
