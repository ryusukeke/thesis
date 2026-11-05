"""KICTの翌日入港予定を日ごとの最終取得CSVから集計する。収集原本は変更しない。"""

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KICT_SEATS = ("PC-14", "PC-15(E)", "PC-16", "PC-17")


def analyze_forecasts(raw_dir, through_date="2026-11-04"):
    """1日1スナップショットの翌日予定を数える。実績寄港数とは解釈しない。"""
    latest = {}
    for path in sorted(Path(raw_dir).glob("*/*_入港船予定表.csv")):
        capture = datetime.strptime(path.name[:14], "%Y%m%d%H%M%S")
        day = capture.date().isoformat()
        if day <= through_date:
            latest[day] = path
    if not latest:
        raise ValueError("指定期間の入港船予定表がありません。")
    daily, selected, sources = [], [], []
    for day, path in sorted(latest.items()):
        capture = datetime.strptime(path.name[:14], "%Y%m%d%H%M%S")
        next_day = capture.date() + timedelta(days=1)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"コールサイン", "船名", "船席", "入港日時"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"見出しを確認してください: {path.name}")
            rows = list(reader)
        seen = set()
        count, merged = 0, 0
        for line, row in enumerate(rows, 2):
            if row["船席"] not in KICT_SEATS:
                continue
            if not row["入港日時"].startswith(next_day.strftime("%m/%d ")):
                continue
            arrival = datetime.strptime(f"{next_day.year}/{row['入港日時']}", "%Y/%m/%d %H:%M")
            identity = row["コールサイン"].strip() or row["船名"].strip()
            if not identity:
                raise ValueError(f"船の識別情報がありません: {path.name}:{line}")
            key = (identity, next_day.isoformat())
            duplicate = key in seen
            merged += int(duplicate)
            count += int(not duplicate)
            seen.add(key)
            selected.append({"snapshot_date": day, "capture_time_jst": capture.isoformat(),
                             "forecast_date": next_day.isoformat(), "call_sign": row["コールサイン"],
                             "ship_name": row["船名"], "berth_code": row["船席"],
                             "scheduled_arrival": arrival.isoformat(), "counted": not duplicate,
                             "source_file": path.name, "source_line": line})
        daily.append({"snapshot_date": day, "forecast_date": next_day.isoformat(),
                      "capture_time_jst": capture.isoformat(), "scheduled_ship_days": count,
                      "merged_same_ship_day_rows": merged, "source_file": path.name})
        sources.append({"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    total = sum(row["scheduled_ship_days"] for row in daily)
    if total == 0:
        raise ValueError("対象船席の翌日入港予定がありません。")
    reference = {
        "method": "last_snapshot_each_day_next_calendar_day_forecasts",
        "through_snapshot_date": through_date, "berth_codes": list(KICT_SEATS),
        "observed_snapshot_days": len(daily), "scheduled_ship_days": total,
        "forecast_period_start": daily[0]["forecast_date"], "forecast_period_end": daily[-1]["forecast_date"],
        "mean_daily_forecasts": total / len(daily),
        "mean_candidate_interval_hours": 24 * len(daily) / total,
        "expected_weekly_candidates": 7 * total / len(daily),
        "deduplication": "call_sign_or_ship_name_and_forecast_date_within_selected_snapshot",
        "limitations": ["予定件数であり実績寄港数ではない", "同一船・同一日の複数寄港は1件にまとめる",
                        "予定日の変更で同じ実寄港が別日の予報に現れる可能性がある",
                        "未取得日は0件と補わない", "分布形・荷役時間・船席ごとの能力は推定していない"],
        "sources": sources,
    }
    return reference, daily, selected


def save_reference(output, reference, daily, selected):
    """解析結果を原本と別の場所へ保存する。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("kobe_daily_forecasts", daily), ("kobe_reference_rows", selected)):
        with (output / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "kobe_reference.json").write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/kobe_port/raw")
    parser.add_argument("--through", default="2026-11-04")
    parser.add_argument("--output", type=Path, default=ROOT / "results/kobe_reference_20261105")
    args = parser.parse_args()
    ref, days, rows = analyze_forecasts(args.raw, args.through)
    save_reference(args.output, ref, days, rows)
    print(f"{ref['observed_snapshot_days']}日・{ref['scheduled_ship_days']}件の翌日予定。候補の平均間隔 {ref['mean_candidate_interval_hours']:.6f}時間。")
