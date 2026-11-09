"""実験条件の記録、計算結果のCSV・JSON保存、画面への概要表示を担当する。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulation import HANDLING_SCALE, HANDLING_SHAPE

ROOT = Path(__file__).resolve().parent


def create_config(args, results, duration, interval, policies, reference):
    """実行した条件と、再現に必要なコード・ライブラリの情報をまとめる。"""
    if args.interval is not None:
        interval_source = "explicit_argument"
    elif reference:
        interval_source = "kobe_day_ahead_forecasts"
    else:
        interval_source = "rounded_assumption_3h"

    attacks = results["run_metrics"]
    attacks = attacks[attacks.scenario != "normal"]
    return {
        "model_version": "discrete_4berth_bounded_advance_v2",
        "full_study": args.study,
        "trials_per_cell": args.trials,
        "first_seed": args.seed,
        "last_seed": args.seed + args.trials - 1,
        "num_berths": args.berths,
        "num_ships": args.ships,
        "candidate_duration_hours": duration,
        "base_mean_candidate_interval_hours": interval,
        "interval_source": interval_source,
        "demand_multipliers": [0.75, 1.0, 1.5] if args.study else [1.0],
        "target_positions": [0.1, 0.5, 0.9] if args.study else ["V5"],
        "shift_magnitudes_hours": [2.0, 4.0, 8.0] if args.study else [args.shift],
        "main_early_start_limit_hours": args.advance,
        "unrestricted_early_start_limit_hours": None,
        "sensitivity": "reassign, middle target, +/-4h, limit0" if args.study else None,
        "policies": list(policies),
        "handling_shape": HANDLING_SHAPE,
        "handling_scale": HANDLING_SCALE,
        "handling_model": "base_times_1_plus_0.1_times_berth_index_distance",
        "cargo_nearest_berth": "uniform_integer_from_1_to_berths",
        "attribute_rng_seed": "trial_seed+100000; per-ship gamma then cargo position",
        "normal_arrival_adjustment": "normal_reserved_start; no upper limit; no speed or fuel model",
        "plan_update": "keep_normal_reservations; replan_once_if_report_changes",
        "initial_berth_state": "all_empty",
        "period_end": "finish_all_candidate_ships_including_adjusted_arrivals_after_period",
        "same_time_order": "arrivals_and_releases_and_completions_together; stable_input_order",
        "distinct_normal_plans": len(results["plan_diagnostics"]),
        "main_attack_executions": int((attacks.experiment_group == "main").sum()),
        "sensitivity_attack_executions": int((attacks.experiment_group == "sensitivity").sum()),
        "normal_metrics": "shared across shifts; repeated target-specific metrics are not independent plans",
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "code_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "simulation.py", "main.py", "cli.py", "result_output.py", "plotting.py",
                "port_collection/analyze_port_data.py",
            )
        },
        "sources": {
            "berth_count": "https://www.mol.co.jp/info/article/2026/0819.html",
            "rescheduling": "https://transp-or.epfl.ch/documents/technicalReports/NitBierEra2013.pdf",
            "information_manipulation": "https://doi.org/10.1088/1742-6596/2867/1/012047",
        },
    }


def save_results(results, config, output):
    """計算結果7表をCSV、実験条件をJSONで保存する。"""
    output.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def print_summary(results, interval, output):
    """実行件数と全主実験条件の平均を表示する。outputがNoneなら保存なし。"""
    runs = results["run_metrics"]
    normal = runs[runs.scenario == "normal"]
    attacks = runs[runs.scenario != "normal"]
    main_attacks = attacks[attacks.experiment_group == "main"]
    sensitivity_attacks = attacks[attacks.experiment_group == "sensitivity"]
    print(
        f"正常計画 {len(results['plan_diagnostics'])}組、"
        f"主実験の改ざん運用 {len(main_attacks)}ケース、補足 {len(sensitivity_attacks)}ケース"
    )
    print(
        f"正常時の最大待ち時間：{normal.max_waiting_time.max():.9f}時間。"
        f"到着候補の基準平均間隔：{interval:.6f}時間。"
    )
    print("下表は全主実験条件を同じ重みで平均した概要です。条件別の比較はsummary.csvを参照してください。")
    metrics = ["other_average_waiting_time", "other_waiting_ships", "berth_changes"]
    print(main_attacks.groupby(["policy", "scenario"])[metrics].mean().round(4).to_string())
    print("実測の荷役時間・実待ち時間による検証ではありません。到着調整量はplan_diagnostics.csvに記録します。")
    print("保存なし" if output is None else f"保存先：{output.resolve()}")
