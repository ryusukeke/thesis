"""実験条件の記録、計算結果のCSV・JSON保存、画面への概要表示を担当する。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulation import DURATION_HOURS, NUM_BERTHS, POLICIES, SHIFT_HOURS

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "data/kobe_reference_20261105"
REFERENCE_FILES = ("kobe_reference.json", "kobe_daily_forecasts.csv", "kobe_reference_rows.csv")
CODE_FILES = ("simulation.py", "main.py", "cli.py", "result_output.py", "plotting.py")


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
        ["demand_multiplier", "trial", "policy"]
    )
    normal_waiting = normal.groupby(["demand_multiplier", "policy"], as_index=False).agg(
        trials=("trial", "count"),
        mean_waiting_hours=("average_waiting_time", "mean"),
        max_waiting_hours=("max_waiting_time", "max"),
    )
    attacks = attacks[attacks.scenario != "normal"]
    return {
        # 実験の対象条件
        "model_version": "discrete_4berth_multi_ship_v5",
        "full_study": args.study,
        "trials_per_cell": args.trials,
        "first_seed": args.seed,
        "last_seed": args.seed + args.trials - 1,
        "num_berths": NUM_BERTHS,
        "num_ships": None,  # 固定隻数を指定せず、期間内の実到着をすべて生成する。
        "arrival_duration_hours": DURATION_HOURS,
        "base_mean_interarrival_hours": interval,
        "interval_source": "kobe_day_ahead_forecasts",
        "demand_multipliers": [0.75, 1.0, 1.5] if args.study else [1.0],
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
        "normal_occupancy_by_demand": results["plan_diagnostics"].groupby(
            "demand_multiplier"
        )["normal_occupancy"].mean().to_dict(),
        "schedule_storage": "one illustrative trial; maximum fixed mean-wait increase in predefined cell",
        "example_cell": "demand_1_p30_mixed_4h" if args.study else "demand_1_single_early_4h",
        "normal_plan_diagnostics": {
            "last_actual_arrival_hours": results["plan_diagnostics"]["last_actual_arrival"].max(),
            "last_planned_completion_hours": results["plan_diagnostics"]["last_planned_completion"].max(),
        },
        # 再現確認に必要な版・ファイル・資料
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "code_sha256": calculate_file_hashes(ROOT, CODE_FILES),
        "reference_sha256": calculate_file_hashes(REFERENCE_DIR, REFERENCE_FILES),
        "sources": {
            "berth_count": "https://www.mol.co.jp/info/article/2026/0819.html",
            "rescheduling": "https://transp-or.epfl.ch/documents/technicalReports/NitBierEra2013.pdf",
            "information_manipulation": "https://doi.org/10.1088/1742-6596/2867/1/012047",
        },
    }


def save_results(results, config, output):
    """試行別結果・集計の2表と実験条件を保存する。"""
    output.mkdir(parents=True, exist_ok=True)
    for name in ("run_metrics", "summary"):
        results[name].to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def print_summary(results, interval, output):
    """実行件数と全主実験条件の平均を表示する。outputがNoneなら保存なし。"""
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
    print("下表は全主実験条件を同じ重みで平均した概要です。条件別の比較はsummary.csvを参照してください。")
    summary = results["summary"]
    all_wait = summary[summary.metric == "average_waiting_time"]
    print("全船の平均待ち時間（時間）。mean_differenceは同じ方式の正常時からの増減です。")
    print(all_wait.groupby(["policy", "scenario"])[
        ["normal_mean", "mean", "mean_difference"]
    ].mean().round(4).to_string())
    print("実到着は固定し、正常時にも待ちを認めます。実測の荷役時間・実待ち時間による検証ではありません。")
    print("保存なし" if output is None else f"保存先：{output.resolve()}")
