"""4バースの到着情報改ざん実験を実行し、設定・CSV・論文用グラフを保存する。"""

from pathlib import Path

from cli import create_parser, parse_args
from port_collection.analyze_port_data import analyze_forecasts, save_reference
from result_output import create_config, print_summary, save_results
from simulation import (DURATION_HOURS, EARLY_START_LIMIT, NUM_BERTHS, POLICIES,
                        SHIFT_HOURS, run_experiment)

# 研究に使う参照データ。日々の収集で過去の実験条件が変わらないよう取得日上限を固定する。
KOBE_RAW_DIR = Path(__file__).resolve().parent / "data/kobe_port/raw"
KOBE_THROUGH_DATE = "2026-11-04"


def main():
    """引数の確認 → 参照データの集計 → 実験の計算 → 保存・描画・表示。"""
    parser = create_parser()
    args = parse_args(parser)
    try:
        reference, daily, selected = analyze_forecasts(KOBE_RAW_DIR, KOBE_THROUGH_DATE)
        interval = reference["mean_candidate_interval_hours"]
        results = run_experiment(
            trials=args.trials,
            num_berths=NUM_BERTHS,
            seed=args.seed,
            shift_hours=SHIFT_HOURS,
            policies=POLICIES,
            duration_hours=DURATION_HOURS,
            mean_interarrival_time=interval,
            early_start_limit=EARLY_START_LIMIT,
            full_study=args.study,
        )
    except ValueError as error:
        parser.error(str(error))

    if not args.no_save:
        config = create_config(args, results, interval)
        save_results(results, config, args.output)
        save_reference(args.output, reference, daily, selected)
        if not args.no_plot:
            from plotting import plot_results
            plot_results(results, args.output, args.study)
    print_summary(results, interval, None if args.no_save else args.output)


if __name__ == "__main__":
    main()
