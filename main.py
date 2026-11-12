"""4バースの到着情報改ざん実験を実行し、設定・Excel・論文用グラフを保存する。"""

import json

from cli import create_parser, parse_args
from result_output import REFERENCE_DIR, create_config, print_summary, save_results
from simulation import DURATION_HOURS, NUM_BERTHS, POLICIES, run_experiment


def main():
    """引数の確認 → 保存済み参照値の読み込み → 実験の計算 → 保存・描画・表示。"""
    parser = create_parser()
    args = parse_args(parser)
    try:
        reference = json.loads((REFERENCE_DIR / "kobe_reference.json").read_text())
        interval = reference["mean_candidate_interval_hours"]
        results = run_experiment(
            trials=args.trials,
            num_berths=NUM_BERTHS,
            seed=args.seed,
            policies=POLICIES,
            duration_hours=DURATION_HOURS,
            mean_interarrival_time=interval,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if not args.no_save:
        config = create_config(args, results, interval)
        save_results(results, config, args.output)
        if not args.no_plot:
            from plotting import plot_results
            plot_results(results, args.output)
    print_summary(results, interval, None if args.no_save else args.output)


if __name__ == "__main__":
    main()
