"""4バースの到着予定と実際のずれの比較実験を実行し、設定・Excel・論文用グラフを保存する。"""

from cli import create_parser, parse_args
from result_output import create_config, print_summary, save_results
from simulation import DURATION_HOURS, NUM_BERTHS, POLICIES, run_experiment


def main():
    """引数の確認 → 実験の計算 → 保存・描画・表示。"""
    parser = create_parser()
    args = parse_args(parser)
    # 神戸港EDIの翌日予定30日分・226件。根拠は data/kobe_reference_20261105/README.md。
    interval = 3.186
    try:
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
