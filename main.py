"""4バースの到着情報改ざん実験を実行し、設定・CSV・論文用グラフを保存する。"""

from cli import create_parser, parse_args
from port_collection.analyze_port_data import analyze_forecasts, save_reference
from result_output import create_config, print_summary, save_results
from simulation import MEAN_INTERARRIVAL_TIME, POLICIES, run_experiment


def main():
    """引数の確認 → 参照データの集計 → 実験の計算 → 保存・描画・表示。"""
    parser = create_parser()
    args = parse_args(parser)
    duration = None if args.ships is not None else (args.duration if args.duration is not None else 168.0)
    policies = POLICIES if args.policy == "all" else (args.policy,)
    try:
        reference = None
        if args.study:
            reference, daily, selected = analyze_forecasts(args.raw, args.through)
        if args.interval is not None:
            interval = args.interval
        elif reference:
            interval = reference["mean_candidate_interval_hours"]
        else:
            interval = MEAN_INTERARRIVAL_TIME
        results = run_experiment(
            trials=args.trials,
            num_ships=args.ships if args.ships is not None else 10,
            num_berths=args.berths,
            seed=args.seed,
            shift_hours=args.shift,
            policies=policies,
            duration_hours=duration,
            mean_interarrival_time=interval,
            early_start_limit=args.advance,
            full_study=args.study,
        )
    except ValueError as error:
        parser.error(str(error))

    if not args.no_save:
        config = create_config(args, results, duration, interval, policies, reference)
        save_results(results, config, args.output)
        if reference:
            save_reference(args.output, reference, daily, selected)
        if not args.no_plot:
            from plotting import plot_results
            plot_results(results, args.output, args.study)
    print_summary(results, interval, None if args.no_save else args.output)


if __name__ == "__main__":
    main()
