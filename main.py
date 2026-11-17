"""引数を読み、TOS停止実験の計算・保存・描画・表示を順に呼び出す。"""

import argparse
import json
from pathlib import Path

from result_output import ROOT, make_config, print_summary, save_results
from simulation import run_experiment


def create_parser():
    parser = argparse.ArgumentParser(description="TOS停止時の手動運用を準備2時間・同時上限2隻で比較する")
    parser.add_argument("--trials", type=int, default=10, help="試行数（既定：10）")
    parser.add_argument("--output", type=Path, default=Path("results"), help="空の保存先（既定：results）")
    return parser


def make_settings(trials, interval):
    """頻繁に変更しない研究条件は引数を増やさず、ここへまとめる。"""
    return {
        "trials": trials, "seed": 42,
        "interval": interval, "horizon": 168.0, "outage_start": 48.0,
        "loads": [1.0, 2.0], "num_berths": 4,
    }


def main():
    parser = create_parser()
    args = parser.parse_args()
    output = args.output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("保存先が空ではありません。別の--outputを指定してください。")
    try:
        reference = json.loads((ROOT / "data/kobe_reference_20261105/kobe_reference.json").read_text())
        settings = make_settings(args.trials, reference["mean_candidate_interval_hours"])
        results = run_experiment(settings)
        save_results(results, make_config(settings, results), output)
        from plotting import plot_results
        plot_results(results, output, settings)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print_summary(results, output)


if __name__ == "__main__":
    main()
