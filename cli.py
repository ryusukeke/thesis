"""コマンド引数の定義・読み取りと、保存先の確認を行う。"""

import argparse
from pathlib import Path


def create_parser():
    """利用できるコマンド引数と、その既定値を定義する。"""
    parser = argparse.ArgumentParser(
        description="4バースの到着情報改ざん実験を実行し、設定・CSV・論文用グラフを保存する。"
    )
    parser.add_argument("--study", action="store_true", help="神戸港の予定規模を参照した主実験と補足実験を実行")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/final_20261105"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser


def parse_args(parser):
    """定義した引数を読み、保存先を確認する。"""
    args = parser.parse_args()
    if not args.no_save and args.output.exists() and any(args.output.iterdir()):
        parser.error("保存先にファイルがあります。既存結果を残すため、別の--outputを指定してください。")
    return args
