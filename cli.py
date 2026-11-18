"""コマンド引数の定義・読み取りと、保存先の確認を行う。"""

import argparse
from pathlib import Path


def create_parser():
    """利用できるコマンド引数と、その既定値を定義する。"""
    parser = argparse.ArgumentParser(
        description="4バースの到着予定と実際のずれの比較実験を実行し、設定・Excel・論文用グラフを保存する。"
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser


def parse_args(parser):
    """定義した引数を読み、保存先を確認する。"""
    args = parser.parse_args()
    # 直下にファイルがあれば、保存済みの結果を上書きしない。
    if not args.no_save and args.output.exists() and any(path.is_file() for path in args.output.iterdir()):
        parser.error("保存先にファイルがあります。既存結果を残すため、別の--outputを指定してください。")
    return args
