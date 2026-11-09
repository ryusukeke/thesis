"""コマンド引数の定義・読み取りと、主実験の条件・保存先の確認を行う。"""

import argparse
from pathlib import Path

from simulation import POLICIES

ROOT = Path(__file__).resolve().parent


def create_parser():
    """利用できるコマンド引数と、その既定値を定義する。"""
    parser = argparse.ArgumentParser(
        description="4バースの到着情報改ざん実験を実行し、設定・CSV・論文用グラフを保存する。"
    )
    parser.add_argument("--study", action="store_true", help="神戸港の予定規模を参照した主実験と補足実験を実行")
    parser.add_argument("--trials", type=int, default=10)
    period = parser.add_mutually_exclusive_group()
    period.add_argument("--ships", type=int, help="単一設定の確認用の固定隻数")
    period.add_argument("--duration", type=float, default=None)
    parser.add_argument("--berths", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shift", type=float, default=4.0, help="単一設定での変更幅。主実験は2・4・8時間")
    parser.add_argument("--interval", type=float, default=None, help="候補の平均間隔。主実験の省略時は神戸港予定の集計値")
    parser.add_argument("--advance", type=float, default=2.0, help="同じバース内調整・バース変更の前倒し上限")
    parser.add_argument("--policy", choices=("all", *POLICIES), default="all")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/kobe_port/raw")
    parser.add_argument("--through", default="2026-11-04", help="参照するCSVの取得日上限")
    parser.add_argument("--output", type=Path, default=Path("results/final_20261105"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser


def parse_args(parser):
    """定義した引数を読み、主実験の固定条件と保存先を確認する。"""
    args = parser.parse_args()
    if args.study and (args.ships is not None or args.policy != "all" or args.berths != 4
                       or args.advance != 2 or args.duration not in (None, 168.0)):
        parser.error("主実験は168時間・4バース・全方式・前倒し上限2時間で実行します。")
    if not args.no_save and args.output.exists() and any(args.output.iterdir()):
        parser.error("保存先にファイルがあります。既存結果を残すため、別の--outputを指定してください。")
    return args
