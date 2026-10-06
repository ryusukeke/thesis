# run_experiment.py の実行・出力処理の説明

作成日：2026年10月7日

対象：[run_experiment.py](../run_experiment.py)

## 1. ファイルの役割

`sample.py` で計算した結果を、CSV・設定JSON・グラフとして保存し、画面に表示する。
コマンド引数の読み取りと実験の呼び出しもこのファイルが担当する。
計算方法と実験条件は、2026年10月6日の実装から変更していない。

```text
python run_experiment.py
    ↓
main()：コマンド引数を読み取る
    ↓
sample.run_experiment()：計算結果を5つの表として返す
    ↓
main()：CSV5個とconfig.jsonを保存
    ↓
save_plot()：有効な場合にグラフを保存
    ↓
main()：平均値・変化の回数・保存先を表示
```

計算部分の詳細は [sample.pyの実装説明](sample_implementation.md) を参照する。

## 2. ライブラリと表示名

| 読み込むもの | 用途 |
| --- | --- |
| `argparse` | コマンドの引数を読み取る |
| `json` | 実験条件を設定ファイルに変換する |
| `pathlib.Path` | 保存先フォルダとファイルを扱う |
| `numpy`、`pandas` | 設定JSONへのバージョン記録などに使う |
| `sample` の定数と `run_experiment()` | 計算条件を参照し、実験を実行する |
| `matplotlib` | `save_plot()` の中で読み込み、グラフを作る |

`SCENARIO_NAMES` は条件名と日本語表示名、`METRIC_NAMES` は評価指標名と日本語表示名を対応づける辞書である。
計算処理の識別名は `sample.py` にあり、表示用の日本語名はこのファイルで管理する。

## 3. save_plot()：比較グラフの作成

```python
save_plot(summary, output_dir, target_ship, shift_hours)
```

集計表、保存先、対象船、変更幅を受け取り、平均待ち時間の棒グラフを作る。

`matplotlib.use("Agg")` により、グラフのウィンドウを表示せずに画像ファイルを作成する。
グラフは左右2つに分け、左は全船の平均待ち時間、右は対象船以外の平均待ち時間を表示する。
各図に正常・早い申告・遅い申告の3本の棒を並べ、小数点以下3桁の値を付ける。

フォントへの依存を減らすため、図中の見出しと軸名は英語表記とする。
タイトルには対象船、変更幅、試行回数を入れ、縦軸は時間で表示する。
左右の縦軸の上限は、それぞれの平均値に合わせて設定する。
棒は平均値であり、標準偏差や信頼区間を示す線は付けない。

`waiting_time_comparison.png` を180dpiで保存し、`plt.close(fig)` で図を閉じる。
この関数はファイルを保存する処理であり、結果の表は返さない。

## 4. main()：設定、保存、表示

`main()` はコマンドからの実行を取りまとめる。

### 4.1 コマンド引数の読み取り

| 引数 | 初期値 | 意味 |
| --- | --- | --- |
| `--trials` | 100 | 試行回数 |
| `--ships` | 10 | 船数 |
| `--berths` | 3 | バース数 |
| `--seed` | 42 | 最初の試行のseed |
| `--attack-ship` | V5 | 対象船 |
| `--shift` | 4.0 | 早める・遅らせる時間の大きさ |
| `--output` | results | 保存先のフォルダ |
| `--no-plot` | 指定なし | 指定するとグラフを作成しない |

`argparse` が引数の型と値を読み取り、`run_experiment()` に渡す。
実験設定に関する `ValueError` は `parser.error()` でコマンドのエラーとして表示する。
コマンド引数の一覧は `python run_experiment.py --help` で確認できる。

### 4.2 CSVと設定JSONの保存

`mkdir(parents=True, exist_ok=True)` で保存先フォルダを作る。
実験の戻り値の5つの表について、辞書のキーをファイル名にしてCSVを保存する。

| 保存ファイル | 元となる表 |
| --- | --- |
| `ship_data.csv` | 元の船舶データ |
| `schedules.csv` | 船別の割当・実行結果 |
| `run_metrics.csv` | 試行・条件別の評価指標 |
| `summary.csv` | 条件別の平均・標準偏差 |
| `paired_comparison.csv` | 正常時との差と変化の回数 |

`index=False` は表の行索引をCSVに書かない指定である。
`encoding="utf-8-sig"` は、Excelで読み込みやすいUTF-8 BOM付きの形式を指定する。

`config.json` には、試行回数、船数、バース数、最初と最後のseed、対象船、変更幅、分布パラメータ、バース係数、時間単位、実行方式、NumPy・Pandasのバージョンを保存する。
`ensure_ascii=False` と `indent=2` により、日本語を保った読みやすいJSONにする。

保存先は、コマンドを実行した場所からの相対パス、または指定した絶対パスで決まる。
同じ保存先では同名のファイルを上書きするため、別の実験結果を残す場合は `--output` を変える。

### 4.3 グラフの保存と例外処理

古いグラフが新しいCSVと混在しないよう、既存の `waiting_time_comparison.png` を削除する。
`missing_ok=True` により、ファイルが存在しなくてもエラーにはならない。

`--no-plot` が指定されていなければ `save_plot()` を呼ぶ。
Matplotlib自体が見つからない場合は案内を表示し、保存済みのCSVを利用できるようにする。
Matplotlib以外のモジュールに関する読み込みエラーは、そのままエラーとして扱う。

### 4.4 ターミナルへの表示

実験条件と、次の4指標の100試行平均を表示する。

- 全船の平均待ち時間。
- 全船の平均在港時間。
- 対象船の待ち時間。
- 対象船以外の平均待ち時間。

`pivot()` で、条件を行、指標を列にした表示用の表に変換する。
条件名と指標名を日本語に置き換え、小数点以下3桁に丸めて表示する。
試行回数を変更した場合は、その指定回数の平均となる。

続いて、早い申告・遅い申告について、全船の平均待ち時間の正常時との差を表示する。
悪化・改善・変化なしの試行回数と、保存先の絶対パスも表示する。
表示時の丸めはCSVに保存する値には適用しない。

## 5. ファイル末尾：実行の入口

```python
if __name__ == "__main__":
    main()
```

この条件は、`python run_experiment.py` のようにファイルを直接実行したときだけ `main()` を呼ぶためのものである。
別ファイルから `import run_experiment` した場合は、関数と設定値を読み込むだけで、実験やファイル保存は始まらない。
`sample.py` 自体にはこの入口処理は置かず、計算関数を定義するだけにしている。

## 6. 実行と結果の確認

リポジトリのルートで、ライブラリをインストールした仮想環境から実行する。

```bash
# 初期設定で実験する
python run_experiment.py

# 1回だけ実行し、元の100回の結果とは別の場所に保存する
python run_experiment.py --trials 1 --output results_single

# 対象船と変更幅を変える
python run_experiment.py --attack-ship V7 --shift 2 --output results_v7

# 自動テストを実行する
python -m unittest -v
```

結果を見るときは、最初に `config.json` で設定を確認し、`summary.csv` で条件別の平均を見る。
次に `paired_comparison.csv` で同じ試行の正常時との差を確認する。
特定の船の変化を調べる場合は、`schedules.csv` を試行番号・条件名・船番号で絞る。
