"""GitHubのデータ用ブランチからCSVを取り込み、MacでExcelを更新する。"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from import_port_csv import JST, PROGRAM_DIR, ROOT, read_csv


DATA_BRANCH = "codex/kobe-port-data"


def sync(repository, output, csv_only=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", repository):
        raise ValueError("リポジトリは所有者/名前の形式で指定してください。")

    with tempfile.TemporaryDirectory(prefix="kobe_port_cloud_") as directory:
        clone = Path(directory) / "repository"
        environment = os.environ.copy()
        environment["GIT_TERMINAL_PROMPT"] = "0"
        for key in ("GIT_TRACE", "GIT_TRACE_CURL", "GIT_CURL_VERBOSE"):
            environment.pop(key, None)
        # データのブランチだけを読み取る。ダウンロードしたコードは実行しない。
        subprocess.run([
            "git", "clone", "--depth", "1", "--single-branch", "--no-tags",
            "--branch", DATA_BRANCH, f"https://github.com/{repository}.git", str(clone),
        ], check=True, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        source = clone / "data/kobe_port"
        files = sorted(source.glob("raw/*/*.csv"))
        if not files:
            raise ValueError("クラウドに取得済みCSVがありません。")

        # 全CSVと同名ファイルを確認してから追加する。ローカルの原本は削除しない。
        additions = []
        for path in files:
            for part in (clone / "data", source, source / "raw", path.parent, path):
                if part.is_symlink():
                    raise ValueError("CSVの保存先にシンボリックリンクがあります。")
            _, captured, _, _ = read_csv(path)
            if path.parent.name != captured.strftime("%Y-%m-%d"):
                raise ValueError("CSVの出力日と保存フォルダが一致しません。")
            destination = output / "raw" / path.parent.name / path.name
            if destination.exists():
                if destination.read_bytes() != path.read_bytes():
                    raise ValueError("同名で内容の異なるCSVがあるため、上書きを中止しました。")
            else:
                additions.append((path, destination))

        status_file = source / "last_collection.json"
        if status_file.is_symlink():
            raise ValueError("収集記録にシンボリックリンクがあります。")
        status = json.loads(status_file.read_text(encoding="utf-8"))
        for path, destination in additions:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        output.mkdir(parents=True, exist_ok=True)
        pending = output / "cloud_collection.pending.json"
        pending.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pending.replace(output / "cloud_collection.json")

    if not csv_only:
        subprocess.run([
            sys.executable, str(PROGRAM_DIR / "import_port_csv.py"), "--output", str(output),
        ], check=True)
    summary = {
        "synced_at": datetime.now(JST).isoformat(), "repository": repository,
        "branch": DATA_BRANCH, "added_csv": len(additions), "cloud_csv": len(files),
        "excel_updated": not csv_only,
    }
    pending = output / "last_cloud_sync.pending.json"
    pending.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(output / "last_cloud_sync.json")
    print(f"クラウドのCSV: {len(files)}ファイル、今回追加: {len(additions)}ファイル")
    if status.get("failed"):
        print("直近のクラウド収集に失敗があります。cloud_collection.jsonを確認してください。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, help="非公開リポジトリの所有者/名前")
    parser.add_argument("--output", type=Path, default=ROOT / "data/kobe_port")
    parser.add_argument("--csv-only", action="store_true", help="Excel作成を省略する")
    args = parser.parse_args()
    try:
        sync(args.repository, args.output.resolve(), args.csv_only)
    except Exception:
        print("同期を完了できませんでした。GitHubのHTTPS認証・データ用ブランチ・CSV・Excel作成環境を確認してください。")
        print("保存済みのCSVと前回のExcelは残しています。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
