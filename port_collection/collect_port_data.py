"""神戸港EDIからCSVを取得する。ローカルではExcelの更新も行う。"""

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
from urllib.parse import urljoin, urlsplit

from import_port_csv import JST, PROGRAM_DIR, ROOT, TABLES, read_csv


SITE = "https://edi.port.city.kobe.jp"
LOGIN_URL = SITE + "/WW01010701"
OPTIONAL_TABLES = {
    "航路通航船情報（神戸西）": "神戸西",
    "航路通航船情報（六甲東）": "六甲東",
}


def read_credentials(source="file"):
    """ローカルの.env、またはクラウドのSecretsから値を読み込む。"""
    if source == "env":
        user_id = os.environ.pop("KOBE_EDI_ID", None)
        password = os.environ.pop("KOBE_EDI_PASSWORD", None)
        if not user_id or not password:
            raise ValueError("クラウドの認証情報を設定してください。")
        return user_id, password

    from dotenv import dotenv_values

    path = ROOT / ".env"
    if path.is_symlink() or not path.is_file():
        raise ValueError("研究フォルダの.envファイルを用意してください。")
    if path.stat().st_uid != os.getuid():
        raise ValueError(".envの所有者が実行者と異なります。")
    path.chmod(0o600)
    values = dotenv_values(path, interpolate=False)
    user_id = values.get("KOBE_EDI_ID")
    password = values.get("KOBE_EDI_PASSWORD")
    if not user_id or not password:
        raise ValueError(".envのKOBE_EDI_IDとKOBE_EDI_PASSWORDを入力してください。")
    return user_id, password


def require_official_site(url):
    """認証情報の入力先を、確認済みのHTTPSサイトに限定する。"""
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != "edi.port.city.kobe.jp"
            or parsed.port not in (None, 443) or parsed.username or parsed.password):
        raise ValueError("神戸港EDI以外のページには認証情報を入力しません。")


def login(page, user_id, password):
    page.goto(LOGIN_URL, wait_until="domcontentloaded")
    require_official_site(page.url)
    action = page.locator("form").get_attribute("action")
    require_official_site(urljoin(page.url, action or page.url))
    page.locator("#UserId").fill(user_id)
    page.locator("#Password").fill(password)
    page.locator("#Login").click()
    page.get_by_role("link", name="ログアウト", exact=True).wait_for(state="visible")
    page.goto(SITE + "/", wait_until="domcontentloaded")
    require_official_site(page.url)
    page.get_by_role("link", name="ログアウト", exact=True).wait_for(state="visible")


def choose_tables(page, status):
    """基本4表と中央に、工事終了を確認できた航路を加える。"""
    targets = {name: url for name, url in TABLES.items() if name not in OPTIONAL_TABLES}
    for name, label in OPTIONAL_TABLES.items():
        link = page.get_by_role("link", name=re.compile(r"^" + label + r"(?:\s|※|$)"))
        if link.count() != 1:
            status["skipped"][name] = "ログイン後のホームでリンクを確認できない"
            status["failed"].append(name)
            continue
        if "工事中" in link.inner_text():
            status["skipped"][name] = "工事中"
            continue
        url = urljoin(SITE, link.get_attribute("href") or "")
        require_official_site(url)
        targets[name] = url
    return targets


def download_table(page, name, url, work, output):
    page.goto(url, wait_until="domcontentloaded")
    require_official_site(page.url)
    page.get_by_role("heading", name=name, exact=True).wait_for(state="visible")
    page.get_by_role("button", name="更新", exact=True).click()
    page.get_by_role("heading", name=name, exact=True).wait_for(state="visible")
    with page.expect_download(timeout=30000) as event:
        page.locator("#download_csv").click()
    download = event.value
    filename = download.suggested_filename
    if Path(filename).name != filename or "\\" in filename:
        raise ValueError("CSVファイル名を確認できません。")
    temporary = work / filename
    download.save_as(temporary)
    table, captured, _, rows = read_csv(temporary)
    if table != name:
        raise ValueError("取得したCSVの表名が一致しません。")
    destination = output / "raw" / captured.strftime("%Y-%m-%d") / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != temporary.read_bytes():
            raise ValueError("同名で内容の異なるCSVがあります。")
    else:
        shutil.copy2(temporary, destination)
    return len(rows)


def collect(user_id, password, output, status, browser_name="chrome"):
    # ブラウザの詳細ログにfillの引数などを出さない。
    for key in ("DEBUG", "DEBUG_FILE", "PWDEBUG"):
        os.environ.pop(key, None)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        status["stage"] = "収集用ブラウザの起動"
        channel = "chrome" if browser_name == "chrome" else None
        browser = playwright.chromium.launch(channel=channel, headless=True)
        try:
            # 普段のChromeのプロファイルや保存済みパスワードは使わない。
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()
            page.set_default_timeout(30000)
            page.set_default_navigation_timeout(30000)
            status["stage"] = "ログイン"
            login(page, user_id, password)
            status["stage"] = "工事中表示の確認"
            targets = choose_tables(page, status)
            with tempfile.TemporaryDirectory(prefix="kobe_port_download_") as directory:
                for name, url in targets.items():
                    status["stage"] = "CSV取得"
                    try:
                        count = download_table(page, name, url, Path(directory), output)
                        status["collected"][name] = count
                        print(f"取得: {name} {count}行")
                    except Exception:
                        # 生の例外には入力値やページ内容が含まれ得るため出さない。
                        status["failed"].append(name)
                        print(f"取得失敗: {name}（ページ・通信・CSV形式を確認してください）")
        finally:
            browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/kobe_port")
    parser.add_argument("--csv-only", action="store_true", help="CSVのみ取得し、Excel作成を省略する")
    parser.add_argument("--credentials", choices=("file", "env"), default="file",
                        help="fileはローカルの.env、envはクラウドのSecretsを使用")
    parser.add_argument("--browser", choices=("chrome", "chromium"), default="chrome")
    args = parser.parse_args()
    output = args.output.resolve()
    status = {
        "started_at": datetime.now(JST).isoformat(), "stage": "認証情報の読み込み",
        "collected": {}, "skipped": {}, "failed": [], "excel_updated": False,
        "csv_only": args.csv_only,
    }
    try:
        user_id, password = read_credentials(args.credentials)
        collect(user_id, password, output, status, args.browser)
    except Exception:
        # ID・パスワード、例外本文、環境変数、ブラウザ画面は記録しない。
        if status["stage"] == "認証情報の読み込み":
            print("認証情報の設定を確認してください（ローカルは.env、クラウドはSecrets）。")
        else:
            print(f"{status['stage']}で停止しました。通信・ログイン方式を確認してください。")
        status["failed"].append(status["stage"])
    # ブラウザの終了などで失敗しても、取得できたCSVはExcelへ反映する。
    if status["collected"] and not args.csv_only:
        try:
            status["stage"] = "Excelの更新"
            command = [sys.executable, str(PROGRAM_DIR / "import_port_csv.py"), "--output", str(output)]
            subprocess.run(command, check=True)
            status["excel_updated"] = True
        except Exception:
            print("Excelの更新に失敗しました。取得済みCSVと前回のExcelは残しています。")
            status["failed"].append("Excelの更新")
    if not status["failed"]:
        status["stage"] = "完了"
    elif status["collected"]:
        status["stage"] = "一部取得失敗"
    for name, reason in status["skipped"].items():
        print(f"見送り: {name}（{reason}）")
    status["finished_at"] = datetime.now(JST).isoformat()
    output.mkdir(parents=True, exist_ok=True)
    pending = output / "last_collection.pending.json"
    pending.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(output / "last_collection.json")
    return 1 if status["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
