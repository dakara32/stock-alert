import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf


# 30-40程度の固定ティッカーリスト
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO",
    "AMD", "NFLX", "CRM", "ORCL", "ADBE", "INTU", "QCOM", "AMAT",
    "MU", "PANW", "ANET", "LRCX", "KLAC", "CDNS", "SNPS", "NOW",
    "UBER", "SHOP", "CRWD", "DDOG", "MDB", "ZS", "NET", "PLTR",
    "TTD", "CELH", "ELF", "ONON", "DUOL", "HIMS"
]

CHART_DIR = Path("charts")
CHART_PERIOD_LABEL = "18mo"


def log(message: str) -> None:
    print(message, flush=True)


def fetch_daily_data(ticker: str) -> pd.DataFrame:
    """
    過去1年半分の日足データを取得
    """
    df = yf.download(
        ticker,
        period="18mo",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"{ticker}: 株価データが取得できませんでした")

    # yfinanceの返却列がMultiIndexになるケース対策
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    required_cols = {"Close", "Low", "High", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: 必須列が不足しています: {missing}")

    df = df.dropna(subset=["Close", "Low", "High", "Volume"]).copy()

    if len(df) < 260:
        raise ValueError(f"{ticker}: データ不足のため判定不可（件数={len(df)}）")

    return df


def evaluate_trend_template(ticker: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    ミネルヴィニのトレンドテンプレート8条件を評価
    """
    close = df["Close"].astype(float)
    low = df["Low"].astype(float)
    high = df["High"].astype(float)
    volume = df["Volume"].astype(float)

    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()
    volume_ma50 = volume.rolling(50).mean()

    current_price = float(close.iloc[-1])
    current_ma50 = float(ma50.iloc[-1])
    current_ma150 = float(ma150.iloc[-1])
    current_ma200 = float(ma200.iloc[-1])
    current_volume = float(volume.iloc[-1])
    current_volume_ma50 = float(volume_ma50.iloc[-1])
    volume_ratio = current_volume / current_volume_ma50 if current_volume_ma50 > 0 else 0.0

    # 52週 = 約252営業日
    low_52w = float(low.tail(252).min())
    low_52w_ratio = current_price / low_52w if low_52w > 0 else 0.0

    # 当日を除く直前50営業日の高値最大値
    high_50d = float(high.iloc[-51:-1].max())

    # 200日MAが20営業日前より上なら「少なくとも1ヶ月上昇トレンド」とみなす
    ma200_20d_ago = float(ma200.iloc[-21])

    conditions = {
        "1_price_gt_150ma": current_price > current_ma150,
        "2_price_gt_200ma": current_price > current_ma200,
        "3_150ma_gt_200ma": current_ma150 > current_ma200,
        "4_200ma_up_1m": current_ma200 > ma200_20d_ago,
        "5_50ma_gt_150ma": current_ma50 > current_ma150,
        "6_50ma_gt_200ma": current_ma50 > current_ma200,
        "7_price_gt_50ma": current_price > current_ma50,
        "8_price_25pct_above_52w_low": current_price >= low_52w * 1.25,
    }

    passed = all(conditions.values())
    volume_passed = passed and volume_ratio >= 1.1
    high_50d_passed = volume_passed and current_price > high_50d
    final_passed = passed and volume_passed and high_50d_passed

    result_text = (
        f"{ticker} | "
        f"Close={current_price:.2f}, "
        f"MA50={current_ma50:.2f}, "
        f"MA150={current_ma150:.2f}, "
        f"MA200={current_ma200:.2f}, "
        f"52WLow={low_52w:.2f}, "
        f"52週安値比={low_52w_ratio:.2f}x "
        f"({(low_52w_ratio - 1) * 100:.1f}%上), "
        f"Volume={volume_ratio:.2f}x, "
        f"50DHigh={high_50d:.2f} | "
        f"判定={'PASS' if passed else 'FAIL'}"
    )

    return {
        "ticker": ticker,
        "passed": passed,
        "volume_passed": volume_passed,
        "high_50d_passed": high_50d_passed,
        "final_passed": final_passed,
        "volume_ratio": volume_ratio,
        "current_price": current_price,
        "high_50d": high_50d,
        "conditions": conditions,
        "result_text": result_text,
    }


def post_to_slack(webhook_url: str, text: str) -> None:
    response = requests.post(
        webhook_url,
        json={"text": text},
        timeout=15,
    )
    response.raise_for_status()


def build_slack_message(
    pass_results: List[Dict[str, Any]],
    volume_pass_results: List[Dict[str, Any]],
    final_pass_results: List[Dict[str, Any]],
) -> str:
    lines = []
    run_date = datetime.now().strftime("%Y-%m-%d")

    lines.append("```")
    lines.append(f"📊 Trend Template + Vol×1.1 + 50D High ｜ {run_date}")
    lines.append("")

    lines.append("【トレンドテンプレート通過銘柄】")
    if pass_results:
        for item in pass_results:
            lines.append(
                f"{item['ticker']}｜${item['current_price']:.2f}｜Vol ×{item['volume_ratio']:.1f}｜50D High ${item['high_50d']:.2f}"
            )
    else:
        lines.append("該当なし")

    lines.append("")
    lines.append("【出来高条件まで通過した銘柄】")
    if volume_pass_results:
        for item in volume_pass_results:
            lines.append(
                f"{item['ticker']}｜${item['current_price']:.2f}｜Vol ×{item['volume_ratio']:.1f}｜50D High ${item['high_50d']:.2f}"
            )
    else:
        lines.append("該当なし")

    lines.append("")
    lines.append("【最終通過銘柄】")
    if final_pass_results:
        for item in final_pass_results:
            lines.append(
                f"{item['ticker']}｜${item['current_price']:.2f}｜Vol ×{item['volume_ratio']:.1f}｜50D High ${item['high_50d']:.2f}"
            )
    else:
        lines.append("該当なし")

    lines.append("```")

    return "\n".join(lines)


def create_price_chart(ticker: str, df: pd.DataFrame) -> Optional[Path]:
    """
    終値のシンプルな折れ線チャートをPNGで保存する。
    失敗した場合はNoneを返し、処理全体は止めない。
    """
    try:
        CHART_DIR.mkdir(parents=True, exist_ok=True)

        close = df["Close"].astype(float).dropna()
        if close.empty:
            log(f"[CHART][SKIP] {ticker}: Closeデータが空です")
            return None

        chart_path = CHART_DIR / f"{ticker}_{CHART_PERIOD_LABEL}.png"

        plt.figure(figsize=(10, 5))
        plt.plot(close.index, close.values)
        plt.title(f"{ticker} Close Price ({CHART_PERIOD_LABEL})")
        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(chart_path, format="png", dpi=150)
        plt.close()

        log(f"画像保存成功: {ticker} path={chart_path}")
        return chart_path

    except Exception as e:
        log(f"[CHART][ERROR] {ticker}: チャート作成に失敗しました: {e}")
        log(traceback.format_exc())
        try:
            plt.close()
        except Exception:
            pass
        return None


def create_price_charts(
    results: List[Dict[str, Any]],
    data_by_ticker: Dict[str, pd.DataFrame],
) -> List[Path]:
    """
    通知対象ティッカーのチャート画像を作成する。
    """
    chart_paths: List[Path] = []

    if not results:
        log("[CHART][SKIP] 通知対象ティッカーがありません")
        return chart_paths

    for item in results:
        ticker = item["ticker"]
        df = data_by_ticker.get(ticker)

        if df is None or df.empty:
            log(f"[CHART][SKIP] {ticker}: 元データがありません")
            continue

        chart_path = create_price_chart(ticker, df)
        if chart_path is not None:
            chart_paths.append(chart_path)

    log(f"[CHART][SUMMARY] 作成枚数={len(chart_paths)}")
    return chart_paths


def slack_api_post(
    bot_token: str,
    endpoint: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Slack Web APIへJSON POSTする。
    chat.postMessage 用。
    """
    response = requests.post(
        f"https://slack.com/api/{endpoint}",
        headers={
            "Authorization": f"Bearer {bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json=payload,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Slack API error: endpoint={endpoint}, response={data}")

    return data


def slack_api_post_form(
    bot_token: str,
    endpoint: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Slack Web APIへ application/x-www-form-urlencoded でPOSTする。
    files.getUploadURLExternal / files.completeUploadExternal 用。
    """
    response = requests.post(
        f"https://slack.com/api/{endpoint}",
        headers={
            "Authorization": f"Bearer {bot_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data=payload,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Slack API error: endpoint={endpoint}, response={data}")

    return data


def post_slack_parent_message(
    bot_token: str,
    channel: str,
    text: str,
) -> Optional[str]:
    """
    画像アップロード用の親メッセージを投稿し、thread_tsを返す。
    """
    try:
        data = slack_api_post(
            bot_token=bot_token,
            endpoint="chat.postMessage",
            payload={
                "channel": channel,
                "text": text,
            },
        )
        thread_ts = data.get("ts")
        if not thread_ts:
            log("[SLACK_IMAGE][ERROR] 親メッセージのtsが取得できませんでした")
            return None

        log(f"Slack親メッセージ投稿成功: thread_ts={thread_ts}")
        return thread_ts

    except Exception as e:
        log(f"[SLACK_IMAGE][ERROR] 親メッセージ投稿に失敗しました: {e}")
        log(traceback.format_exc())
        return None


def get_slack_upload_url(
    bot_token: str,
    file_path: Path,
) -> Optional[Dict[str, str]]:
    """
    files.getUploadURLExternalでupload_urlとfile_idを取得する。
    """
    try:
        if not file_path.exists():
            log(f"[SLACK_IMAGE][SKIP] 画像ファイルが存在しません: {file_path}")
            return None

        file_size = file_path.stat().st_size
        if file_size <= 0:
            log(f"[SLACK_IMAGE][SKIP] 画像ファイルサイズが0です: {file_path}")
            return None

        log(f"[SLACK_IMAGE][UPLOAD_URL_REQUEST] filename={file_path.name}, length={file_size}")

        data = slack_api_post_form(
            bot_token=bot_token,
            endpoint="files.getUploadURLExternal",
            payload={
                "filename": str(file_path.name),
                "length": str(file_size),
            },
        )

        upload_url = data.get("upload_url")
        file_id = data.get("file_id")

        if not upload_url or not file_id:
            log(f"[SLACK_IMAGE][ERROR] upload_urlまたはfile_idが取得できません: {file_path}")
            return None

        log(f"[SLACK_IMAGE][UPLOAD_URL_OK] file_id={file_id}, filename={file_path.name}")

        return {
            "upload_url": upload_url,
            "file_id": file_id,
        }

    except Exception as e:
        log(f"[SLACK_IMAGE][ERROR] upload_url取得に失敗しました: {file_path}: {e}")
        log(traceback.format_exc())
        return None


def upload_file_to_slack_url(upload_url: str, file_path: Path) -> bool:
    """
    Slackから取得したupload_urlへ画像バイナリをPOSTする。
    """
    try:
        with file_path.open("rb") as f:
            response = requests.post(
                upload_url,
                files={
                    "file": (file_path.name, f, "image/png"),
                },
                timeout=60,
            )

        response.raise_for_status()
        log(f"[SLACK_IMAGE][UPLOADED] {file_path}")
        return True

    except Exception as e:
        log(f"[SLACK_IMAGE][ERROR] upload_urlへのPOSTに失敗しました: {file_path}: {e}")
        log(traceback.format_exc())
        return False


def complete_slack_upload(
    bot_token: str,
    channel: str,
    thread_ts: str,
    file_id: str,
    file_path: Path,
) -> bool:
    """
    files.completeUploadExternalでSlack投稿を完了する。
    """
    try:
        log(f"[SLACK_IMAGE][COMPLETE_REQUEST] file_id={file_id}, channel={channel}, thread_ts={thread_ts}")

        slack_api_post_form(
            bot_token=bot_token,
            endpoint="files.completeUploadExternal",
            payload={
                "files": json.dumps([
                    {
                        "id": file_id,
                        "title": file_path.stem,
                    }
                ]),
                "channel_id": channel,
                "thread_ts": thread_ts,
            },
        )
        log(f"[SLACK_IMAGE][COMPLETED] {file_path}")
        return True

    except Exception as e:
        log(f"[SLACK_IMAGE][ERROR] completeUploadExternalに失敗しました: {file_path}: {e}")
        log(traceback.format_exc())
        return False


def upload_chart_images_to_slack_thread(
    bot_token: str,
    channel: str,
    thread_ts: str,
    chart_paths: List[Path],
) -> None:
    """
    作成済みチャート画像をSlackスレッドへ順番にアップロードする。
    1枚失敗しても全体は止めない。
    """
    if not chart_paths:
        log("[SLACK_IMAGE][SKIP] アップロード対象画像がありません")
        return

    success_count = 0

    for chart_path in chart_paths:
        try:
            if not chart_path.exists():
                log(f"[SLACK_IMAGE][SKIP] 画像ファイルが存在しません: {chart_path}")
                continue

            upload_info = get_slack_upload_url(bot_token, chart_path)
            if upload_info is None:
                continue

            uploaded = upload_file_to_slack_url(
                upload_url=upload_info["upload_url"],
                file_path=chart_path,
            )
            if not uploaded:
                continue

            completed = complete_slack_upload(
                bot_token=bot_token,
                channel=channel,
                thread_ts=thread_ts,
                file_id=upload_info["file_id"],
                file_path=chart_path,
            )
            if completed:
                ticker = chart_path.stem.split("_")[0]
                success_count += 1
                log(f"画像アップロード成功: {ticker} path={chart_path}")

        except Exception as e:
            log(f"[SLACK_IMAGE][ERROR] 画像アップロード処理に失敗しました: {chart_path}: {e}")
            log(traceback.format_exc())

    log(f"[SLACK_IMAGE][SUMMARY] 成功={success_count}, 対象={len(chart_paths)}")


def send_chart_images_to_slack(
    chart_paths: List[Path],
    final_pass_results: List[Dict[str, Any]],
) -> None:
    """
    Slackへ画像アップロード用の親メッセージを投稿し、
    そのスレッドへチャート画像を複数枚アップロードする。
    """
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    channel = os.getenv("SLACK_CHANNEL")

    if not bot_token:
        log("[SLACK_IMAGE][SKIP] 環境変数 SLACK_BOT_TOKEN が設定されていません")
        return

    if not channel:
        log("[SLACK_IMAGE][SKIP] 環境変数 SLACK_CHANNEL が設定されていません")
        return

    if not chart_paths:
        log("[SLACK_IMAGE][SKIP] 送信対象のチャート画像がありません")
        return

    run_date = datetime.now().strftime("%Y-%m-%d")
    tickers = [item["ticker"] for item in final_pass_results]
    ticker_text = ", ".join(tickers) if tickers else "該当なし"

    parent_text = (
        f"📈 Trend Template 最終通過銘柄チャート ｜ {run_date}\n"
        f"対象: {ticker_text}"
    )

    thread_ts = post_slack_parent_message(
        bot_token=bot_token,
        channel=channel,
        text=parent_text,
    )
    if not thread_ts:
        return

    upload_chart_images_to_slack_thread(
        bot_token=bot_token,
        channel=channel,
        thread_ts=thread_ts,
        chart_paths=chart_paths,
    )


def main() -> None:
    log("開始: ミネルヴィニ条件スクリーニング")

    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        log("エラー: 環境変数 SLACK_WEBHOOK_URL が設定されていません")
        return

    pass_results: List[Dict[str, Any]] = []
    volume_pass_results: List[Dict[str, Any]] = []
    final_pass_results: List[Dict[str, Any]] = []
    data_by_ticker: Dict[str, pd.DataFrame] = {}

    for ticker in TICKERS:
        try:
            log(f"取得中: {ticker}")
            df = fetch_daily_data(ticker)
            result = evaluate_trend_template(ticker, df)

            log(result["result_text"])

            if result["passed"]:
                pass_results.append(result)

            if result["volume_passed"]:
                volume_pass_results.append(result)

            if result["final_passed"]:
                final_pass_results.append(result)
                data_by_ticker[ticker] = df

        except Exception as e:
            log(f"エラー: {ticker} の処理に失敗しました: {e}")
            log(traceback.format_exc())

    message = build_slack_message(pass_results, volume_pass_results, final_pass_results)

    try:
        post_to_slack(webhook_url, message)
        log("Slack送信完了")
    except Exception as e:
        log(f"エラー: Slack送信に失敗しました: {e}")
        log(traceback.format_exc())

    try:
        chart_paths = create_price_charts(final_pass_results, data_by_ticker)
        send_chart_images_to_slack(chart_paths, final_pass_results)
    except Exception as e:
        log(f"[CHART][ERROR] チャート画像送信まわりの処理に失敗しました: {e}")
        log(traceback.format_exc())


if __name__ == "__main__":
    main()