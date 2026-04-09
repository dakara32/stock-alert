import os
import traceback
from typing import List, Dict, Any

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

    required_cols = {"Close", "Low"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: 必須列が不足しています: {missing}")

    df = df.dropna(subset=["Close", "Low"]).copy()

    if len(df) < 260:
        raise ValueError(f"{ticker}: データ不足のため判定不可（件数={len(df)}）")

    return df


def evaluate_trend_template(ticker: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    ミネルヴィニのトレンドテンプレート8条件を評価
    """
    close = df["Close"].astype(float)
    low = df["Low"].astype(float)

    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    current_price = float(close.iloc[-1])
    current_ma50 = float(ma50.iloc[-1])
    current_ma150 = float(ma150.iloc[-1])
    current_ma200 = float(ma200.iloc[-1])

    # 52週 = 約252営業日
    low_52w = float(low.tail(252).min())
    low_52w_ratio = current_price / low_52w if low_52w > 0 else 0.0

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

    result_text = (
        f"{ticker} | "
        f"Close={current_price:.2f}, "
        f"MA50={current_ma50:.2f}, "
        f"MA150={current_ma150:.2f}, "
        f"MA200={current_ma200:.2f}, "
        f"52WLow={low_52w:.2f}, "
        f"52週安値比={low_52w_ratio:.2f}x "
        f"({(low_52w_ratio - 1) * 100:.1f}%上) | "
        f"判定={'PASS' if passed else 'FAIL'}"
    )

    return {
        "ticker": ticker,
        "passed": passed,
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


def build_slack_message(pass_results: List[Dict[str, Any]]) -> str:
    if not pass_results:
        return "ミネルヴィニ・トレンドテンプレート合致銘柄: 該当なし"

    lines = ["ミネルヴィニ・トレンドテンプレート合致銘柄"]
    for item in pass_results:
        lines.append(item["result_text"])
    return "\n".join(lines)


def main() -> None:
    log("開始: ミネルヴィニ条件スクリーニング")

    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        log("エラー: 環境変数 SLACK_WEBHOOK_URL が設定されていません")
        return

    pass_results: List[Dict[str, Any]] = []

    for ticker in TICKERS:
        try:
            log(f"取得中: {ticker}")
            df = fetch_daily_data(ticker)
            result = evaluate_trend_template(ticker, df)

            log(result["result_text"])

            if result["passed"]:
                pass_results.append(result)

        except Exception as e:
            log(f"エラー: {ticker} の処理に失敗しました: {e}")
            log(traceback.format_exc())

    message = build_slack_message(pass_results)

    try:
        post_to_slack(webhook_url, message)
        log("Slack送信完了")
    except Exception as e:
        log(f"エラー: Slack送信に失敗しました: {e}")
        log(traceback.format_exc())


if __name__ == "__main__":
    main()