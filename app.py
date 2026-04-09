import os
import sys
import traceback
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf


# 固定ティッカーリスト（30-40銘柄程度）
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSM",
    "LLY", "NFLX", "COST", "AMD", "ADBE", "CRM", "ORCL", "NOW",
    "UBER", "PANW", "ANET", "INTU", "AMAT", "LRCX", "KLAC", "MU",
    "ASML", "MELI", "SHOP", "VRTX", "ISRG", "BKNG", "CDNS", "SNPS",
    "CRWD", "PLTR", "GE", "JPM", "V", "MA", "GS", "AXON"
]


def log(message: str) -> None:
    print(message, flush=True)


def fetch_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    try:
        # 約1年半分の日足データを取得
        df = yf.download(
            ticker,
            period="18mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            log(f"[WARN] {ticker}: データ取得結果が空です")
            return None

        # yfinance の戻り値が MultiIndex になるケースを吸収
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        missing = required_cols - set(df.columns)
        if missing:
            log(f"[WARN] {ticker}: 必須列が不足しています missing={sorted(missing)}")
            return None

        df = df.dropna(subset=["Close", "Low", "Volume"]).copy()
        if df.empty:
            log(f"[WARN] {ticker}: 有効な終値または出来高データがありません")
            return None

        return df

    except Exception as e:
        log(f"[ERROR] {ticker}: データ取得中に例外が発生しました: {e}")
        log(traceback.format_exc())
        return None


def evaluate_trend_template(ticker: str, df: pd.DataFrame) -> Optional[Dict]:
    try:
        if len(df) < 252:
            log(f"[WARN] {ticker}: 52週判定に必要なデータが不足しています len={len(df)}")
            return None

        df = df.copy()
        df["ma50"] = df["Close"].rolling(window=50).mean()
        df["ma150"] = df["Close"].rolling(window=150).mean()
        df["ma200"] = df["Close"].rolling(window=200).mean()
        df["avg_volume_50"] = df["Volume"].rolling(window=50).mean()

        latest = df.iloc[-1]

        # 200MAの20営業日前比較が必要
        if len(df) < 220:
            log(f"[WARN] {ticker}: 200MA上昇トレンド判定に必要なデータが不足しています len={len(df)}")
            return None

        ma200_20d_ago = df["ma200"].iloc[-21]

        close_price = latest["Close"]
        ma50 = latest["ma50"]
        ma150 = latest["ma150"]
        ma200 = latest["ma200"]
        latest_volume = latest["Volume"]
        avg_volume_50 = latest["avg_volume_50"]

        if (
            pd.isna(ma50)
            or pd.isna(ma150)
            or pd.isna(ma200)
            or pd.isna(ma200_20d_ago)
            or pd.isna(avg_volume_50)
        ):
            log(f"[WARN] {ticker}: 移動平均または平均出来高の計算に必要なデータが不足しています")
            return None

        if avg_volume_50 <= 0:
            log(f"[WARN] {ticker}: 50日平均出来高が不正です avg_volume_50={avg_volume_50}")
            return None

        low_52w = df["Low"].tail(252).min()
        if pd.isna(low_52w) or low_52w <= 0:
            log(f"[WARN] {ticker}: 52週安値の計算結果が不正です low_52w={low_52w}")
            return None

        pct_from_52w_low = ((close_price / low_52w) - 1.0) * 100.0
        volume_ratio = float(latest_volume) / float(avg_volume_50)

        conditions = {
            "1_close_gt_ma150": close_price > ma150,
            "2_close_gt_ma200": close_price > ma200,
            "3_ma150_gt_ma200": ma150 > ma200,
            "4_ma200_up_20d": ma200 > ma200_20d_ago,
            "5_ma50_gt_ma150": ma50 > ma150,
            "6_ma50_gt_ma200": ma50 > ma200,
            "7_close_gt_ma50": close_price > ma50,
            "8_close_25pct_above_52w_low": pct_from_52w_low >= 25.0,
        }

        trend_passed = all(conditions.values())
        volume_passed = volume_ratio >= 1.1
        passed = trend_passed and volume_passed

        return {
            "ticker": ticker,
            "close": float(close_price),
            "ma50": float(ma50),
            "ma150": float(ma150),
            "ma200": float(ma200),
            "ma200_20d_ago": float(ma200_20d_ago),
            "low_52w": float(low_52w),
            "pct_from_52w_low": float(pct_from_52w_low),
            "latest_volume": float(latest_volume),
            "avg_volume_50": float(avg_volume_50),
            "volume_ratio": float(volume_ratio),
            "conditions": conditions,
            "trend_passed": trend_passed,
            "volume_passed": volume_passed,
            "passed": passed,
        }

    except Exception as e:
        log(f"[ERROR] {ticker}: 判定中に例外が発生しました: {e}")
        log(traceback.format_exc())
        return None


def format_result_line(result: Dict) -> str:
    return (
        f"{result['ticker']}: "
        f"Close={result['close']:.2f}, "
        f"Volume x{result['volume_ratio']:.2f}"
    )


def build_slack_message(passed_results: List[Dict], all_results: List[Dict]) -> str:
    header = f"ミネルヴィニ・トレンドテンプレート+出来高条件 合致銘柄: {len(passed_results)} / {len(all_results)}"

    if not passed_results:
        return header + "\n該当なし"

    body_lines = [format_result_line(r) for r in passed_results]
    return header + "\n" + "\n".join(body_lines)


def post_to_slack(webhook_url: str, text: str) -> None:
    try:
        response = requests.post(
            webhook_url,
            json={"text": text},
            timeout=15,
        )
        if response.status_code != 200:
            log(f"[ERROR] Slack送信失敗 status={response.status_code} body={response.text}")
            raise RuntimeError(f"Slack送信失敗 status={response.status_code}")

        log("[INFO] Slack送信成功")

    except Exception as e:
        log(f"[ERROR] Slack送信中に例外が発生しました: {e}")
        log(traceback.format_exc())
        raise


def main() -> int:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        log("[ERROR] 環境変数 SLACK_WEBHOOK_URL が設定されていません")
        return 1

    log("[INFO] スクリーニング開始")

    all_results: List[Dict] = []
    passed_results: List[Dict] = []

    for ticker in TICKERS:
        log(f"[INFO] 処理中: {ticker}")

        df = fetch_ohlcv(ticker)
        if df is None:
            continue

        result = evaluate_trend_template(ticker, df)
        if result is None:
            continue

        all_results.append(result)

        if result["passed"]:
            passed_results.append(result)
            log(f"[PASS] {format_result_line(result)}")
        else:
            failed_conditions = [k for k, v in result["conditions"].items() if not v]
            if not result["volume_passed"]:
                failed_conditions.append("9_volume_gte_1.1x_avg50")
            log(f"[FAIL] {ticker}: 未達条件={failed_conditions}")

    # 出来高倍率が高い順で見やすく並べる
    passed_results.sort(key=lambda x: x["volume_ratio"], reverse=True)

    message = build_slack_message(passed_results, all_results)
    log("[INFO] Slack送信用メッセージ")
    log(message)

    try:
        post_to_slack(webhook_url, message)
    except Exception:
        return 1

    log("[INFO] スクリーニング完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())