"""A股每日开盘前选股：更新最近行情和业绩预告，写 docs/astock/picks.json 并跟踪历史清单。

有 astock/forecast_model.json 时用“业绩预期 + 涨停情绪开关”的 20 只持仓组合（2026-10-28 起）；
没有时退回 astock/model.json 的单日清单。
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import alt_study, data, screen  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--trade-date", help="默认取今天之后（含今天）的第一个交易日")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    fm = json.loads(alt_study.LIVE_FILE.read_text()) if alt_study.LIVE_FILE.exists() else None
    log_file = screen.PORT_LOG if fm else screen.LOG_FILE
    log_df = pd.read_csv(log_file, dtype={"code": str}) if log_file.exists() else None

    if args.offline:
        kline, basic = data.load_kline(), data.load_basic()
        trade_date = pd.Timestamp(args.trade_date) if args.trade_date else kline["date"].max() + pd.offsets.BDay()
    else:
        with data.Session() as sess:
            days = sess.trade_dates(today.strftime("%Y-%m-%d"), (today + pd.Timedelta(days=20)).strftime("%Y-%m-%d"))
            trade_date = pd.Timestamp(args.trade_date) if args.trade_date else days[0]
            prev = sess.trade_dates((trade_date - pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
                                    (trade_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))[-1]
            if not args.trade_date and trade_date != today:
                logging.info("today %s is not a trading day, skip", today.date())
                return
            basic = data.update_basic(sess)
        codes = data.stock_codes(basic, listed_only=True)
        tracked_codes = set(log_df["code"]) if log_df is not None else set()
        start = (today - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
        kline, failed = data.update_kline(sorted(set(codes) | tracked_codes), start=start)
        data.update_kline(codes, start=start, kind="valuation")
        if fm:
            fc_start = (today - pd.Timedelta(days=screen.PANEL_DAYS + 60)).strftime("%Y-%m-%d")
            _, fc_failed = data.update_forecast(codes, start=fc_start, merge=True)
            if len(fc_failed) > 0.1 * len(codes):
                raise RuntimeError(f"too many failed forecast codes: {len(fc_failed)}")
        # 收盘后手动运行时最新一根 K 线就是今天，清单应给下一个交易日
        if kline["date"].max() >= trade_date and not args.trade_date:
            trade_date = days[1]
        if len(failed) > 0.1 * len(codes):
            raise RuntimeError(f"too many failed codes: {len(failed)}")
        if kline["date"].max() < prev:
            raise RuntimeError(f"baostock has no bars for {prev.date()} yet")
        if os.environ.get("GITHUB_ACTIONS"):
            # 拉行情的这段时间里恐慌指数（07:30 的任务）多半已经更新，选股前取最新的 data/index.csv
            subprocess.run(["git", "pull", "--rebase", "--autostash"], check=False)
    last = kline["date"].max()
    if (trade_date - last).days > 7:
        raise RuntimeError(f"stale data: last bar {last.date()} for trade date {trade_date.date()}")
    if fm:
        forecast = data.load_forecast()
        if forecast is None:
            # 分支上 --offline 检查流程时缓存里可能还没有业绩预告：补拉一次（Actions 上能连 baostock）
            forecast, _ = data.update_forecast(data.stock_codes(basic, listed_only=True),
                                               start=(kline["date"].max() - pd.Timedelta(days=screen.PANEL_DAYS + 60)).strftime("%Y-%m-%d"))
        result, state = screen.screen_forecast(kline, basic, trade_date, fm, valuation=data.load_valuation(),
                                               forecast=forecast)
        screen.save_state(state)
        log_df = screen.append_port_log(result)
        rows, days_done = screen.track_portfolio(kline, log_df, fm["n"])
        screen.write(result, rows, days_done)
    else:
        result = screen.screen(kline, basic, trade_date, valuation=data.load_valuation())
        log_df = screen.append_log(result)
        screen.write(result, screen.track(kline, log_df))
    print({k: v for k, v in result.items() if k != "ranking"})


if __name__ == "__main__":
    main()
