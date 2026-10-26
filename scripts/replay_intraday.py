"""盘中估算回放：用 Yahoo 最近约 60 天的 15 分钟行情，按盘中任务的同一套逻辑逐个时点重算各区域分数。

回答三个问题：收盘时的最后一个估算和正式读数差多少；盘中读数一天里摆动多大；
“盘中急跌”提醒（比上一交易日低 6 分以上）有多少在收盘时没有成立。
A股回放只有沪深300 有 15 分钟行情，成交额和 QVIX 沿用前一天，所以 A股的差距会比实盘大。
结果写入 reports/intraday_replay.md。用法（需能访问 Yahoo，在 GitHub Actions 里跑）：python scripts/replay_intraday.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fearindex import compute, config, fetch, intraday  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reports" / "intraday_replay.md"
SYMBOLS = {**intraday.YAHOO, "csi300": "000300.SS"}
HIST_ROWS = 1700  # 滚动分位 1260 天 + 回撤 252 天 + 余量，截断后结果与全量一致


def bars():
    import yfinance as yf

    out = {}
    for key, sym in SYMBOLS.items():
        df = yf.Ticker(sym).history(period="60d", interval="15m", prepost=False, auto_adjust=False)
        if df is None or df.empty:
            print("no bars", key, sym)
            continue
        close = df["Close"].dropna()
        # 15 分钟线的时间戳是这根线的开始，结束时刻才知道收盘价
        out[key] = pd.DataFrame({"value": close.values,
                                 "time": close.index.tz_convert("UTC") + pd.Timedelta(minutes=15),
                                 "date": [pd.Timestamp(t.date()) for t in close.index]})
    return out


def replay(series, official, b):
    rows = []
    for region in config.REGIONS:
        keys = {k for c in config.COMPONENTS.values() if c["region"] == region for k in c["inputs"]}
        prim = intraday.PRIMARY[region]
        if prim not in b:
            continue
        days = sorted(set(b[prim]["date"]) & set(official.index))[1:]
        for day in days:
            base = {k: s[s.index < day].iloc[-HIST_ROWS:] for k, s in series.items() if k in keys}
            prev = compute.build(base)[0][region].dropna()
            if prev.empty:
                continue
            times = b[prim].loc[b[prim]["date"] == day, "time"]
            for t in times:
                quotes = {}
                for k in keys & set(b):
                    sub = b[k][(b[k]["date"] == day) & (b[k]["time"] <= t)]
                    if len(sub):
                        last = sub.iloc[-1]
                        quotes[k] = dict(value=float(last["value"]), time=last["time"], date=day)
                live_series, fresh = intraday.apply_quotes(base, quotes)
                if prim not in fresh:
                    continue
                live = compute.build(live_series)[0][region].dropna()
                rows.append(dict(region=region, day=day, time=t, score=float(live.iloc[-1]),
                                 prev=float(prev.iloc[-1]), official=float(official.loc[day, region])))
        print(region, len(rows))
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for region, g in df.groupby("region", sort=False):
        days = g.groupby("day")
        last = days.last()
        gap = (last["score"] - last["official"]).abs()
        swing = days["score"].agg(lambda s: s.max() - s.min())
        lo = days["score"].min()
        alert = lo <= last["prev"] - intraday.DROP_ALERT
        confirmed = last["official"] <= last["prev"] - intraday.DROP_ALERT
        out.append({"区域": config.REGIONS[region], "天数": len(last),
                    "收盘估算与正式读数差（中位/最大）": f"{gap.median():.1f} / {gap.max():.1f}",
                    "一天内摆动（中位/最大）": f"{swing.median():.1f} / {swing.max():.1f}",
                    "盘中急跌提醒天数": int(alert.sum()),
                    "其中收盘也成立": int((alert & confirmed).sum()),
                    "收盘成立但盘中没提醒": int((~alert & confirmed).sum())})
    return pd.DataFrame(out)


def main():
    series = fetch.load_all()
    official = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "index.csv",
                           index_col="date", parse_dates=["date"])
    b = bars()
    df = replay(series, official, b)
    s = summarize(df)
    first, last = df["day"].min().date(), df["day"].max().date()
    lines = ["# 盘中估算回放", "",
             f"用 Yahoo {first} 至 {last} 的 15 分钟行情，按盘中任务的逻辑在每根 15 分钟线收盘时重算一次（共 {len(df)} 个时点）。",
             "A股只有沪深300 有 15 分钟行情，成交额和 QVIX 用前一天的值，所以 A股的收盘差距比实盘大。", "",
             "- 收盘估算与正式读数差：当天最后一个盘中估算和第二天早上发布的正式读数相差多少分。",
             "- 一天内摆动：盘中估算当天的最高减最低。",
             f"- 盘中急跌提醒：盘中某个时点比上一交易日低 {intraday.DROP_ALERT} 分以上；“收盘也成立”指正式读数也低了这么多。", ""]
    lines.append("| " + " | ".join(s.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(s.columns)) + " |")
    for _, r in s.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    df.to_csv(OUT.with_suffix(".csv"), index=False, float_format="%.2f")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
