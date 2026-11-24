"""场内 ETF 日线：不复权 + 后复权（腾讯行情），自己算含分红的真实收益。

腾讯的前复权是“减去分红”的加法复权，早年价格被压得很低，日涨跌会被放大（2015 年沪深300ETF 显示 −12.8%，
指数只跌 7.7%）。所以同时拉不复权和后复权：两者之差的跳变就是当天每份分红，
含分红收益 = (不复权收盘 + 当天分红) / 前一天不复权收盘 − 1；拆分（不复权价格突然缩小、后复权连续）也能识别。
写 data/research/etf/<code>.csv（date, open, close, high, low, volume, div, ret_close, ret_open）。
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from astock_diversify_dump import ETFS, UA, _Session  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "research" / "etf"
log = logging.getLogger("etf")


def kline(session, sym, fq, start=2012):
    rows = []
    for y in range(start, pd.Timestamp.today().year + 1, 2):
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,{y}-01-01,{y + 1}-12-31,640,{fq}"
        body = session.get(url, headers=UA, timeout=20).json()["data"][sym]
        rows += [x[:6] for x in (body.get(fq + "day") or body.get("day") or [])]
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"]).drop_duplicates("date")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").set_index("date")


def total_return(raw, hfq):
    """raw 不复权、hfq 后复权（同日期）。返回 DataFrame，含 div（每份分红）、ret_close、ret_open（开盘到开盘）。"""
    df = raw.join(hfq[["open", "close"]].rename(columns={"open": "h_open", "close": "h_close"}), how="inner")
    rc = df["close"] / df["close"].shift(1)
    hc = df["h_close"] / df["h_close"].shift(1)
    d = df["h_close"] - df["close"]  # 加法后复权：= 累计分红（不变就是没分红）
    div = (d - d.shift(1)).clip(lower=0).fillna(0)
    div[div < 1e-4] = 0
    ret = (df["close"] + div) / df["close"].shift(1) - 1
    # 拆分/折算：不复权价格跳变很大、后复权正常，就用后复权的比值
    split = (abs(rc - 1) > 0.25) & (abs(hc - 1) < 0.2)
    ret[split] = hc[split] - 1
    ro = (df["open"] + div) / df["open"].shift(1) - 1
    ho = df["h_open"] / df["h_open"].shift(1)
    ro[split] = ho[split] - 1
    df["div"], df["ret_close"], df["ret_open"] = div, ret, ro
    df["split"] = split
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = _Session()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    for sym, name, cat in ETFS:
        try:
            raw, hfq = kline(s, sym, ""), kline(s, sym, "hfq")
        except Exception as exc:
            log.warning("%s failed: %s", sym, exc)
            continue
        df = total_return(raw, hfq)
        df.to_csv(OUT / f"{sym}.csv", float_format="%.6g")
        ann = float((1 + df["ret_close"].fillna(0)).prod() ** (244 / max(len(df), 1)) - 1)
        meta.append(dict(code=sym, name=name, cat=cat, first=df.index[0], last=df.index[-1], n=len(df),
                         n_div=int((df["div"] > 0).sum()), n_split=int(df["split"].sum()), ann=round(ann, 4)))
        log.info("%s", meta[-1])
    (OUT / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
