"""收盘后马上补上当天的日线：腾讯行情 qt.gtimg.cn（实时，GitHub Actions 上能访问）。

baostock 的当天日线要到北京时间 17:30–18:00 以后才有。15:00 收盘后用腾讯的实时报价拼出当天这一根：
开、高、低、收、昨收、成交量、成交额、换手率；估值（市盈率、市净率、市销率）按昨天的值乘以收盘价变化，
当天没有新财报时这和真实的 TTM 估值一致。第二天 baostock 出正式数据后，update_kline 的重叠回补会覆盖这一行。
"""

import logging
import re
from datetime import datetime

import numpy as np
import pandas as pd

from . import data

log = logging.getLogger(__name__)

URL = "http://qt.gtimg.cn/q="
BATCH = 60
UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}


def _parse(text):
    out = {}
    for sym, body in re.findall(r'v_(\w+)="([^"]*)"', text):
        f = body.split("~")
        if len(f) > 48:
            out[sym] = f
    return out


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def fetch_quotes(codes, session=None, pause=0.05):
    """codes 用 baostock 格式（sh.600000）。返回 DataFrame：code, time, open, high, low, close, preclose, volume, amount, turn。"""
    import time

    import requests

    session = session or requests.Session()
    rows, failed = [], []
    syms = [c.replace(".", "") for c in codes]
    for i in range(0, len(syms), BATCH):
        part = syms[i:i + BATCH]
        try:
            r = session.get(URL + ",".join(part), headers=UA, timeout=15)
            r.raise_for_status()
            got = _parse(r.content.decode("gbk", errors="replace"))
        except Exception as exc:
            log.warning("tencent batch %d failed: %s", i, exc)
            failed += part
            continue
        for s in part:
            f = got.get(s)
            if f is None:
                failed.append(s)
                continue
            try:
                t = datetime.strptime(f[30], "%Y%m%d%H%M%S")
            except ValueError:
                failed.append(s)
                continue
            rows.append(dict(code=s[:2] + "." + s[2:], time=t, close=_num(f[3]), preclose=_num(f[4]), open=_num(f[5]),
                             high=_num(f[33]), low=_num(f[34]), volume=_num(f[36]) * 100, amount=_num(f[37]) * 1e4,
                             turn=_num(f[38])))
        time.sleep(pause)
    return pd.DataFrame(rows), failed


def today_bars(codes, day, kline, valuation, session=None):
    """拼出交易日 day 的日线和估值行。只收报价时间是 day 15:00 以后的（已收盘）。"""
    q, failed = fetch_quotes(codes, session, pause=0 if session is not None else 0.05)
    if q.empty:
        return None, None, failed
    day = pd.Timestamp(day)
    q = q[(q["time"].dt.normalize() == day) & (q["time"].dt.hour >= 15)].copy()
    if q.empty:
        return None, None, failed
    q["date"] = day
    traded = (q["volume"] > 0) & (q["open"] > 0)
    q["tradestatus"] = traded.astype(int)
    # 停牌：价格沿用昨收
    for c in ("open", "high", "low", "close"):
        q.loc[~traded, c] = q.loc[~traded, "preclose"]
    last = kline.sort_values("date").groupby("code").tail(1).set_index("code")
    q["isST"] = q["code"].map(last["isST"]).fillna(0) if "isST" in last else 0
    bars = data.clean_kline(q[data.KLINE_FIELDS.split(",")].reset_index(drop=True))

    val = None
    if valuation is not None and len(valuation):
        lv = valuation.sort_values("date").groupby("code").tail(1).set_index("code")
        lc = last["close"]
        ratio = (q.set_index("code")["close"] / lc.reindex(q["code"]).values).replace([np.inf, -np.inf], np.nan)
        val = pd.DataFrame(dict(date=day, code=q["code"].values))
        for c in data.VALUATION_NUMERIC:
            val[c] = (lv[c].reindex(q["code"]).values * ratio.values) if c in lv else np.nan
        val = data.clean_valuation(val)
    return bars, val, failed


def check(bars, kline):
    """字段对不上时宁可不补：腾讯的昨收要和缓存里上一天的收盘价一致，最高、最低要包住开盘和收盘。"""
    last = kline.sort_values("date").groupby("code").tail(1).set_index("code")["close"]
    pc = bars.set_index("code")["preclose"]
    lc = last.reindex(pc.index)
    ok = ((pc - lc).abs() / lc < 0.002)[lc.notna()]
    t = bars[bars["tradestatus"] == 1]
    sane = (t["low"] <= t[["open", "close"]].min(axis=1) + 1e-6) & (t["high"] >= t[["open", "close"]].max(axis=1) - 1e-6)
    if ok.mean() < 0.9 or (len(t) and sane.mean() < 0.99):
        raise RuntimeError(f"tencent quotes look wrong: preclose matches {ok.mean():.0%}, high/low sane {sane.mean():.0%}")


def fill_today(codes, day, kline, valuation, session=None):
    """把当天收盘拼出的日线和估值并进缓存；返回新的 (kline, valuation)。"""
    bars, val, failed = today_bars(codes, day, kline, valuation, session)
    if bars is None or len(bars) < 0.9 * len(codes):
        raise RuntimeError(f"tencent quotes for {pd.Timestamp(day).date()}: got {0 if bars is None else len(bars)} "
                           f"of {len(codes)} ({len(failed)} failed)")
    check(bars, kline)
    log.info("tencent: %d bars for %s (%d failed)", len(bars), pd.Timestamp(day).date(), len(failed))
    kline = data._merge(kline, [bars], "kline")
    if val is not None:
        valuation = data._merge(valuation, [val], "valuation")
    return kline, valuation
