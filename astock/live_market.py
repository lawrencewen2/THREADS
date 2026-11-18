"""盘中市场温度：用腾讯实时行情算当天的涨跌家数、涨跌停、成交额、站上 20 日线比例和温度分。

收盘后的正式温度在 astock/insight.py 里用日线算；这里用同一套公式（temperature_from_metrics），
只是把“今天”这一行换成实时行情。需要 insight 每天写出的 docs/astock/market_ref.json：
- codes: {code: [前 19 个交易日收盘均价, 涨跌停幅度]}，今天价格高于前 19 日均价就等于站上 20 日线
- hist: 最近约 270 个交易日的每日指标，用来算分位

只依赖 numpy / pandas / requests，盘中工作流不用装 baostock。
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CN_TZ = ZoneInfo("Asia/Shanghai")
TEMP_RANK_WINDOW = 250  # 温度分：各指标在近 250 个交易日里的分位
HIST_DAYS = 270  # market_ref.json 里存的历史天数（250 天分位 + 5 日均线）
URL = "http://qt.gtimg.cn/q="
UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}
BATCH = 80
WORKERS = 8
SESSION_MINUTES = 240


def temperature_from_metrics(df):
    """df 每行一天，含 up、down、limit_up、limit_down、amount（亿元）、above_ma20。加上派生列和温度分（0 冷 100 热）。"""
    df = df.copy()
    df["up_share"] = df["up"] / (df["up"] + df["down"]).replace(0, np.nan)
    df["net_limit"] = df["limit_up"] - df["limit_down"]
    df["amount_ratio"] = df["amount"] / df["amount"].rolling(20, min_periods=10).mean()

    def pr(s):
        return s.rolling(TEMP_RANK_WINDOW, min_periods=60).rank(pct=True)

    parts = [pr(df["up_share"].rolling(5, min_periods=3).mean()), pr(df["above_ma20"]),
             pr(df["net_limit"].rolling(5, min_periods=3).mean()), pr(df["amount_ratio"])]
    df["temperature"] = pd.concat(parts, axis=1).mean(axis=1) * 100
    return df


def temperature_label(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "未知"
    for upper, label in ((20, "冰点"), (40, "偏冷"), (60, "正常"), (80, "偏热"), (101, "过热")):
        if v < upper:
            return label
    return "过热"


def minutes_traded(now_cn):
    """A股当天已经交易的分钟数（9:30–11:30、13:00–15:00），用来把成交额折算成全天。"""
    t = now_cn.hour * 60 + now_cn.minute + now_cn.second / 60
    am = min(max(t - (9 * 60 + 30), 0), 120)
    pm = min(max(t - 13 * 60, 0), 120)
    return am + pm


def _fetch_batch(session, part):
    try:
        r = session.get(URL + ",".join(part), headers=UA, timeout=15)
        r.raise_for_status()
    except Exception as exc:  # 单批失败不影响其他批
        log.warning("tencent batch failed: %s", exc)
        return {}
    text = r.content.decode("gbk", errors="replace")
    out = {}
    for sym, body in re.findall(r'v_(\w+)="([^"]*)"', text):
        f = body.split("~")
        if len(f) > 37:
            out[sym] = f
    return out


def fetch_quotes(codes, session=None):
    """codes 用 baostock 格式（sh.600000）。返回 DataFrame：code, time, close, preclose, volume, amount（元）。"""
    import requests

    session = session or requests.Session()
    syms = [c.replace(".", "") for c in codes]
    parts = [syms[i:i + BATCH] for i in range(0, len(syms), BATCH)]
    raw = {}
    with ThreadPoolExecutor(WORKERS) as ex:
        for got in ex.map(lambda p: _fetch_batch(session, p), parts):
            raw.update(got)
    rows = []
    for s, f in raw.items():
        try:
            t = datetime.strptime(f[30], "%Y%m%d%H%M%S")
            rows.append(dict(code=s[:2] + "." + s[2:], time=t, close=float(f[3] or 0), preclose=float(f[4] or 0),
                             volume=float(f[6] or 0), amount=float(f[37] or 0) * 1e4, open=float(f[5] or 0),
                             high=float(f[33] or 0), low=float(f[34] or 0)))
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows, columns=["code", "time", "close", "preclose", "volume", "amount", "open", "high", "low"])


def today_metrics(ref, quotes, day):
    """用当天（day）的实时报价算一行指标。只数报价日期是 day、有成交的股票。"""
    codes = ref["codes"]
    q = quotes[quotes["code"].isin(codes.keys())].copy()
    q = q[(q["time"].dt.date == day) & (q["volume"] > 0) & (q["close"] > 0) & (q["preclose"] > 0)]
    if q.empty:
        return None
    mean19 = q["code"].map(lambda c: codes[c][0])
    pct = q["code"].map(lambda c: codes[c][1])
    ret = q["close"] / q["preclose"] - 1
    up_px = (q["preclose"] * (1 + pct) + 1e-6).round(2)
    dn_px = (q["preclose"] * (1 - pct) + 1e-6).round(2)
    has_ma = mean19.notna()
    return dict(
        up=int((ret > 0).sum()),
        down=int((ret < 0).sum()),
        flat=int((ret == 0).sum()),
        limit_up=int((q["close"] >= up_px - 0.005).sum()),
        limit_down=int((q["close"] <= dn_px + 0.005).sum()),
        amount=float(q["amount"].sum() / 1e8),
        above_ma20=float((q["close"][has_ma] > mean19[has_ma]).sum() / max(int(has_ma.sum()), 1)),
        median_ret=float(ret.median()),
        stocks=int(len(q)),
        last_quote=q["time"].max().strftime("%H:%M:%S"),
    )


def build(ref, quotes, now_cn):
    """返回 market.json 的内容，没有当天行情时返回 None。"""
    day = now_cn.date()
    if ref.get("signal_date") and ref["signal_date"] >= str(day):
        return None
    m = today_metrics(ref, quotes, day)
    if m is None:
        return None
    mins = minutes_traded(now_cn)
    closed = now_cn.time() >= time(15, 0)
    # 成交额按已交易分钟折算成全天，再和过去 20 天比。集合竞价和开盘 15 分钟内折算误差太大，
    # 这时先用昨天的成交额代替（量比按昨天算），免得温度被一个很小的成交额拉低
    hist = pd.DataFrame(ref["hist"])
    early = not closed and mins < 15
    scale = 1.0 if closed or early else SESSION_MINUTES / mins
    full_amount = float(hist["amount"].iloc[-1]) if early and len(hist) else m["amount"] * scale
    today = {k: m[k] for k in ("up", "down", "limit_up", "limit_down", "above_ma20")}
    today.update(amount=full_amount, date=str(day))
    df = temperature_from_metrics(pd.concat([hist, pd.DataFrame([today])], ignore_index=True))
    last, prev = df.iloc[-1], df.iloc[-2]
    temp = None if np.isnan(last["temperature"]) else round(float(last["temperature"]), 2)
    prev_temp = None if np.isnan(prev["temperature"]) else round(float(prev["temperature"]), 2)
    return dict(
        date=str(day),
        time=now_cn.strftime("%m-%d %H:%M"),
        session="已收盘" if closed else ("午休" if time(11, 30) <= now_cn.time() < time(13, 0) else "盘中"),
        temperature=temp,
        label=temperature_label(temp),
        prev_temperature=prev_temp,
        change=None if temp is None or prev_temp is None else round(temp - prev_temp, 2),
        up=m["up"], down=m["down"], flat=m["flat"], limit_up=m["limit_up"], limit_down=m["limit_down"],
        net_limit=m["limit_up"] - m["limit_down"],
        up_share=round(m["up"] / max(m["up"] + m["down"], 1), 4),
        amount=round(m["amount"], 1),
        amount_full_day=None if early else round(m["amount"] * scale, 1),
        amount_ratio=None if np.isnan(last["amount_ratio"]) else round(float(last["amount_ratio"]), 4),
        above_ma20=round(m["above_ma20"], 4),
        median_ret=round(m["median_ret"], 4),
        stocks=m["stocks"],
        last_quote=m["last_quote"],
        signal_date=ref.get("signal_date"),
    )


def run(ref, now_cn=None, session=None, quotes=None):
    now_cn = now_cn or datetime.now(CN_TZ)
    if now_cn.weekday() >= 5 or now_cn.time() < time(9, 25):
        return None
    if quotes is None:
        quotes = fetch_quotes(list(ref["codes"].keys()), session)
    log.info("live market: %d quotes", len(quotes))
    return build(ref, quotes, now_cn)
