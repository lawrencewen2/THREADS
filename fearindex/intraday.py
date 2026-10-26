"""盘中估算：把各序列的最新行情当作“今天的收盘价”接在历史后面，用每日版完全相同的算法重算。

读数回答的是“如果现在收盘，指数是多少”。拿不到盘中数据的序列（融资余额、MOVE）沿用上一交易日。
盘中读数不写回 data/raw，也不改动每日版的历史和回测。
"""

import logging
import math
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import compute, config

log = logging.getLogger(__name__)

CN_TZ = ZoneInfo("Asia/Shanghai")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

# 每个序列的盘中行情来源。Yahoo 代码与每日版所用的一致（data/raw/_sources.json）。
YAHOO = {"spx": "^GSPC", "vix": "^VIX", "vix3m": "^VIX3M", "hyg": "HYG", "ief": "IEF", "sx5e": "^STOXX50E",
         "eurchf": "EURCHF=X", "n225": "^N225", "usdjpy": "JPY=X", "topix": "1306.T", "hsi": "^HSI",
         "hstech": "3033.HK", "gold": "GC=F", "us10y": "^TNX"}
TENCENT = {"csi300": "sh000300"}
TURNOVER_CODES = ("sh000001", "sz399001")
NO_INTRADAY = ("sh_margin", "move")

# 盘中急跌提醒：分数比上一交易日正式读数低这么多分（每日版历史上每区域每年约 4–10 次）
DROP_ALERT = 6
EXTREME_LOW, EXTREME_HIGH = 20, 80


# ---------- 行情 ----------

def _yahoo_quote(symbol, session):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = session.get(url, params={"interval": "1m", "range": "1d"}, headers=UA, timeout=15)
    r.raise_for_status()
    meta = r.json()["chart"]["result"][0]["meta"]
    price, ts = meta.get("regularMarketPrice"), meta.get("regularMarketTime")
    if price is None or ts is None:
        raise ValueError(f"no price for {symbol}")
    tz = ZoneInfo(meta.get("exchangeTimezoneName") or "UTC")
    t = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dict(value=float(price), time=t, date=pd.Timestamp(t.astimezone(tz).date()))


def _tencent_raw(codes, session):
    r = session.get("http://qt.gtimg.cn/q=" + ",".join(codes), headers=UA, timeout=15)
    r.raise_for_status()
    text = r.content.decode("gbk", errors="replace")
    out = {}
    for code, body in re.findall(r'v_(\w+)="([^"]*)"', text):
        f = body.split("~")
        if len(f) > 37:
            out[code] = f
    return out


def _tencent_time(f):
    t = datetime.strptime(f[30], "%Y%m%d%H%M%S").replace(tzinfo=CN_TZ)
    return t.astimezone(timezone.utc), pd.Timestamp(t.date())


def _a_share_minutes(t_cn):
    """A股当天已交易的分钟数（9:30–11:30、13:00–15:00，共 240 分钟）。"""
    m = t_cn.hour * 60 + t_cn.minute
    return max(0, min(m, 690) - 570) + max(0, min(m, 900) - 780)


def fetch_quotes(session=None):
    """返回 (dict key -> {value, time, date}, dict key -> 错误)。"""
    import requests

    session = session or requests.Session()
    quotes, errors = {}, {}
    for key, sym in YAHOO.items():
        try:
            quotes[key] = _yahoo_quote(sym, session)
        except Exception as exc:
            errors[key] = f"yahoo {sym}: {type(exc).__name__}: {str(exc)[:100]}"
    try:
        raw = _tencent_raw(list(TENCENT.values()) + list(TURNOVER_CODES), session)
        for key, code in TENCENT.items():
            if code in raw:
                t, d = _tencent_time(raw[code])
                quotes[key] = dict(value=float(raw[code][3]), time=t, date=d)
            else:
                errors[key] = f"tencent {code}: missing"
        if all(c in raw for c in TURNOVER_CODES):
            t, d = _tencent_time(raw[TURNOVER_CODES[0]])
            amount = sum(float(raw[c][37]) for c in TURNOVER_CODES)  # 万元
            quotes["a_turnover"] = dict(value=amount, time=t, date=d, partial=True,
                                        minutes=_a_share_minutes(t.astimezone(CN_TZ)))
    except Exception as exc:
        errors["csi300"] = errors["a_turnover"] = f"tencent: {type(exc).__name__}: {str(exc)[:100]}"
    try:
        quotes["qvix50"] = _qvix_quote()
    except Exception as exc:
        errors["qvix50"] = f"qvix: {type(exc).__name__}: {str(exc)[:100]}"
    return quotes, errors


def _qvix_quote():
    import akshare as ak

    df = ak.index_option_50etf_min_qvix()
    df = df.dropna()
    if df.empty:
        raise ValueError("empty")
    last = df.iloc[-1]
    now = datetime.now(CN_TZ)
    hh, mm = (int(x) for x in str(last.iloc[0]).split(":")[:2])
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return dict(value=float(last.iloc[-1]), time=t.astimezone(timezone.utc), date=pd.Timestamp(now.date()))


# ---------- 估算 ----------

def _turnover_value(q, hist):
    """把当天已成交额按已交易时间折算成全天，并换算成历史数据的单位。开盘前 30 分钟不折算。"""
    if q.get("minutes", 0) < 30 or hist is None or hist.empty:
        return None
    full = q["value"] * 240 / q["minutes"]
    scale = 10 ** round(math.log10(hist.iloc[-1] / full))
    return full * scale


def apply_quotes(series, quotes):
    """把比历史更新的行情接到序列末尾。返回 (新的 series, 接上了的 key 列表)。"""
    out, fresh = dict(series), []
    for key, q in quotes.items():
        hist = series.get(key)
        if hist is None or hist.empty or key in NO_INTRADAY:
            continue
        if q["date"] <= hist.index[-1] or q["date"].dayofweek >= 5:
            continue
        value = _turnover_value(q, hist) if q.get("partial") else q["value"]
        if value is None or not value > 0:
            continue
        out[key] = pd.concat([hist, pd.Series([value], index=[q["date"]])])
        fresh.append(key)
    return out, fresh


def _region_inputs(region):
    return {k for c in config.COMPONENTS.values() if c["region"] == region for k in c["inputs"]}


def estimate(series, quotes):
    """返回盘中读数 dict。没有新行情的区域显示上一交易日正式读数（fresh=False）。"""
    official, _, _ = compute.build(series)
    live_series, fresh = apply_quotes(series, quotes)
    live, _, _ = compute.build(live_series) if fresh else (official, None, None)
    regions = {}
    fresh_regions = set()
    for r in config.REGIONS:
        if _region_inputs(r) & set(fresh):
            fresh_regions.add(r)
    for key in ["global"] + list(config.REGIONS):
        off = official[key].dropna()
        is_fresh = key in fresh_regions or (key == "global" and bool(fresh_regions))
        src = live[key].dropna() if is_fresh else off
        score = float(src.iloc[-1])
        sig = config.signals(src).iloc[-1]
        regions[key] = dict(
            name=config.REGIONS.get(key, "全球"), fresh=is_fresh, score=round(score, 1), band=config.band(score),
            date=str(src.index[-1].date()), official=float(off.iloc[-1]), official_date=str(off.index[-1].date()),
            change=round(score - float(off.iloc[-1]), 1), signal=sig, signal_text=config.SIGNAL_TEXT[sig],
            official_signal=config.signals(off).iloc[-1],
            new_event={k: bool(is_fresh and (ev := config.extreme_events(src, k)) and ev[-1] == src.index[-1])
                       for k in ("short_bottom", "bottom")},
        )
    q_out = {k: dict(value=round(q["value"], 6), time=q["time"].isoformat(timespec="minutes"), date=str(q["date"].date()),
                     fresh=k in fresh) for k, q in quotes.items()}
    return dict(as_of=datetime.now(timezone.utc).isoformat(timespec="minutes"), method=config.fingerprint(),
                regions=regions, quotes=q_out, fresh=sorted(fresh))


def alerts(est):
    """根据盘中读数生成提醒：(区域, 种类, 文字)。只看有新行情的区域。"""
    out = []
    for key, r in est["regions"].items():
        if not r["fresh"]:
            continue
        name, s = r["name"], r["score"]
        if r["change"] <= -DROP_ALERT:
            out.append((key, "drop", f"{name}盘中急跌：{s}，比上一交易日（{r['official_date']}）低 {-r['change']} 分"))
        if r["signal"] == "risk_off" and r["official_signal"] != "risk_off":
            out.append((key, "risk_off", f"{name}如果现在收盘，会发出“恐慌加深，减仓”：{s}"))
        if r["new_event"]["bottom"]:
            out.append((key, "bottom", f"{name}如果现在收盘，会提示“恐慌解除（可能触底）”：{s}"))
        elif r["new_event"]["short_bottom"]:
            out.append((key, "short_bottom", f"{name}如果现在收盘，会提示“短期触底”：{s}"))
        if s <= EXTREME_LOW < r["official"]:
            out.append((key, "extreme_low", f"{name}盘中进入极度恐慌：{s}"))
        if s >= EXTREME_HIGH > r["official"]:
            out.append((key, "extreme_high", f"{name}盘中进入极度贪婪：{s}"))
    return out
