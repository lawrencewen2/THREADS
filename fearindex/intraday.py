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
# 区域的主指数有了新行情，这个区域才算进入新的一天。汇率、黄金等几乎 24 小时交易，
# 主指数还没开盘时只接汇率会让区域在“大部分成分还是昨天”的状态下被平滑一次，读数失真。
PRIMARY = {"us": "spx", "eu": "sx5e", "jp": "n225", "cn": "csi300", "hk": "hsi", "global": "spx"}

# 盘中急跌提醒：分数比上一交易日正式读数低这么多分（每日版历史上每区域每年约 4–10 次）
DROP_ALERT = 6
EXTREME_LOW, EXTREME_HIGH = 20, 80

# 各市场当地交易时段（时区自动处理夏令时），用于标注“未开盘 / 盘中 / 午休 / 已收盘”
MARKET_HOURS = {"us": ("America/New_York", [(9, 30, 16, 0)]),
                "eu": ("Europe/Berlin", [(9, 0, 17, 30)]),
                "jp": ("Asia/Tokyo", [(9, 0, 11, 30), (12, 30, 15, 30)]),
                "cn": ("Asia/Shanghai", [(9, 30, 11, 30), (13, 0, 15, 0)]),
                "hk": ("Asia/Hong_Kong", [(9, 30, 12, 0), (13, 0, 16, 0)])}


# ---------- 行情 ----------

def _yahoo_quote(symbol, session=None):
    """最近一根分钟线的收盘价。yfinance 自带 Yahoo 需要的 cookie，直接请求接口会被限流（429）。"""
    import yfinance as yf

    last_exc = None
    for period, interval in (("1d", "1m"), ("5d", "5m")):
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval, prepost=False, auto_adjust=False)
        except Exception as exc:
            last_exc = exc
            continue
        close = df["Close"].dropna() if df is not None and not df.empty else None
        if close is None or close.empty:
            continue
        ts = close.index[-1]
        return dict(value=float(close.iloc[-1]), time=ts.tz_convert("UTC").to_pydatetime(),
                    date=pd.Timestamp(ts.date()))
    raise ValueError(f"no intraday data for {symbol}: {last_exc}")


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
            quotes[key] = _yahoo_quote(sym)
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


def market_session(region, now=None):
    """按当地时钟返回 "未开盘"、"盘中"、"午休" 或 "已收盘"。节假日按时钟判断，是否有新行情以 fresh 为准。"""
    tz, spans = MARKET_HOURS[region]
    t = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo(tz))
    if t.weekday() >= 5:
        return "已收盘" if t.weekday() == 5 else "未开盘"
    m = t.hour * 60 + t.minute
    spans = [(h1 * 60 + m1, h2 * 60 + m2) for h1, m1, h2, m2 in spans]
    if m < spans[0][0]:
        return "未开盘"
    if m >= spans[-1][1]:
        return "已收盘"
    return "盘中" if any(a <= m < b for a, b in spans) else "午休"


# ---------- 估算 ----------

def _turnover_value(q, hist):
    """把当天已成交额按已交易时间折算成全天，并换算成历史数据的单位。开盘前 30 分钟不折算。"""
    if q.get("minutes", 0) < 30 or hist is None or hist.empty:
        return None
    full = q["value"] * 240 / q["minutes"]
    scale = 10 ** round(math.log10(hist.iloc[-1] / full))
    return full * scale


def apply_quotes(series, quotes):
    """把比历史更新的行情接到序列末尾。返回 (新的 series, 接上了的 key 列表)。

    一个序列只在用到它的每个区域的主指数都有新行情时才接上（见 PRIMARY）。
    """
    def newer(key):
        hist, q = series.get(key), quotes.get(key)
        return (q is not None and hist is not None and not hist.empty and key not in NO_INTRADAY
                and q["date"] > hist.index[-1] and q["date"].dayofweek < 5)

    open_regions = {r for r, k in PRIMARY.items() if newer(k)}
    out, fresh = dict(series), []
    for key, q in quotes.items():
        users = {c["region"] for c in config.COMPONENTS.values() if key in c["inputs"]}
        if not newer(key) or not users or not users <= open_regions:
            continue
        hist = series[key]
        value = _turnover_value(q, hist) if q.get("partial") else q["value"]
        if value is None or not value > 0:
            continue
        out[key] = pd.concat([hist, pd.Series([value], index=[q["date"]])])
        fresh.append(key)
    return out, fresh


def _global_live(official, live, fresh_regions, common_fresh):
    """全球盘中读数 = 正式读数 + 有新行情的区域各自的变化 × 其权重。

    全球分是区域分的线性组合，平滑也是线性的，所以按区域拆开相加与整体重算等价；
    但还没开盘的区域如果在新的一天里沿用昨天的成分值，会被平滑一次而失真，拆开算就避开了这一点。
    """
    off = official["global"].dropna()
    total_w = sum(config.REGION_WEIGHTS.values())
    delta = 0.0
    for r in fresh_regions:
        delta += (1 - config.COMMON_WEIGHT) * config.REGION_WEIGHTS[r] / total_w * (
            live[r].dropna().iloc[-1] - official[r].dropna().iloc[-1])
    if common_fresh:
        delta += config.COMMON_WEIGHT * (live["common"].dropna().iloc[-1] - official["common"].dropna().iloc[-1])
    date = max(live[r].dropna().index[-1] for r in fresh_regions)
    return pd.concat([off, pd.Series([round(off.iloc[-1] + delta, 1)], index=[date])])


def estimate(series, quotes):
    """返回盘中读数 dict。主指数没有新行情的区域显示上一交易日正式读数（fresh=False）。"""
    official, _, _ = compute.build(series)
    live_series, fresh = apply_quotes(series, quotes)
    live, _, _ = compute.build(live_series) if fresh else (official, None, None)
    fresh_regions = [r for r in config.REGIONS if PRIMARY[r] in fresh]
    common_fresh = PRIMARY["global"] in fresh
    regions = {}
    for key in ["global"] + list(config.REGIONS):
        off = official[key].dropna()
        is_fresh = key in fresh_regions or (key == "global" and bool(fresh_regions))
        if key == "global" and is_fresh:
            src = _global_live(official, live, fresh_regions, common_fresh)
        else:
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
    now = datetime.now(timezone.utc)
    for r in config.REGIONS:
        sess = market_session(r, now)
        # 当地已跨过午夜、新一天还没开盘，但上一交易日的收盘还没并入正式指数
        regions[r]["session"] = "已收盘" if sess == "未开盘" and regions[r]["fresh"] else sess
    sessions = [regions[r]["session"] for r in config.REGIONS]
    regions["global"]["session"] = ("盘中" if "盘中" in sessions or "午休" in sessions
                                    else "已收盘" if regions["global"]["fresh"] else "未开盘")
    q_out = {k: dict(value=round(q["value"], 6), time=q["time"].isoformat(timespec="minutes"), date=str(q["date"].date()),
                     fresh=k in fresh) for k, q in quotes.items()}
    return dict(as_of=now.isoformat(timespec="minutes"), method=config.fingerprint(),
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
