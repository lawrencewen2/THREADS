"""拉取原始数据，带重试和本地缓存。

每个序列存成 data/raw/<key>.csv（date,value）。每日运行时增量合并：
新数据覆盖同日期旧值，拉取失败时保留旧缓存并记录缺失。
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from . import config

log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _retry(fn, attempts=4, base_delay=2.0):
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # 免费接口的异常类型五花八门
            if i == attempts - 1:
                raise
            delay = base_delay * 2**i
            log.warning("retry %s in %.0fs: %s", getattr(fn, "__name__", fn), delay, exc)
            time.sleep(delay)


def _clean(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = "value"
    return s


def sanitize(key, s):
    """剔除周末日期（各市场周末都不交易，出现即为接口杂质）和未收盘的当天行情，外汇序列再剔除单日坏点。"""
    s = s[s.index.dayofweek < 5]
    # 去掉当天还没收盘的行情：D 日的数据要到 D 日 22:00 UTC（美股收盘后）才算完整。
    # 定时任务延迟到亚洲开盘后才跑时，Yahoo 会把日经等的盘中价当作当天收盘返回。
    cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=22)).tz_localize(None).normalize()
    s = s[s.index <= cutoff]
    thr = config.SPIKE_FILTER.get(key)
    if thr and len(s) > 2:
        prev, nxt = s.shift(1), s.shift(-1)
        bad = ((s / ((prev + nxt) / 2) - 1).abs() > thr) & ((prev / nxt - 1).abs() < thr / 2)
        if bad.any():
            log.info("%s: dropped %d bad ticks %s", key, bad.sum(), [d.date() for d in s.index[bad]][:5])
        s = s[~bad]
    if key in config.GLITCH_KEYS:
        s = _fix_glitches(key, s)
    return s


def _fix_glitches(key, s):
    """修正价格序列里短暂的量级错误（见 config.GLITCH_KEYS 的说明）。"""
    ratio, max_run = config.GLITCH_RATIO, config.GLITCH_MAX_RUN
    v = s.to_numpy(dtype=float).copy()
    keep = np.ones(len(v), dtype=bool)
    off = lambda x, base: x < base / ratio or x > base * ratio  # noqa: E731
    i = 1
    while i < len(v) - 1:
        prev = v[i - 1]
        if not (prev > 0 and off(v[i], prev)):
            i += 1
            continue
        j = i
        while j < len(v) and j - i < max_run and off(v[j], prev):
            j += 1
        if j >= len(v) or not (prev / 1.5 < v[j] < prev * 1.5):
            i += 1  # 没有回到原水平：可能是真实变化或还没走完，不动
            continue
        lo, hi = min(prev, v[j]) / 1.15, max(prev, v[j]) * 1.15
        for k in range(i, j):
            fixed = next((v[k] * f for f in (10, 0.1, 100, 0.01) if lo < v[k] * f < hi), None)
            if fixed is None:
                keep[k] = False
            else:
                v[k] = fixed
        log.info("%s: fixed %d glitch day(s) from %s (%d rescaled, %d dropped)", key, j - i, s.index[i].date(),
                 int(keep[i:j].sum()), int((~keep[i:j]).sum()))
        i = j
    return pd.Series(v, index=s.index, name=s.name)[keep]


def _yahoo_recent(ticker):
    """最近几天每个交易日的收盘价：先取日线，日线缺的日子用 30 分钟线当天最后一根补上。"""
    import yfinance as yf

    t = yf.Ticker(ticker)
    daily = t.history(period="7d", interval="1d", auto_adjust=False)
    daily = _clean(daily["Close"]) if daily is not None and not daily.empty else pd.Series(dtype=float)
    bars = t.history(period="5d", interval="30m", auto_adjust=False, prepost=False)
    if bars is None or bars.empty:
        return daily
    close = bars["Close"].dropna()
    by_day = close.groupby(close.index.date).last()
    by_day.index = pd.to_datetime(by_day.index)
    return daily.combine_first(by_day)


def fetch_yahoo(ticker, start):
    import yfinance as yf

    def go():
        df = yf.download(ticker, start=start, progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            raise ValueError(f"empty yahoo data for {ticker}")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close

    s = _clean(_retry(go))
    # Yahoo 日线的最新一天有时几个小时内收盘价为空（被 dropna 去掉），这里用最近几天的数据补齐
    try:
        recent = _yahoo_recent(ticker)
        missing = recent[recent.index > s.index[-1]]
        if len(missing):
            log.info("%s: daily bars end %s, filled %s from recent bars", ticker, s.index[-1].date(),
                     [d.date() for d in missing.index])
            s = pd.concat([s, missing])
    except Exception as exc:
        log.warning("%s: recent bars failed: %s", ticker, exc)
    return s


def _ak_index(provider, symbol, field):
    from .akcompat import call

    if provider == "em":
        df = call("stock_zh_index_daily_em", symbol=symbol)
    elif provider == "tx":
        df = call("stock_zh_index_daily_tx", symbol=symbol)
    elif provider == "sina":
        df = call("stock_zh_index_daily", symbol=symbol)
    else:
        raise KeyError(provider)
    return pd.to_numeric(df.set_index("date")[field], errors="coerce")


def _sse_margin_tail(s):
    """金十的上交所融资余额有时停更多日（2026-11 停在 11-05），用上交所官网最近 60 天的数据补上缺的日子。

    两边都是元为单位的同一指标（核查里两者比值约为 1），只补金十最后一天之后的日期，不改已有历史。
    """
    from . import akcompat

    try:
        end = pd.Timestamp.today()
        df = akcompat.call("stock_margin_sse", start_date=f"{end - pd.Timedelta(days=60):%Y%m%d}",
                           end_date=f"{end:%Y%m%d}")
        sse = pd.to_numeric(df["融资余额"], errors="coerce")
        sse.index = pd.to_datetime(df["信用交易日期"].astype(str))
        sse = sse.dropna().sort_index()
    except Exception as exc:
        log.warning("sh_margin: SSE official fill failed: %s", str(exc)[:120])
        return s
    last = pd.to_datetime(s.dropna().index).max()
    tail = sse[sse.index > last]
    if len(tail):
        log.info("sh_margin: jin10 ends %s, filled %d days from SSE official", last.date(), len(tail))
        s = pd.concat([s, tail])
    return s


SSE_IV_KEY = "_sse_iv50"   # 上交所口径的 50ETF 30 天平值隐含波动率，只用来给 qvix50 补尾，不是指数成分
# 用最近多少个重叠交易日对齐两者的水平。2026-08-20 至 11-18 的 54 个重叠日里，QVIX 与它的比值在 1.0–1.2 之间慢慢漂移，
# 窗口短一些跟得更紧：用最近 20 天的比值估 11-19，得 14.69，optbbs 分钟线停更前最后一笔（11-19 收盘）是 14.75。
SSE_IV_OVERLAP = 20
SSE_IV_MIN_OVERLAP = 10
SSE_IV_BUDGET = 150        # 每次运行最多花多少秒请求上交所（每个日期约 2 秒；首次要回补一个多月，补不完下次接着补）


def _option_expiry(yymm):
    """上交所 ETF 期权到期日：到期月份的第四个星期三（遇节假日顺延一两天，对 30 天插值影响很小，这里不处理）。"""
    first = pd.Timestamp(year=2000 + int(yymm[:2]), month=int(yymm[2:]), day=1)
    return first + pd.Timedelta(days=(2 - first.dayofweek) % 7) + pd.Timedelta(weeks=3)


def atm_iv_30d(df, date, underlying="510050"):
    """从上交所某一天的期权风险指标表算 30 天平值隐含波动率（%）。

    每个到期月：认购、认沽各自按 Delta 插值到 ±0.5 处的隐含波动率，取平均；
    再把 30 天两侧的两个到期月按总方差线性插值到 30 天（做法同 VIX 的期限插值）。算不出来返回 None。
    """
    m = df["合约交易代码"].astype(str).str.extract(rf"^{underlying}([CP])(\d{{4}})([MA])\d+$")
    d = pd.DataFrame({"cp": m[0], "ym": m[1], "std": m[2], "delta": pd.to_numeric(df["Delta"], errors="coerce"),
                      "iv": pd.to_numeric(df["隐含波动率"], errors="coerce")}).dropna()
    d = d[(d["iv"] > 0) & (d["delta"].abs() > 0.05) & (d["delta"].abs() < 0.95)]
    if d.empty:
        return None
    if d["iv"].median() < 3:  # 官网给的是小数（0.15 = 15%）
        d["iv"] = d["iv"] * 100
    points = []
    for ym, g in d.groupby("ym"):
        days = (_option_expiry(ym) - pd.Timestamp(date)).days
        if days < 1:
            continue
        g = g[g["std"] == "M"] if (g["std"] == "M").sum() >= 4 else g  # 优先用标准合约，分红调整过的合约（A）备用
        sides = []
        for cp in ("C", "P"):
            s = g[g["cp"] == cp].assign(a=lambda x: x["delta"].abs()).sort_values("a")
            if len(s) >= 2 and s["a"].iloc[0] <= 0.5 <= s["a"].iloc[-1]:
                sides.append(float(np.interp(0.5, s["a"], s["iv"])))
        if len(sides) == 2:
            points.append((days, sum(sides) / 2))
    points.sort()
    if len(points) > 1 and points[0][0] < 7:  # 临近到期的合约波动率失真，换到下一个月
        points = points[1:]
    if not points:
        return None
    for (t1, v1), (t2, v2) in zip(points, points[1:]):
        if t1 <= 30 <= t2 and t2 > t1:
            var = ((t2 - 30) * v1 * v1 * t1 + (30 - t1) * v2 * v2 * t2) / (t2 - t1) / 30
            return float(np.sqrt(var))
    return float(min(points, key=lambda p: abs(p[0] - 30))[1])


def _sse_iv_series(start, end):
    """[start, end] 之间每个交易日的上交所口径隐含波动率，带本地缓存，只请求还没有的日期（从最近的日子往前补）。"""
    from . import akcompat

    cached = load_cached(SSE_IV_KEY)
    cached = cached.dropna() if cached is not None else pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    want = [d for d in pd.bdate_range(start, end) if d not in cached.index]
    if len(cached[cached.index >= start]) >= SSE_IV_MIN_OVERLAP:
        # 缓存够用时只看最近一周，不反复请求更早的节假日
        want = [d for d in want if d > cached.index[-1] - pd.Timedelta(days=7)]
    got, misses, t0 = {}, 0, time.monotonic()
    for d in reversed(want):
        if time.monotonic() - t0 > SSE_IV_BUDGET:
            log.info("sse iv: time budget used up at %s, the rest next run", d.date())
            break
        try:
            df = akcompat.call("option_risk_indicator_sse", date=f"{d:%Y%m%d}")
            value, misses = atm_iv_30d(df, d), 0
        except KeyError:  # 节假日或当天还没公布：返回空表
            value = None
        except Exception as exc:
            value, misses = None, misses + 1
            log.warning("sse iv %s: %s: %s", d.date(), type(exc).__name__, str(exc)[:80])
            if misses >= 3:
                break
        if value:
            got[d] = round(value, 4)
        time.sleep(0.2)
    if got:
        cached = pd.concat([cached, pd.Series(got)]).sort_index()
        cached = cached[~cached.index.duplicated(keep="last")]
        save_cached(SSE_IV_KEY, cached)
    return cached


def _sse_iv_tail(s):
    """optbbs 的 QVIX 停更时（2026-11-19 起日线和分钟线都不再更新），用上交所官网的期权风险指标补上之后的日子。

    上交所每天公布每个期权合约的隐含波动率，从中算出 30 天平值隐含波动率（atm_iv_30d），
    再乘以最近 SSE_IV_OVERLAP 个重叠交易日里 QVIX 与它的比值中位数，对齐到 QVIX 的水平。
    只补 QVIX 最后一天之后的日期，不改已有历史；optbbs 恢复后，它的数据会覆盖补上的值。
    """
    s = _clean(s)
    s = s[s.index.notna()]
    try:
        last = s.index[-1]
        today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
        iv = _sse_iv_series(last - pd.Timedelta(days=int(SSE_IV_OVERLAP * 2.2)), today)
        both = pd.concat([s, iv], axis=1, keys=["q", "iv"], sort=True).dropna().tail(SSE_IV_OVERLAP)
        tail = iv[iv.index > last]
        if tail.empty:
            return s
        if len(both) < SSE_IV_MIN_OVERLAP:
            log.warning("qvix50: only %d overlap days with SSE official IV, not filling", len(both))
            return s
        ratio = float((both["q"] / both["iv"]).median())
        err = float((both["iv"] * ratio / both["q"] - 1).abs().median())
        if not 0.6 < ratio < 1.6:
            log.warning("qvix50: SSE official IV ratio %.2f out of range, not filling", ratio)
            return s
    except Exception as exc:
        log.warning("qvix50: SSE official fill failed: %s: %s", type(exc).__name__, str(exc)[:120])
        return s
    log.info("qvix50: optbbs ends %s, filled %d days from SSE official IV (ratio %.3f over %d days, median abs error %.1f%%)",
             last.date(), len(tail), ratio, len(both), err * 100)
    return pd.concat([s, (tail * ratio).round(2)])


def fetch_akshare(spec, start):
    import akshare as ak

    from . import akcompat

    def go():
        if ":" in spec:
            # 形如 em_close:sh000300 或 tx_amount:sh000001+sz399001（多个指数相加）
            kind, symbols = spec.split(":", 1)
            provider, field = kind.split("_", 1)
            parts = [_ak_index(provider, s, field) for s in symbols.split("+")]
            total = parts[0]
            for p in parts[1:]:
                total = total + p
            return total.dropna()
        if spec == "sh_margin":
            df = akcompat.call("macro_china_market_margin_sh")
            date_col = df.columns[0]
            # 融资余额列名在不同版本里略有差异，按关键字匹配
            col = next(c for c in df.columns if "融资余额" in str(c))
            s = df.set_index(date_col)[col]
            s.index = pd.to_datetime(s.index)
            return _sse_margin_tail(s)
        if spec == "qvix50":
            try:
                df = ak.index_option_50etf_qvix()
                s = df.set_index("date")["close"]
            except Exception as exc:
                # optbbs 整个连不上时沿用缓存（里面可能已有之前补上的日子），再接着往后补
                s = load_cached("qvix50")
                if s is None:
                    raise
                log.warning("qvix50: optbbs failed (%s), continuing from cache", str(exc)[:80])
            return _sse_iv_tail(s)
        raise KeyError(spec)

    s = _clean(_retry(go, attempts=3))
    return s[s.index >= pd.Timestamp(start)]


FETCHERS = {"yahoo": fetch_yahoo, "akshare": fetch_akshare}
SOURCES_FILE = RAW_DIR / "_sources.json"
MIN_HISTORY_ROWS = 250  # 新数据源至少要有约 1 年历史，否则视为失败、换下一个源


def load_cached(key):
    path = RAW_DIR / f"{key}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df["value"]


def save_cached(key, s):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    s.rename("value").rename_axis("date").to_csv(RAW_DIR / f"{key}.csv", float_format="%.6g")


def _load_sources():
    try:
        return json.loads(SOURCES_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return {}


LEVEL_BREAK = 3.0  # 新旧数据量级相差超过这个倍数，视为口径变化
LAST_REBASED = []  # 最近一次 update_all 中因口径变化而整段重拉的序列


def _level_break(cached, fresh):
    """增量数据和缓存是否量级不一致。

    同日重叠部分本应几乎相同（只允许小幅补录修正）；接缝处前后各 20 天的中位数也不应相差数倍。
    """
    cached, fresh = cached.dropna(), fresh.dropna()
    if cached.empty or fresh.empty:
        return False
    common = cached.index.intersection(fresh.index)
    if len(common) >= 5:
        ratio = (fresh[common] / cached[common]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) and not (1 / 1.5 < ratio.median() < 1.5):
            return True
    before = cached[cached.index < fresh.index[0]].tail(20)
    if len(before) < 5:
        return False
    a, b = before.median(), fresh.head(20).median()
    if a <= 0 or b <= 0:
        return False
    return not (1 / LEVEL_BREAK < b / a < LEVEL_BREAK)


def update_all(keys=None, start=config.START_DATE, full=False):
    """拉取并合并所有序列。返回 (dict key->Series, dict key->错误信息)。

    full=False 时只拉最近约 1 年并与缓存合并，减轻接口压力；部分接口不支持起始日期，
    会返回全量，同样合并即可。某个序列改用了备用数据源时整段重拉，避免量纲不同的数据拼接。
    """
    keys = keys or list(config.SERIES)
    sources = _load_sources()
    out, errors = {}, {}
    LAST_REBASED.clear()
    for key in keys:
        cached = load_cached(key)
        failures = []
        chain = list(config.SERIES[key])
        # 优先沿用缓存所用的数据源，避免主备来回切换导致整段重拉
        chain.sort(key=lambda sc: f"{sc[0]}:{sc[1]}" != sources.get(key))
        for source, code in chain:
            label = f"{source}:{code}"
            same_source = (cached is not None and len(cached) >= MIN_HISTORY_ROWS
                           and sources.get(key) == label)
            fetch_start = start
            if same_source and not full:
                fetch_start = (cached.index[-1] - pd.Timedelta(days=370)).strftime("%Y-%m-%d")
            try:
                fresh = FETCHERS[source](code, fetch_start)
                if not same_source and len(fresh) < MIN_HISTORY_ROWS:
                    raise ValueError(f"only {len(fresh)} rows of history")
            except Exception as exc:
                failures.append(f"{label} {type(exc).__name__}: {str(exc)[:120]}")
                log.warning("%s via %s failed: %s", key, label, failures[-1])
                continue
            if same_source and not full and _level_break(cached, fresh):
                # 同一数据源但口径变了（例如接口把成交量换成成交额），新旧拼接会出现断层，整段重拉
                log.warning("%s via %s: level break vs cache, refetching full history", key, label)
                try:
                    fresh = FETCHERS[source](code, start)
                    if len(fresh) < MIN_HISTORY_ROWS:
                        raise ValueError(f"only {len(fresh)} rows of history")
                except Exception as exc:
                    failures.append(f"{label} 口径变化后整段重拉失败 {type(exc).__name__}: {str(exc)[:100]}")
                    log.warning("%s via %s failed: %s", key, label, failures[-1])
                    continue
                same_source = False
                LAST_REBASED.append(key)
            merged = pd.concat([cached[cached.index < fresh.index[0]], fresh]) if same_source else fresh
            merged = sanitize(key, _clean(merged))
            save_cached(key, merged)
            sources[key] = label
            # 用存盘后的值计算（CSV 保留 6 位有效数字），离线重算才能和线上结果完全一致
            merged = load_cached(key)
            out[key] = merged
            log.info("%s via %s: %d rows, last %s", key, label, len(merged), merged.index[-1].date())
            break
        else:
            errors[key] = "; ".join(failures)
            log.error("%s failed on all sources", key)
            if cached is not None:
                out[key] = sanitize(key, cached)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_FILE.write_text(json.dumps(sources, indent=2))
    return out, errors


def load_all():
    out = {}
    for key in config.SERIES:
        s = load_cached(key)
        if s is not None:
            out[key] = sanitize(key, s)
    return out
