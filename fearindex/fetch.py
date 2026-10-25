"""拉取原始数据，带重试和本地缓存。

每个序列存成 data/raw/<key>.csv（date,value）。每日运行时增量合并：
新数据覆盖同日期旧值，拉取失败时保留旧缓存并记录缺失。
"""

import json
import logging
import time
from pathlib import Path

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
    """剔除周末日期（各市场周末都不交易，出现即为接口杂质），外汇序列再剔除单日坏点。"""
    s = s[s.index.dayofweek < 5]
    thr = config.SPIKE_FILTER.get(key)
    if thr and len(s) > 2:
        prev, nxt = s.shift(1), s.shift(-1)
        bad = ((s / ((prev + nxt) / 2) - 1).abs() > thr) & ((prev / nxt - 1).abs() < thr / 2)
        if bad.any():
            log.info("%s: dropped %d bad ticks %s", key, bad.sum(), [d.date() for d in s.index[bad]][:5])
        s = s[~bad]
    return s


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

    return _clean(_retry(go))


def _ak_index(provider, symbol, field):
    import akshare as ak

    if provider == "em":
        df = ak.stock_zh_index_daily_em(symbol=symbol)
    elif provider == "tx":
        df = ak.stock_zh_index_daily_tx(symbol=symbol)
    elif provider == "sina":
        df = ak.stock_zh_index_daily(symbol=symbol)
    else:
        raise KeyError(provider)
    return pd.to_numeric(df.set_index("date")[field], errors="coerce")


def fetch_akshare(spec, start):
    import akshare as ak

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
            df = ak.macro_china_market_margin_sh()
            date_col = df.columns[0]
            # 融资余额列名在不同版本里略有差异，按关键字匹配
            col = next(c for c in df.columns if "融资余额" in str(c))
            return df.set_index(date_col)[col]
        if spec == "qvix50":
            df = ak.index_option_50etf_qvix()
            return df.set_index("date")["close"]
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


def update_all(keys=None, start=config.START_DATE, full=False):
    """拉取并合并所有序列。返回 (dict key->Series, dict key->错误信息)。

    full=False 时只拉最近约 1 年并与缓存合并，减轻接口压力；部分接口不支持起始日期，
    会返回全量，同样合并即可。某个序列改用了备用数据源时整段重拉，避免量纲不同的数据拼接。
    """
    keys = keys or list(config.SERIES)
    sources = _load_sources()
    out, errors = {}, {}
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
            merged = pd.concat([cached[cached.index < fresh.index[0]], fresh]) if same_source else fresh
            merged = sanitize(key, _clean(merged))
            save_cached(key, merged)
            sources[key] = label
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
