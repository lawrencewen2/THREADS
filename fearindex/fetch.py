"""拉取原始数据，带重试和本地缓存。

每个序列存成 data/raw/<key>.csv（date,value）。每日运行时增量合并：
新数据覆盖同日期旧值，拉取失败时保留旧缓存并记录缺失。
"""

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

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


def fetch_fred(series_id, start):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"

    def go():
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", "value"]
        return df.set_index("date")["value"].replace(".", None)

    return _clean(_retry(go))


def _ak_index_frame(symbol):
    import akshare as ak

    df = ak.stock_zh_index_daily_em(symbol=symbol)
    return df.set_index("date")


def fetch_akshare(spec, start):
    import akshare as ak

    def go():
        if spec.startswith("index_close:"):
            return _ak_index_frame(spec.split(":", 1)[1])["close"]
        if spec == "a_turnover":
            # 沪市 + 深市成交额（元）
            sh = pd.to_numeric(_ak_index_frame("sh000001")["amount"], errors="coerce")
            sz = pd.to_numeric(_ak_index_frame("sz399001")["amount"], errors="coerce")
            return (sh + sz).dropna()
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

    s = _clean(_retry(go))
    return s[s.index >= pd.Timestamp(start)]


FETCHERS = {"yahoo": fetch_yahoo, "fred": fetch_fred, "akshare": fetch_akshare}


def load_cached(key):
    path = RAW_DIR / f"{key}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df["value"]


def save_cached(key, s):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    s.rename("value").rename_axis("date").to_csv(RAW_DIR / f"{key}.csv", float_format="%.6g")


def update_all(keys=None, start=config.START_DATE, full=False):
    """拉取并合并所有序列。返回 (dict key->Series, dict key->错误信息)。

    full=False 时只拉最近约 1 年并与缓存合并，减轻接口压力；
    yfinance/akshare 部分接口不支持起始日期，会返回全量，同样合并即可。
    """
    keys = keys or list(config.SERIES)
    out, errors = {}, {}
    for key in keys:
        source, code = config.SERIES[key]
        cached = load_cached(key)
        fetch_start = start
        if cached is not None and not full and len(cached):
            fetch_start = (cached.index[-1] - pd.Timedelta(days=370)).strftime("%Y-%m-%d")
        try:
            fresh = FETCHERS[source](code, fetch_start)
            merged = fresh if cached is None else pd.concat([cached[cached.index < fresh.index[0]], fresh])
            merged = _clean(merged)
            save_cached(key, merged)
            out[key] = merged
            log.info("%s: %d rows, last %s", key, len(merged), merged.index[-1].date())
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
            log.error("%s failed: %s", key, errors[key])
            if cached is not None:
                out[key] = cached
    return out, errors


def load_all():
    out = {}
    for key in config.SERIES:
        s = load_cached(key)
        if s is not None:
            out[key] = s
    return out
