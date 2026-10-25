"""从 baostock 拉取 A股日线、指数成分、上市日期和交易日历。

baostock 是免费的 TCP 接口，GitHub Actions 上可以访问（东方财富会拒绝 Actions 的 IP）。
日线用不复权价格，涨跌停判断要用真实价格；复权收益由 close / preclose 自行算出。
"""

import logging
import time

import pandas as pd

from . import config

log = logging.getLogger(__name__)

KLINE_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST"
NUMERIC = ["open", "high", "low", "close", "preclose", "volume", "amount", "turn"]
KLINE_FILE = config.CACHE_DIR / "kline.pkl.gz"
UNIVERSE_FILE = config.DATA_DIR / "universe.csv"
BASIC_FILE = config.DATA_DIR / "basic.csv"


class Session:
    def __enter__(self):
        import baostock as bs

        self.bs = bs
        self._login()
        return self

    def __exit__(self, *exc):
        try:
            self.bs.logout()
        except Exception:
            pass

    def _login(self):
        lg = self.bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")

    def query(self, fn_name, *args, attempts=4, **kwargs):
        """调用 baostock 查询并返回 DataFrame；网络错误时重新登录重试。"""
        for i in range(attempts):
            try:
                rs = getattr(self.bs, fn_name)(*args, **kwargs)
                if rs.error_code != "0":
                    raise RuntimeError(f"{fn_name}: {rs.error_code} {rs.error_msg}")
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rs.error_code != "0":
                    raise RuntimeError(f"{fn_name}: {rs.error_code} {rs.error_msg}")
                return pd.DataFrame(rows, columns=rs.fields)
            except Exception as exc:
                if i == attempts - 1:
                    raise
                log.warning("retry %s %s: %s", fn_name, args[:1], exc)
                time.sleep(2 * 2**i)
                try:
                    self._login()
                except Exception:
                    pass

    def trade_dates(self, start, end):
        df = self.query("query_trade_dates", start_date=start, end_date=end)
        return pd.to_datetime(df.loc[df["is_trading_day"] == "1", "calendar_date"]).tolist()

    def constituents(self, date):
        codes = set()
        for name in config.INDEXES:
            df = self.query(f"query_{name}_stocks", date=date)
            codes |= set(df["code"])
        return sorted(codes)

    def basic(self):
        df = self.query("query_stock_basic")
        return df[["code", "code_name", "ipoDate", "outDate", "type", "status"]]

    def kline(self, code, start, end):
        df = self.query("query_history_k_data_plus", code, KLINE_FIELDS,
                        start_date=start, end_date=end, frequency="d", adjustflag="3")
        return df


def clean_kline(df):
    df = df.copy()
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["date"] = pd.to_datetime(df["date"])
    df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce").fillna(0).astype("int8")
    df["isST"] = pd.to_numeric(df["isST"], errors="coerce").fillna(0).astype("int8")
    return df


def snapshot_dates(sess, start, end):
    """每年 1 月、7 月的第一个交易日，对应指数半年一次的成分调整。"""
    days = pd.Series(sess.trade_dates(start, end))
    first = days.groupby([days.dt.year, (days.dt.month - 1) // 6]).min()
    return [d.strftime("%Y-%m-%d") for d in first]


def update_universe(sess, start=config.HISTORY_START, end=None):
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    old = load_universe()
    done = set(old["snapshot"]) if old is not None else set()
    rows = [] if old is None else [old]
    for d in snapshot_dates(sess, start, end):
        if d in done:
            continue
        codes = sess.constituents(d)
        log.info("universe %s: %d codes", d, len(codes))
        rows.append(pd.DataFrame({"snapshot": d, "code": codes}))
    uni = pd.concat(rows, ignore_index=True).drop_duplicates()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    uni.to_csv(UNIVERSE_FILE, index=False)
    return uni


def load_universe():
    if not UNIVERSE_FILE.exists():
        return None
    return pd.read_csv(UNIVERSE_FILE, dtype=str)


def update_basic(sess):
    df = sess.basic()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(BASIC_FILE, index=False)
    return df


def load_basic():
    return pd.read_csv(BASIC_FILE, dtype=str)


def load_kline():
    if not KLINE_FILE.exists():
        return None
    return pd.read_pickle(KLINE_FILE)


def save_kline(df):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(KLINE_FILE)


def update_kline(sess, codes, start=config.HISTORY_START, end=None, overlap_days=10):
    """增量更新日线缓存：已有的股票从缓存末尾往前 overlap_days 天重拉，新股票从 start 拉。"""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    old = load_kline()
    last = old.groupby("code")["date"].max() if old is not None else pd.Series(dtype="datetime64[ns]")
    parts, failed = [], []
    for i, code in enumerate(codes):
        s = start
        if code in last.index:
            s = max(pd.Timestamp(start), last[code] - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")
        try:
            df = sess.kline(code, s, end)
        except Exception as exc:
            log.warning("kline %s failed: %s", code, exc)
            failed.append(code)
            continue
        if len(df):
            parts.append(clean_kline(df))
        if i % 200 == 0:
            log.info("kline %d/%d", i, len(codes))
        if len(parts) >= 300:  # 定期落盘，任务超时也不丢已下载的数据
            old, parts = _merge(old, parts), []
    if old is None and not parts:
        raise RuntimeError("no kline data")
    return _merge(old, parts), failed


def _merge(old, parts):
    out = pd.concat(([old] if old is not None else []) + parts, ignore_index=True)
    out = out.drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"]).reset_index(drop=True)
    save_kline(out)
    return out
