import pandas as pd

from astock import data, realtime
from test_astock import synthetic_kline


class _Resp:
    def __init__(self, text):
        self.content = text.encode("gbk")

    def raise_for_status(self):
        pass


class _Session:
    def __init__(self, quotes):
        self.quotes = quotes

    def get(self, url, **kw):
        syms = url.split("=", 1)[1].split(",")
        return _Resp("".join(f'v_{s}="{self.quotes[s]}";\n' for s in syms if s in self.quotes))


def _quote(close, preclose, open_, high, low, vol_lots, amount_wan, when):
    f = ["0"] * 50
    f[1], f[3], f[4], f[5], f[30], f[33], f[34], f[36], f[37], f[38] = (
        "名称", close, preclose, open_, when, high, low, vol_lots, amount_wan, "1.5")
    return "~".join(map(str, f))


def test_fill_today_from_tencent(monkeypatch):
    monkeypatch.setattr(data, "save_kline", lambda df, kind="kline": None)
    k, _ = synthetic_kline(n_codes=3, start="2026-09-01", end="2026-10-28", seed=1)
    codes = sorted(k["code"].unique())
    val = k[["date", "code"]].copy()
    val["peTTM"], val["pbMRQ"], val["psTTM"] = 20.0, 2.0, 4.0
    last = k[k["date"] == k["date"].max()].set_index("code")["close"]
    day = pd.Timestamp("2026-10-29")
    q = {}
    for i, c in enumerate(codes):
        pc = float(last[c])
        if i == 2:  # 停牌
            q[c.replace(".", "")] = _quote(pc, pc, 0, 0, 0, 0, 0, "20261029150003")
        else:
            q[c.replace(".", "")] = _quote(pc * 1.1, pc, pc, pc * 1.1, pc * 0.99, 1000, 500, "20261029150003")
    k2, v2 = realtime.fill_today(codes, day, k, val, session=_Session(q))
    t = k2[k2["date"] == day].set_index("code")
    assert len(t) == 3 and k2["date"].max() == day
    a, s = codes[0], codes[2]
    assert abs(t.loc[a, "close"] - last[a] * 1.1) < 1e-3 and t.loc[a, "volume"] == 100000 and t.loc[a, "amount"] == 5e6
    assert t.loc[a, "tradestatus"] == 1 and t.loc[s, "tradestatus"] == 0 and abs(t.loc[s, "close"] - last[s]) < 1e-3
    pe = v2[v2["date"] == day].set_index("code")["peTTM"]
    assert abs(pe[a] - 22.0) < 1e-3 and abs(pe[s] - 20.0) < 1e-3


def test_fill_today_rejects_stale_quotes(monkeypatch):
    monkeypatch.setattr(data, "save_kline", lambda df, kind="kline": None)
    k, _ = synthetic_kline(n_codes=2, start="2026-09-01", end="2026-10-28", seed=1)
    codes = sorted(k["code"].unique())
    q = {c.replace(".", ""): _quote(10, 10, 10, 10, 10, 1, 1, "20261029113000") for c in codes}  # 午间，未收盘
    try:
        realtime.fill_today(codes, pd.Timestamp("2026-10-29"), k, None, session=_Session(q))
    except RuntimeError:
        return
    raise AssertionError("should refuse quotes taken before the close")


def test_baostock_blacklist_fails_fast(monkeypatch):
    class _RS:
        error_code, error_msg = "10001011", "黑名单用户，请与管理员联系"

    class _BS:
        calls = 0

        def query_forecast_report(self, *a, **k):
            _BS.calls += 1
            return _RS()

    s = data.Session()
    s.bs = _BS()
    monkeypatch.setattr(data.time, "sleep", lambda x: None)
    try:
        s.query("query_forecast_report", "sh.600000")
    except data.BaostockBlocked:
        assert _BS.calls == 1
        return
    raise AssertionError("should raise BaostockBlocked")


def test_forecast_chunk_stops_when_blocked(monkeypatch):
    class _Sess:
        n = 0

        def query(self, *a, **k):
            _Sess.n += 1
            raise data.BaostockBlocked("黑名单用户")

    monkeypatch.setattr(data, "_worker_session", _Sess())
    parts, failed = data._fetch_forecast_chunk((["sh.1", "sh.2", "sh.3"], "2026-01-01", "2026-10-29"))
    assert parts == [] and failed == ["sh.1", "sh.2", "sh.3"] and _Sess.n == 1


def test_fill_today_rejects_mismatched_fields(monkeypatch):
    monkeypatch.setattr(data, "save_kline", lambda df, kind="kline": None)
    k, _ = synthetic_kline(n_codes=3, start="2026-09-01", end="2026-10-28", seed=1)
    codes = sorted(k["code"].unique())
    q = {c.replace(".", ""): _quote(10, 999, 10, 11, 9, 100, 10, "20261029150003") for c in codes}  # 昨收对不上
    try:
        realtime.fill_today(codes, pd.Timestamp("2026-10-29"), k, None, session=_Session(q))
    except RuntimeError as exc:
        assert "look wrong" in str(exc)
        return
    raise AssertionError("should refuse")
