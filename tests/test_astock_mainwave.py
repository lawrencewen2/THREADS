import numpy as np
import pandas as pd

from astock import mainwave
from astock.factors import Panel, eligible
from test_astock import synthetic_kline
from test_astock_alt_study import _forecast


def _live(p, elig):
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.random(p.close.shape), index=p.dates, columns=p.codes).where(elig) - 0.5


def test_mainwave_signals_have_no_lookahead():
    k, basic = synthetic_kline(n_codes=20, start="2014-01-01", end="2016-06-30", seed=5)
    cut = pd.Timestamp("2015-12-31")
    p1 = Panel(k)
    e1 = eligible(p1, basic)
    s1 = mainwave.signals(p1, e1, _live(p1, e1))
    r1, _ = mainwave.market_regimes(p1, e1)
    k2 = k.copy()
    k2.loc[k2["date"] > cut, ["open", "high", "low", "close"]] *= 1.5
    k2.loc[k2["date"] > cut, "turn"] *= 3
    p2 = Panel(k2)
    e2 = eligible(p2, basic)
    live2 = _live(p2, e2)
    live2.loc[live2.index > cut] = 0.3
    s2 = mainwave.signals(p2, e2, live2)
    r2, _ = mainwave.market_regimes(p2, e2)
    for name in s1:
        a, b = s1[name].loc[:cut], s2[name].loc[:cut]
        pd.testing.assert_frame_equal(a.notna(), b.notna())
    for name in r1:
        pd.testing.assert_series_equal(r1[name].loc[:cut], r2[name].loc[:cut])


def test_mainwave_run_end_to_end(monkeypatch):
    k, basic = synthetic_kline(n_codes=24, start="2014-01-01", end="2021-08-31", seed=7)
    codes = sorted(k["code"].unique())
    fc = pd.concat([_forecast(codes, d) for d in pd.date_range("2014-03-01", "2021-06-01", freq="QS")])
    monkeypatch.setattr(mainwave, "STAGE2_TOP", 1)
    monkeypatch.setattr(mainwave, "LEVELS", mainwave.LEVELS[:1])
    val = k[["date", "code"]].copy()
    val["peTTM"], val["pbMRQ"], val["psTTM"] = 15.0, 2.0, 3.0
    res = mainwave.run(k, basic, val, fc)
    assert res["rows"][0]["variant"] == "线上规则（对照）"
    assert any(r["group"] == "regime" for r in res["rows"])
    md = mainwave.to_markdown(res)
    assert "主升浪" in md
    assert len(res["curve"]["dates"]) == len(res["curve"]["market"])


def test_cap_zero_sells_everything():
    k, basic = synthetic_kline(n_codes=10, start="2014-01-01", end="2015-12-31", seed=2)
    p = Panel(k)
    elig = eligible(p, basic)
    from astock.winrate import adjusted_ohlc
    ohl = adjusted_ohlc(p)
    sc = _live(p, elig)
    gate = pd.Series(True, index=p.dates)
    cap = pd.Series(1.0, index=p.dates)
    cap.loc["2015-06-01":] = 0.0
    ex = (None, "x", None, 0.5, None, None, 1000)
    d, t, expo = mainwave.simulate(p, ohl, sc, gate, 5, ex, "2014-09-01", "2015-12-31", cap)
    assert expo.loc["2015-06-03":].max() == 0
    assert expo.loc["2014-10-01":"2015-05-29"].max() > 0


def test_industry_cap_limits_holdings_per_industry():
    k, basic = synthetic_kline(n_codes=12, start="2014-01-01", end="2015-12-31", seed=4)
    p = Panel(k)
    elig = eligible(p, basic)
    from astock.winrate import adjusted_ohlc
    ohl = adjusted_ohlc(p)
    sc = _live(p, elig)
    gate = pd.Series(True, index=p.dates)
    ind = np.zeros(len(p.codes), dtype=int)  # 全部同一行业
    ex = (None, "x", None, 0.5, None, None, 1000)
    _, _, expo = mainwave.simulate(p, ohl, sc, gate, 10, ex, "2014-09-01", "2015-12-31", None, ind, 2)
    assert expo.max() <= 0.2 + 1e-9
    _, _, expo2 = mainwave.simulate(p, ohl, sc, gate, 10, ex, "2014-09-01", "2015-12-31")
    assert expo2.max() > 0.2
