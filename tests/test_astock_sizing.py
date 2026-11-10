import numpy as np
import pandas as pd

from astock import sizing
from astock.factors import Panel, eligible
from astock.winrate import adjusted_ohlc
from test_astock import synthetic_kline
from test_astock_alt_study import _forecast


def _setup(n_codes=12, seed=4):
    k, basic = synthetic_kline(n_codes=n_codes, start="2014-01-01", end="2015-12-31", seed=seed)
    p = Panel(k)
    elig = eligible(p, basic)
    rng = np.random.default_rng(0)
    sc = pd.DataFrame(rng.random(p.close.shape), index=p.dates, columns=p.codes).where(elig)
    return p, elig, adjusted_ohlc(p), sc, pd.Series(True, index=p.dates)


def test_weights_never_exceed_full_capital():
    p, elig, ohl, sc, gate = _setup()
    mult = np.full(p.close.shape, 1.5, dtype="float32")
    d, t = sizing.simulate(p, ohl, sc, gate, 5, "atr2", "atr3", 10, "2014-09-01", "2015-12-31", mult=mult)
    assert len(t) > 0
    assert max(x["w"] for x in t) <= 1.5 + 1e-9
    d1, t1 = sizing.simulate(p, ohl, sc, gate, 5, "atr2", "atr3", 10, "2014-09-01", "2015-12-31")
    assert all(abs(x["w"] - 1) < 1e-9 for x in t1)


def test_unit_weights_match_live_simulation():
    from astock import slots
    p, elig, ohl, sc, gate = _setup(seed=6)
    d0, t0 = slots.simulate(p, ohl, sc, gate, 5, "atr2", "atr3", 10, "2014-09-01", "2015-12-31")
    d1, t1 = sizing.simulate(p, ohl, sc, gate, 5, "atr2", "atr3", 10, "2014-09-01", "2015-12-31")
    common = d0.index.intersection(d1.index)
    np.testing.assert_allclose(d0.loc[common].to_numpy(), d1.loc[common].to_numpy(), atol=1e-7)
    assert len(t0) == len(t1)


def test_trade_multipliers_have_no_lookahead():
    k, basic = synthetic_kline(n_codes=12, start="2014-01-01", end="2015-12-31", seed=3)
    val = k[["date", "code"]].copy()
    val["peTTM"], val["pbMRQ"], val["psTTM"] = 15.0, 2.0, 3.0
    cut = pd.Timestamp("2015-06-30")
    p1 = Panel(k, val)
    m1 = sizing.trade_multipliers(p1, eligible(p1, basic), adjusted_ohlc(p1)[3])
    k2, v2 = k.copy(), val.copy()
    k2.loc[k2["date"] > cut, ["open", "high", "low", "close"]] *= 1.3
    v2.loc[v2["date"] > cut, "peTTM"] = -5.0
    p2 = Panel(k2, v2)
    m2 = sizing.trade_multipliers(p2, eligible(p2, basic), adjusted_ohlc(p2)[3])
    i = p1.dates.searchsorted(cut, side="right")
    for key in m1:
        np.testing.assert_array_equal(np.nan_to_num(m1[key][:i], nan=-1), np.nan_to_num(m2[key][:i], nan=-1))
        assert np.nanmin(m1[key]) >= sizing.LO - 1e-6 and np.nanmax(m1[key]) <= sizing.HI + 1e-6


def test_matched_share_hits_target_drawdown():
    rng = np.random.default_rng(1)
    d = pd.Series(rng.normal(0.0005, 0.01, 1000), index=pd.bdate_range("2015-01-01", periods=1000))
    c = sizing.matched_share(d, -0.10)
    eq = (1 + d * c).cumprod()
    assert abs((eq / eq.cummax() - 1).min() + 0.10) < 1e-3


def test_run_end_to_end(monkeypatch):
    k, basic = synthetic_kline(n_codes=24, start="2014-01-01", end="2021-08-31", seed=7)
    codes = sorted(k["code"].unique())
    fc = pd.concat([_forecast(codes, d) for d in pd.date_range("2014-03-01", "2021-06-01", freq="QS")])
    val = k[["date", "code"]].copy()
    val["peTTM"], val["pbMRQ"], val["psTTM"] = 15.0, 2.0, 3.0
    monkeypatch.setattr(sizing, "N_RANDOM", 2)
    res = sizing.run(k, basic, val, fc)
    assert res["rows"][0]["kind"] == "live"
    assert {r["group"] for r in res["rows"]} >= {"trade", "market", "combo", "both"}
    md = sizing.to_markdown(res)
    assert "动态仓位" in md
    assert len(res["curve"]["dates"]) == len(res["curve"]["series"]["live"])
