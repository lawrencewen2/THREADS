import numpy as np
import pandas as pd

from astock import jlaw
from astock.factors import Panel, eligible
from astock.winrate import adjusted_ohlc
from test_astock import synthetic_kline
from test_astock_alt_study import _forecast


def _ind(p):
    return {c: "AB"[i % 2] for i, c in enumerate(p.codes)}


def test_features_have_no_lookahead():
    k, basic = synthetic_kline(n_codes=20, start="2014-01-01", end="2016-06-30", seed=5)
    cut = pd.Timestamp("2015-12-31")
    p1 = Panel(k)
    f1 = jlaw.features(p1, eligible(p1, basic), _ind(p1))
    k2 = k.copy()
    k2.loc[k2["date"] > cut, ["open", "high", "low", "close"]] *= 1.5
    k2.loc[k2["date"] > cut, ["volume", "turn"]] *= 3
    p2 = Panel(k2)
    f2 = jlaw.features(p2, eligible(p2, basic), _ind(p2))
    for name in ("stage2", "leader", "pull", "meta", "above200", "ind_ok"):
        pd.testing.assert_frame_equal(f1[name].loc[:cut], f2[name].loc[:cut])
    for name in f1["markets"]:
        pd.testing.assert_series_equal(f1["markets"][name].loc[:cut], f2["markets"][name].loc[:cut])


def test_risk_sizing_and_weak_exit():
    k, basic = synthetic_kline(n_codes=12, start="2014-01-01", end="2015-12-31", seed=2)
    p = Panel(k)
    elig = eligible(p, basic)
    ohl = adjusted_ohlc(p)
    sc = pd.DataFrame(np.random.default_rng(0).random(p.close.shape), index=p.dates, columns=p.codes).where(elig)
    gate = pd.Series(True, index=p.dates)
    d, t, ex = jlaw.simulate(p, ohl, sc, gate, "2014-09-01", "2015-12-31", n=40, sizing=0.003)
    assert ex.max() <= 1.0 + 1e-9 and ex.max() > 0
    d1, t1, ex1 = jlaw.simulate(p, ohl, sc, gate, "2014-09-01", "2015-12-31", n=5)
    assert abs(ex1.max() - 1.0) < 1e-9
    weak = pd.DataFrame(True, index=p.dates, columns=p.codes)
    _, t2, _ = jlaw.simulate(p, ohl, sc, gate, "2014-09-01", "2015-12-31", n=5, target=None, weak=weak, hold=50)
    days = [np.busday_count(x["entry"].date(), x["exit"].date()) for x in t2]
    assert max(days) <= 3  # 第二天开盘就卖（停牌顺延除外）


def test_equal_sizing_matches_live_simulator():
    from astock import slots
    k, basic = synthetic_kline(n_codes=15, start="2014-01-01", end="2015-12-31", seed=8)
    p = Panel(k)
    elig = eligible(p, basic)
    ohl = adjusted_ohlc(p)
    sc = pd.DataFrame(np.random.default_rng(1).random(p.close.shape), index=p.dates, columns=p.codes).where(elig)
    gate = pd.Series(True, index=p.dates)
    a, ta = slots.simulate(p, ohl, sc, gate, 5, "atr2", "atr3", 10, "2014-09-01", "2015-12-31")
    b, tb, _ = jlaw.simulate(p, ohl, sc, gate, "2014-09-01", "2015-12-31", n=5)
    assert len(ta) == len(tb)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-6)


def test_run_end_to_end(monkeypatch):
    k, basic = synthetic_kline(n_codes=24, start="2014-01-01", end="2021-08-31", seed=7)
    codes = sorted(k["code"].unique())
    fc = pd.concat([_forecast(codes, d) for d in pd.date_range("2014-03-01", "2021-06-01", freq="QS")])
    monkeypatch.setattr(jlaw, "META_K", (3,))
    monkeypatch.setattr(jlaw, "EXITS", (("ma10", None),))
    monkeypatch.setattr(jlaw, "RISKS", (0.003,))
    monkeypatch.setattr(jlaw, "MARKETS", ("none",))
    val = k[["date", "code"]].copy()
    val["peTTM"], val["pbMRQ"], val["psTTM"] = 15.0, 2.0, 3.0
    res = jlaw.run(k, basic, val, fc, {c: "AB"[i % 2] for i, c in enumerate(codes)})
    assert res["rows"][0]["name"] == "线上规则（对照）"
    assert any(r["group"] == "part" for r in res["rows"])
    assert "J Law" in jlaw.to_markdown(res)
