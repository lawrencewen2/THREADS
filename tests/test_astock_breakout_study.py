import json

import numpy as np
import pandas as pd

from astock import breakout_study as bs
from astock.factors import Panel, eligible
from astock.slots import simulate
from astock.winrate import adjusted_ohlc
from test_astock import synthetic_kline


def _setup(seed=4):
    k, basic = synthetic_kline(n_codes=20, start="2014-01-01", end="2015-12-31", seed=seed)
    p = Panel(k)
    elig = eligible(p, basic)
    sc = pd.DataFrame(np.random.default_rng(seed).random(elig.shape), index=p.dates, columns=p.codes).where(elig)
    return p, elig, sc


def test_features_no_lookahead():
    p, elig, _ = _setup()
    f = bs.features(p, elig)
    cut = p.dates[300]
    p2 = Panel.__new__(Panel)
    p2.__dict__.update(p.__dict__)
    for name in ("close", "high", "low", "preclose"):
        x = getattr(p, name).copy()
        x.loc[x.index > cut] *= 1.5
        setattr(p2, name, x)
    f2 = bs.features(p2, elig)
    for key in f:
        a, b = f[key].loc[:cut], f2[key].loc[:cut]
        assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True), key


def test_breakout_closes_above_prior_platform_high():
    p, elig, _ = _setup()
    f = bs.features(p, elig)
    ret = (p.close / p.preclose - 1).where(p.traded)
    adj = (1 + ret.fillna(0)).cumprod()
    hi = p.high * adj / p.close
    h_prev = hi.shift(1).rolling(40, min_periods=32).max()
    b = f["brk40"]
    assert b.to_numpy().any()
    assert (adj.where(b) > h_prev.where(b)).to_numpy()[b.to_numpy()].all()
    assert not (f["brk40_vol"] & ~b).to_numpy().any()
    assert not (f["setup40"] & ~f["plat40"]).to_numpy().any()
    assert not (f["brk40_strong"] & f["brk40_weak"]).to_numpy().any()
    assert not (f["held10"] & f["failed10"]).to_numpy().any()
    # 站稳 3 天：3 天前突破，之后 3 天收盘都高于当时的平台高点
    c = f["confirm3"].to_numpy()
    lvl = h_prev.shift(3).to_numpy()
    for s in (0, 1, 2):
        assert (adj.shift(s).to_numpy()[c] > lvl[c]).all()


def test_stop_entry_matches_open_entry_when_pivot_below_open(monkeypatch):
    p, elig, sc = _setup()
    gate = pd.Series(True, index=p.dates)
    ohl = adjusted_ohlc(p)
    monkeypatch.setattr(bs, "CHASE", 1e9)
    piv = pd.DataFrame(0.01, index=p.dates, columns=p.codes)
    # 买单是开盘前挂好的，买不到（停牌、开盘涨停）时不会像开盘买那样顺延到下一只；
    # 测试里先把第二天买不到的候选去掉，两种模拟就应该完全一样
    up, _ = p.limit_prices()
    buyable = (p.traded & (p.open < up - 0.005)).shift(-1, fill_value=False)
    sc = sc.where(buyable)
    d1, t1 = simulate(p, ohl, sc, gate, 5, "atr2", "atr3", 10, "2015-01-01", "2015-12-31")
    d2, t2 = bs.simulate_stop(p, ohl, sc, sc.notna(), piv, gate, 5, "atr2", "atr3", 10, "2015-01-01", "2015-12-31")
    assert len(t1) == len(t2) > 0
    assert np.allclose(d1.to_numpy(), d2.to_numpy())


def test_stop_entry_never_fills_above_high():
    p, elig, sc = _setup()
    gate = pd.Series(True, index=p.dates)
    ohl = adjusted_ohlc(p)
    d, t = bs.simulate_stop(p, ohl, sc, sc.notna(), p.high * 10, gate, 5, "atr2", "atr3", 10, "2015-01-01", "2015-12-31")
    assert len(t) == 0 and (d == 0).all()


def test_run_smoke(tmp_path, monkeypatch):
    k, basic = synthetic_kline(n_codes=12, start="2019-06-01", end="2021-03-31", seed=5)
    codes = sorted(k["code"].unique())
    fc = pd.DataFrame([(c, "2020-03-02", "2019-12-31", "预增", 40.0 + i, 20.0) for i, c in enumerate(codes)]
                      + [(c, "2020-08-10", "2020-06-30", "预增", 80.0 - 5 * i, 50.0) for i, c in enumerate(codes)],
                      columns=["code", "pub", "stat", "type", "up", "dn"])
    fc["pub"], fc["stat"] = pd.to_datetime(fc["pub"]), pd.to_datetime(fc["stat"])
    monkeypatch.setattr(bs, "N_RANDOM", 1)
    sm = json.loads(bs.MODEL_FILE.read_text())
    sm["weights"].pop("eps_chg60")  # 合成数据没有估值
    mf = tmp_path / "m.json"
    mf.write_text(json.dumps(sm))
    monkeypatch.setattr(bs, "MODEL_FILE", mf)
    res = bs.run(k, basic, None, fc, dump=tmp_path / "d.csv.gz")
    assert res["rows"][0]["key"] == "live" and "total" in res["rows"][0]
    assert len(res["cons"]) == 4
    md = bs.to_markdown(res)
    assert "平台突破" in md and "真假突破统计" in md
    assert {"t_confirm3", "r_retest20"} <= {r["key"] for r in res["rows"]} and len(res["stats"]) == 18
    out = pd.read_csv(tmp_path / "d.csv.gz", index_col="date")
    assert "live|raw" in out and "s_plat40|held" in out
