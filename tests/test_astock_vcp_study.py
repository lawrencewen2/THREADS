import numpy as np
import pandas as pd

from astock import vcp_study
from astock.alt_study import emotion_regime
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
    f = vcp_study.features(p, elig)
    cut = p.dates[300]
    k2 = p.close.copy()
    p2 = Panel.__new__(Panel)
    p2.__dict__.update(p.__dict__)
    # 改掉 cut 之后的价格，cut 当天及之前的特征不应变化
    for name in ("close", "high", "low", "preclose"):
        x = getattr(p, name).copy()
        x.loc[x.index > cut] *= 1.5
        setattr(p2, name, x)
    f2 = vcp_study.features(p2, elig)
    for key in ("tt", "contract", "seq", "breakout", "r_vcp", "pivot"):
        a, b = f[key].loc[:cut], f2[key].loc[:cut]
        assert a.equals(b) or np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True), key
    assert k2.equals(p.close)


def test_stop_entry_matches_open_entry_when_pivot_below_open(monkeypatch):
    p, elig, sc = _setup()
    gate = pd.Series(True, index=p.dates)
    ohl = adjusted_ohlc(p)
    monkeypatch.setattr(vcp_study, "CHASE", 1e9)
    piv = pd.DataFrame(0.01, index=p.dates, columns=p.codes)
    setup = sc.notna()
    d1, t1 = simulate(p, ohl, sc, gate, 5, "atr2", "atr3", 10, "2015-01-01", "2015-12-31")
    d2, t2 = vcp_study.simulate_stop(p, ohl, sc, setup, piv, gate, 5, "atr2", "atr3", 10, "2015-01-01", "2015-12-31")
    assert len(t1) == len(t2) > 0
    assert np.allclose(d1.to_numpy(), d2.to_numpy())


def test_stop_entry_never_fills_above_high():
    p, elig, sc = _setup()
    gate = pd.Series(True, index=p.dates)
    ohl = adjusted_ohlc(p)
    piv = p.high * 10
    d, t = vcp_study.simulate_stop(p, ohl, sc, sc.notna(), piv, gate, 5, "atr2", "atr3", 10, "2015-01-01", "2015-12-31")
    assert len(t) == 0 and (d == 0).all()


def test_run_smoke(tmp_path, monkeypatch):
    k, basic = synthetic_kline(n_codes=12, start="2019-06-01", end="2021-03-31", seed=5)
    codes = sorted(k["code"].unique())
    fc = pd.DataFrame([(c, "2020-03-02", "2019-12-31", "预增", 40.0 + i, 20.0) for i, c in enumerate(codes)]
                      + [(c, "2020-08-10", "2020-06-30", "预增", 80.0 - 5 * i, 50.0) for i, c in enumerate(codes)],
                      columns=["code", "pub", "stat", "type", "up", "dn"])
    fc["pub"], fc["stat"] = pd.to_datetime(fc["pub"]), pd.to_datetime(fc["stat"])
    monkeypatch.setattr(vcp_study, "N_RANDOM", 1)
    import json
    sm = json.loads(vcp_study.MODEL_FILE.read_text())
    sm["weights"].pop("eps_chg60")  # 合成数据没有估值
    mf = tmp_path / "m.json"
    mf.write_text(json.dumps(sm))
    monkeypatch.setattr(vcp_study, "MODEL_FILE", mf)
    res = vcp_study.run(k, basic, None, fc, dump=tmp_path / "d.csv.gz")
    assert res["rows"][0]["key"] == "live"
    md = vcp_study.to_markdown(res)
    assert "VCP" in md
    out = pd.read_csv(tmp_path / "d.csv.gz", index_col="date")
    assert "live|raw" in out and "s_vcp|held" in out
