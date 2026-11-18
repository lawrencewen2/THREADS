import json

import numpy as np
import pandas as pd

from astock import kline_study as ks
from astock.factors import Panel, eligible
from test_astock import synthetic_kline


def _setup(seed=4, n=20):
    k, basic = synthetic_kline(n_codes=n, start="2014-01-01", end="2015-12-31", seed=seed)
    p = Panel(k)
    return p, eligible(p, basic)


def test_extra_factors_no_lookahead():
    p, elig = _setup()
    f = ks.extra_factors(p)
    cut = p.dates[400]
    p2 = Panel.__new__(Panel)
    p2.__dict__.update(p.__dict__)
    for name in ("open", "close", "high", "low", "preclose", "turn", "volume"):
        x = getattr(p, name).copy()
        x.loc[x.index > cut] *= 1.5
        setattr(p2, name, x)
    f2 = ks.extra_factors(p2)
    assert set(f) == {"mom120", "ma200gap", "fip", "salience20", "abturn", "turn_std20", "pv_corr20", "upvol20",
                      "lshadow20", "ideal_amp"}
    for key in f:
        a, b = f[key].loc[:cut], f2[key].loc[:cut]
        assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True, atol=1e-5), key
        assert a.notna().to_numpy().any(), key


def test_ideal_amp_matches_bruteforce():
    p, _ = _setup(n=4)
    ret = (p.close / p.preclose - 1)
    adj = (1 + ret.fillna(0)).cumprod()
    amp = (p.high - p.low) / p.preclose
    got = ks._ideal_amp(adj, amp)
    c, i = p.codes[1], 100
    w = pd.DataFrame({"a": adj[c].iloc[i - 19:i + 1], "m": amp[c].iloc[i - 19:i + 1]}).sort_values("a")
    want = w["m"].iloc[-5:].mean() - w["m"].iloc[:5].mean()
    assert abs(got[c].iloc[i] - want) < 1e-5
    assert got.iloc[:19].isna().all().all()


def test_all_ranks_cover_library_and_are_signed():
    p, elig = _setup()
    ind = pd.Series({c: ("A" if i % 2 else "B") for i, c in enumerate(p.codes)})
    r = ks.all_ranks(p, elig, ind)
    assert set(r) == set(ks.LIB)
    # 方向已经乘进去：rev20 的排名和近 20 日涨幅负相关
    ret = (p.close / p.preclose - 1)
    adj = (1 + ret.fillna(0)).cumprod()
    rev20 = (adj / adj.shift(20) - 1).where(elig)
    d = p.dates[300]
    assert np.corrcoef(r["rev20"].loc[d].dropna(), rev20.loc[d][r["rev20"].loc[d].notna()])[0, 1] < -0.9
    fam = ks.family_scores(r, elig)
    assert set(fam) == set(ks.FAMILIES)
    assert not fam["反转"].where(~elig).notna().to_numpy().any()


def test_forecast_price_uses_only_past_prices():
    p, elig = _setup()
    ret = (p.close / p.preclose - 1)
    adj = (1 + ret.fillna(0)).cumprod().astype("float32")
    c = p.codes[0]
    fc = pd.DataFrame([(c, "2015-03-02", "2014-12-31", "预增", 60.0, 40.0)], columns=["code", "pub", "stat", "type", "up", "dn"])
    fc["pub"], fc["stat"] = pd.to_datetime(fc["pub"]), pd.to_datetime(fc["stat"])
    since, react, (pos, col, good) = ks.forecast_price(p, fc, adj)
    i = p.dates.searchsorted(pd.Timestamp("2015-03-02"), side="right")
    assert pos[0] == i and good[0]
    assert np.isnan(since[c].iloc[i - 1])
    assert abs(since[c].iloc[i] - (adj[c].iloc[i] / adj[c].iloc[i - 1] - 1)) < 1e-6
    assert abs(react[c].iloc[i + 5] - (adj[c].iloc[i] / adj[c].iloc[i - 1] - 1)) < 1e-6
    assert abs(since[c].iloc[i + 5] - (adj[c].iloc[i + 5] / adj[c].iloc[i - 1] - 1)) < 1e-6


def test_run_smoke(tmp_path, monkeypatch):
    k, basic = synthetic_kline(n_codes=40, start="2019-01-01", end="2021-06-30", seed=5)
    codes = sorted(k["code"].unique())
    fc = pd.DataFrame([(c, "2020-03-02", "2019-12-31", "预增" if i % 3 else "预减", 40.0 + i, 20.0) for i, c in enumerate(codes)]
                      + [(c, "2020-08-10", "2020-06-30", "预增", 80.0 - 5 * i, 50.0) for i, c in enumerate(codes)]
                      + [(c, "2021-01-20", "2020-12-31", "预增", 30.0 + i, 10.0) for i, c in enumerate(codes)],
                      columns=["code", "pub", "stat", "type", "up", "dn"])
    fc["pub"], fc["stat"] = pd.to_datetime(fc["pub"]), pd.to_datetime(fc["stat"])
    monkeypatch.setattr(ks, "N_RANDOM", 2)
    monkeypatch.setattr(ks, "N_RANDOM_MIX", 1)
    sm = json.loads(ks.MODEL_FILE.read_text())
    sm["weights"].pop("eps_chg60")  # 合成数据没有估值
    mf = tmp_path / "m.json"
    mf.write_text(json.dumps(sm))
    monkeypatch.setattr(ks, "MODEL_FILE", mf)
    ind = pd.Series({c: ("A" if i % 2 else "B") for i, c in enumerate(codes)})
    res = ks.run(k, basic, None, fc, ind, dump=tmp_path / "d.csv.gz")
    keys = {r["key"] for r in res["rows"]}
    assert res["rows"][0]["key"] == "live" and "total" in res["rows"][0]
    assert {"k_lit", "k_all6", "k_chase", "k_lit_h20", "h_notfall", "h_nothigh", "b_lit_0.3", "m_lit_veto", "rnd_all2"} <= keys
    assert res["best_pure"] in keys and res["best_hybrid"] in keys
    assert len(res["factors"]) == len(ks.LIB) and len(res["events"]) == 22
    md = ks.to_markdown(res)
    assert "纯 K 线" in md and "单因子检验" in md and "业绩预喜公告" in md
    out = pd.read_csv(tmp_path / "d.csv.gz", index_col="date")
    assert "live|total" in out and "k_lit|raw" in out and "rnd_all1|raw" not in out
    json.dumps(res, ensure_ascii=False, default=float)
