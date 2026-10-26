import json

import numpy as np
import pandas as pd

from astock import insight
from astock.factors import Panel
from tests.test_astock import synthetic_kline


def _setup():
    k, basic = synthetic_kline(n_codes=40, start="2015-01-01", end="2016-06-30")
    val = k[["date", "code"]].copy()
    rng = np.random.default_rng(1)
    val["peTTM"] = rng.normal(20, 5, len(val)).astype("float32")
    val.loc[val["code"] == k["code"].iloc[0], "peTTM"] = -5.0
    val["pbMRQ"] = rng.normal(2, 0.3, len(val)).astype("float32")
    val["psTTM"] = 1.0
    codes = sorted(k["code"].unique())
    industry = pd.DataFrame({"code": codes, "industry": ["甲" if i % 2 else "乙" for i in range(len(codes))]})
    ranking = pd.DataFrame({"rank": range(1, len(codes) + 1), "code": codes, "name": "测试",
                            "score": np.linspace(1, 0, len(codes))})
    ranking["pct"] = 1 - (ranking["rank"] - 1) / len(codes)
    return k, basic, val, industry, ranking


def test_clean_industry():
    df = pd.DataFrame({"code": ["a", "b", "c"], "industry": ["C39计算机、通信和其他电子设备制造业", "J66货币金融服务", ""]})
    out = insight.clean_industry(df)
    assert list(out["industry"]) == ["计算机、通信和其他电子设备制造业", "货币金融服务"]


def test_temperature_counts_limits():
    k, basic, *_ = _setup()
    k = k.copy()
    last = k["date"].max()
    i = k.index[(k["date"] == last)][:3]
    k.loc[i, "close"] = (k.loc[i, "preclose"] * 1.1 + 1e-6).round(2)
    t = insight.market_temperature(Panel(k))
    row = t.iloc[-1]
    assert row["limit_up"] >= 3
    assert row["up"] + row["down"] <= k["code"].nunique()
    assert 0 <= row["temperature"] <= 100


def test_temperature_has_no_lookahead():
    k, *_ = _setup()
    cut = pd.Timestamp("2016-01-29")
    a = insight.market_temperature(Panel(k)).loc[:cut]
    k2 = k.copy()
    k2.loc[k2["date"] > cut, ["open", "high", "low", "close"]] *= 2
    b = insight.market_temperature(Panel(k2)).loc[:cut]
    pd.testing.assert_frame_equal(a, b)


def test_build_and_json(tmp_path, monkeypatch):
    k, basic, val, industry, ranking = _setup()
    insight.MIN_INDUSTRY_SIZE = 5
    picks = dict(fear_cn=18.0, top_n=3, picks=[dict(code=c) for c in ranking["code"][:3]])
    bt = dict(test_start="2015-06-01",
              chosen=dict(test=dict(avg_next_open_net=-0.002)),
              equity=[dict(date=str(d.date()), strategy=1 + 0.001 * i, baseline=1 - 0.0005 * i)
                      for i, d in enumerate(pd.bdate_range("2015-06-01", periods=60, freq="W-MON"))],
              contrarian=dict(chosen=dict(buy=20, sell=70), equity=dict(
                  strategy={str(d.date()): 1 + 0.002 * i for i, d in enumerate(pd.bdate_range("2015-06-01", periods=60, freq="W-MON"))},
                  always={str(d.date()): 1 + 0.001 * i for i, d in enumerate(pd.bdate_range("2015-06-01", periods=60, freq="W-MON"))})))
    out = insight.build(k, basic, val, industry, ranking, picks, bt)
    s = json.dumps(out, ensure_ascii=False, default=insight._clean)
    assert "NaN" not in s
    assert out["sectors"] and out["checkups"] and out["confidence"]
    first = next(r for r in out["checkups"] if r["code"] == k["code"].iloc[0])
    assert "亏损" in first["flags"]
    items = {i["item"]: i for i in out["checklist"]}
    assert items["市场情绪"]["status"] == "好"
    assert items["隔夜选股能否赚钱"]["status"] == "差"
    c = next(c for c in out["confidence"] if c["strategy"].startswith("恐慌买贪婪卖（选股）"))
    assert c["p_positive"] > 0.9 and c["p_beat_hold"] > 0.9


def test_bootstrap_interval_contains_actual():
    rng = np.random.default_rng(0)
    r = rng.normal(0.002, 0.02, 200)
    b = insight.bootstrap(r)
    assert b["ci_low"] < b["ann"] < b["ci_high"]
    assert 0 <= b["p_positive"] <= 1
