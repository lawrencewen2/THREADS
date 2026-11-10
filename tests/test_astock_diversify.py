import json

import numpy as np
import pandas as pd

from astock import config, diversify, live_sleeve


def test_park_weight_sells_at_once_and_buys_back_past_band():
    idx = pd.date_range("2020-01-01", periods=6)
    expo = pd.Series([0.0, 0.5, 0.45, 0.3, 0.3, 1.0], index=idx)
    ws = diversify.sleeve_weight_path(expo, 0.6, "park", band=0.10)
    # 0.6 x expo used by stocks; sleeve never above 1 - used
    assert np.allclose(ws, [1.0, 0.7, 0.7, 0.82, 0.82, 0.4])
    assert (ws + 0.6 * expo <= 1 + 1e-9).all()
    assert np.allclose(diversify.sleeve_weight_path(expo, 0.6, "fixed"), 0.4)


def test_sleeve_returns_equal_weight_and_cost():
    idx = pd.date_range("2020-01-01", periods=3)
    rets = pd.DataFrame({"a": [0.1, 0.0, 0.0], "b": [-0.1, 0.0, 0.0]}, index=idx)
    s = diversify.sleeve_returns(rets, ["a", "b"], rebalance=2)
    assert abs(s.iloc[0]) < 1e-12
    # day 3 rebalances drifted weights 0.55/0.45 back to 0.5 → cost 0.1 * ETF_COST
    assert abs(s.iloc[2] + 0.1 * diversify.ETF_COST) < 1e-12


def test_plan_rules():
    cfg = dict(park_band=0.10, rebalance=20)
    assert live_sleeve.plan(0.3, None, cfg, "d", 0)[0] == 0.7
    w, act, reb = live_sleeve.plan(0.5, 0.7, cfg, "d", 3)
    assert w == 0.5 and reb
    w, act, reb = live_sleeve.plan(0.45, 0.5, cfg, "d", 3)
    assert w == 0.5 and not reb
    w, act, reb = live_sleeve.plan(0.3, 0.5, cfg, "d", 3)
    assert abs(w - 0.7) < 1e-9 and reb
    assert live_sleeve.plan(0.45, 0.5, cfg, "d", 20)[2]


def test_update_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PAGE_DIR", tmp_path)
    monkeypatch.setattr(live_sleeve, "OUT_FILE", tmp_path / "sleeve.json")
    monkeypatch.setattr(live_sleeve, "STATE_FILE", tmp_path / "state.json")
    (tmp_path / "picks.json").write_text(json.dumps({"picks": []}))
    sm = dict(capital_share=0.65, n=20, sleeve=dict(codes=["sh1", "sh2"], names=["A", "B"], rebalance=20, park_band=0.1))
    res = dict(trade_date="2026-11-11", signal_date="2026-11-10", invest=True,
               picks=[dict(selected=True)] * 4 + [dict(selected=False)])
    closes = {"sh1": pd.Series([1.0, 1.1], index=pd.to_datetime(["2026-11-09", "2026-11-10"]))}
    out = live_sleeve.update(res, sm, closes=closes)
    assert abs(out["stock_used"] - 0.13) < 1e-9 and abs(out["sleeve_weight"] - 0.87) < 1e-9
    assert out["etfs"][0]["close"] == 1.1 and out["etfs"][1]["close"] is None
    # rerun same day is stable
    out2 = live_sleeve.update(res, sm, closes=closes)
    assert out2["sleeve_weight"] == out["sleeve_weight"]
    assert json.loads((tmp_path / "picks.json").read_text())["sleeve"]["sleeve_weight"] == out["sleeve_weight"]
