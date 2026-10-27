import numpy as np
import pandas as pd

from astock import config, fix_study
from astock.factors import Panel
from test_astock import synthetic_kline


def test_knife_mask_flags_falling_stock():
    k, _ = synthetic_kline(n_codes=4, start="2015-01-01", end="2015-02-27")
    code = k["code"].iloc[0]
    rows = k.index[k["code"] == code][-4:]
    pre = k.at[rows[0], "preclose"]
    for i, r in enumerate(rows):  # 最后 4 天每天跌 3%，收在最低价
        c = round(pre * 0.97 ** (i + 1), 2)
        k.loc[r, ["preclose", "open", "high", "low", "close"]] = [round(pre * 0.97 ** i, 2), c * 1.01, c * 1.02, c, c]
    p = Panel(k)
    ok = fix_study.knife_mask(p)
    assert not ok.at[p.dates[-1], code]


def test_fix_study_runs_and_gates(monkeypatch, tmp_path):
    k, basic = synthetic_kline(n_codes=40, start="2014-01-01", end="2016-06-30", seed=1)
    monkeypatch.setattr(config, "TRAIN_START", "2014-06-01")
    monkeypatch.setattr(config, "TRAIN_END", "2015-12-31")
    monkeypatch.setattr(config, "TEST_START", "2016-01-01")
    monkeypatch.setattr(config, "MIN_AMOUNT_20D", 0)
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "rep")
    monkeypatch.setattr(config, "PAGE_DIR", tmp_path / "page")
    res = fix_study.run(k, basic)
    fix_study.write(res)
    assert (tmp_path / "rep" / "fix_study.md").exists()
    for r in res["rows"]:
        share = r["test"].get("trade_day_share", 0)
        if r["gate"] == "none":
            assert share > 0.9
        assert 0 <= share <= 1
    assert res["chosen"] in res["rows"]
    assert np.isfinite(res["current"]["test"].get("avg_next_open_net", np.nan))
