import numpy as np
import pandas as pd

from astock import alt_study, config
from astock.factors import Panel
from test_astock import synthetic_kline


def _forecast(codes, when):
    return pd.DataFrame(dict(code=codes, pub=pd.to_datetime(when), stat=pd.to_datetime(when),
                             type=["预增"] * len(codes), up=[80.0] * len(codes), dn=[40.0] * len(codes)))


def test_alt_factors_have_no_lookahead():
    k, basic = synthetic_kline(n_codes=12, start="2014-01-01", end="2015-12-31", seed=3)
    val = k[["date", "code"]].copy()
    val["peTTM"], val["pbMRQ"], val["psTTM"] = 15.0, 2.0, 3.0
    codes = sorted(k["code"].unique())
    ind = pd.Series(["A", "B"] * 6, index=codes)
    cut = pd.Timestamp("2015-06-30")
    fc = _forecast(codes[:3], "2015-03-10")
    f1 = dict(alt_study.iter_alt_factors(Panel(k, val), fc, ind))
    k2, v2 = k.copy(), val.copy()
    k2.loc[k2["date"] > cut, ["open", "high", "low", "close"]] *= 1.3
    v2.loc[v2["date"] > cut, "peTTM"] = -5.0
    fc2 = pd.concat([fc, _forecast(codes[3:6], "2015-08-01")])  # 之后才公告的预告
    f2 = dict(alt_study.iter_alt_factors(Panel(k2, v2), fc2, ind))
    assert set(alt_study.ALT_FACTORS) <= set(f1)
    for name in f1:
        pd.testing.assert_frame_equal(f1[name].loc[:cut], f2[name].loc[:cut], check_exact=False, rtol=1e-4)


def test_forecast_effective_day_after_announcement():
    k, _ = synthetic_kline(n_codes=4, start="2015-01-01", end="2015-03-31")
    p = Panel(k)
    code = p.codes[0]
    pub = p.dates[10]
    chg, sgn, fresh = alt_study.forecast_frames(p, _forecast([code], pub))
    assert np.isnan(chg.at[pub, code])
    assert chg.at[p.dates[11], code] == 60.0 and sgn.at[p.dates[11], code] == 1.0
    assert fresh.iloc[11 + 20][code] == 1.0 and np.isnan(fresh.iloc[11 + 21][code])


def test_alt_study_runs(monkeypatch, tmp_path):
    k, basic = synthetic_kline(n_codes=40, start="2014-01-01", end="2016-06-30", seed=1)
    monkeypatch.setattr(config, "TRAIN_START", "2014-06-01")
    monkeypatch.setattr(config, "TRAIN_END", "2015-12-31")
    monkeypatch.setattr(config, "TEST_START", "2016-01-01")
    monkeypatch.setattr(config, "MIN_AMOUNT_20D", 0)
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "rep")
    monkeypatch.setattr(config, "PAGE_DIR", tmp_path / "page")
    val = k[["date", "code"]].copy()
    val["peTTM"], val["pbMRQ"], val["psTTM"] = 15.0, 2.0, 3.0
    codes = sorted(k["code"].unique())
    ind = pd.Series(["A", "B", "C", "D"] * 10, index=codes)
    res = alt_study.run(k, basic, val, _forecast(codes[:5], "2015-04-01"), ind)
    alt_study.write(res)
    md = (tmp_path / "rep" / "alt_study.md").read_text()
    assert "高手与量化经理" in md and "业绩预期" in md
    assert {g["group"] for g in res["groups"]} == set(alt_study.GROUPS)
