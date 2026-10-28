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
    monkeypatch.setattr(alt_study, "LIVE_FILE", tmp_path / "forecast_model.json")
    val = k[["date", "code"]].copy()
    val["peTTM"], val["pbMRQ"], val["psTTM"] = 15.0, 2.0, 3.0
    codes = sorted(k["code"].unique())
    ind = pd.Series(["A", "B", "C", "D"] * 10, index=codes)
    res = alt_study.run(k, basic, val, _forecast(codes[:5], "2015-04-01"), ind)
    alt_study.write(res)
    md = (tmp_path / "rep" / "alt_study.md").read_text()
    assert "高手与量化经理" in md and "业绩预期" in md
    assert {g["group"] for g in res["groups"]} == set(alt_study.GROUPS)


def test_live_forecast_portfolio(monkeypatch, tmp_path):
    from astock import screen

    k, basic = synthetic_kline(n_codes=40, start="2015-01-01", end="2016-06-30", seed=2)
    monkeypatch.setattr(config, "MIN_AMOUNT_20D", 0)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "PAGE_DIR", tmp_path / "page")
    monkeypatch.setattr(screen, "STATE_FILE", tmp_path / "data" / "holdings.json")
    monkeypatch.setattr(screen, "PORT_LOG", tmp_path / "data" / "portfolio_log.csv")
    monkeypatch.setattr(alt_study, "emotion_regime", lambda p, elig: pd.Series(True, index=p.dates))
    val = k[["date", "code"]].copy()
    val["peTTM"], val["pbMRQ"], val["psTTM"] = 15.0, 2.0, 3.0
    codes = sorted(k["code"].unique())
    fc = _forecast(codes[:8], "2016-05-01")
    fm = dict(strategy="业绩预期 + 涨停情绪开关", weights=dict(eps_chg60=0.4, fc_chg=0.3, fc_sign=0.3), n=5, buffer=2)
    dates = sorted(k["date"].unique())
    d1, d2, d3 = dates[-3], dates[-2], dates[-1]

    res1, st1 = screen.screen_forecast(k[k["date"] <= d1], basic, d2, fm, val, fc)
    assert len(res1["picks"]) == 5 and all(pk["action"] == "买入" for pk in res1["picks"])
    assert {pk["code"] for pk in res1["picks"]} <= set(codes[:8])  # 有预增预告的排前面
    screen.save_state(st1)
    screen.append_port_log(res1)
    res2, st2 = screen.screen_forecast(k[k["date"] <= d2], basic, d3, fm, val, fc)
    held2 = {pk["code"] for pk in res2["picks"]}
    assert len(held2) == 5 and {pk["code"] for pk in res2["sells"]}.isdisjoint(held2)
    assert any(pk["action"] == "继续持有" for pk in res2["picks"])
    # 同一信号日重跑，结果不变
    screen.save_state(st2)
    res2b, st2b = screen.screen_forecast(k[k["date"] <= d2], basic, d3, fm, val, fc)
    assert st2b["codes"] == st2["codes"] and st2b["prev"] == st2["prev"]
    log_df = screen.append_port_log(res2)
    rows, days = screen.track_portfolio(k, log_df, fm["n"])
    assert [d["trade_date"] for d in days] == [str(pd.Timestamp(d2).date())]
    screen.write(res2, rows, days)
    live = (tmp_path / "page" / "live.json").read_text()
    assert '"mode": "portfolio"' in live


def test_stability_runs(monkeypatch, tmp_path):
    import json

    from astock import stability

    k, basic = synthetic_kline(n_codes=40, start="2014-01-01", end="2016-06-30", seed=4)
    monkeypatch.setattr(config, "TRAIN_START", "2014-06-01")
    monkeypatch.setattr(config, "TRAIN_END", "2015-12-31")
    monkeypatch.setattr(config, "TEST_START", "2016-01-01")
    monkeypatch.setattr(config, "MIN_AMOUNT_20D", 0)
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "rep")
    monkeypatch.setattr(config, "PAGE_DIR", tmp_path / "page")
    live = tmp_path / "forecast_model.json"
    live.write_text(json.dumps(dict(weights=dict(eps_chg60=0.5, fc_chg=0.5))))
    monkeypatch.setattr(stability, "LIVE_FILE", live)
    val = k[["date", "code"]].copy()
    val["peTTM"], val["pbMRQ"], val["psTTM"] = 15.0, 2.0, 3.0
    codes = sorted(k["code"].unique())
    res = stability.run(k, basic, val, _forecast(codes[:10], "2015-04-01"))
    stability.write(res)
    assert len(res["rows"]) == len(stability.VARIANTS) + len(stability.OVERLAYS)
    assert res["chosen"] in {r["variant"] for r in res["rows"] if r["gate"]}
    assert "样本外分年收益" in (tmp_path / "rep" / "stability.md").read_text()


def test_winrate_study_runs(monkeypatch, tmp_path):
    import json

    from astock import winrate

    k, basic = synthetic_kline(n_codes=40, start="2014-01-01", end="2016-06-30", seed=5)
    monkeypatch.setattr(config, "TRAIN_START", "2014-06-01")
    monkeypatch.setattr(config, "TRAIN_END", "2015-12-31")
    monkeypatch.setattr(config, "TEST_START", "2016-01-01")
    monkeypatch.setattr(config, "MIN_AMOUNT_20D", 0)
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "rep")
    monkeypatch.setattr(config, "PAGE_DIR", tmp_path / "page")
    live = tmp_path / "forecast_model.json"
    live.write_text(json.dumps(dict(weights=dict(eps_chg60=0.5, fc_chg=0.5))))
    monkeypatch.setattr(winrate, "LIVE_FILE", live)
    monkeypatch.setattr(winrate, "emotion_regime", lambda p, elig: pd.Series(True, index=p.dates))
    val = k[["date", "code"]].copy()
    val["peTTM"], val["pbMRQ"], val["psTTM"] = 15.0, 2.0, 3.0
    codes = sorted(k["code"].unique())
    res = winrate.run(k, basic, val, _forecast(codes[:10], "2015-04-01"))
    winrate.write(res)
    r = res["rows"][0]
    assert r["test"]["trades"] > 0 and 0 <= r["test"]["win"] <= 1
    # 止盈越小胜率越高（同样的买入），每笔收益不会超过止盈减成本太多
    small = next(x for x in res["rows"] if x["take_profit"] == 0.02 and x["stop_loss"] is None and x["max_hold"] == 20 and x["top_n"] == 20)
    big = next(x for x in res["rows"] if x["take_profit"] == 0.08 and x["stop_loss"] is None and x["max_hold"] == 20 and x["top_n"] == 20)
    assert small["train"]["win"] >= big["train"]["win"]
    assert "胜率能做到多高" in (tmp_path / "rep" / "winrate.md").read_text()
