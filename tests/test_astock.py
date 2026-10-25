import numpy as np
import pandas as pd
import pytest

from astock import backtest, config, model, screen
from astock.factors import Panel, compute_factors, eligible, trade_outcomes


def synthetic_kline(n_codes=60, start="2014-01-01", end="2016-12-31", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    rows = []
    for j in range(n_codes):
        code = f"sh.{600000 + j}" if j % 2 else f"sz.{1 + j:06d}"
        close = 10.0
        for d in dates:
            pre = close
            o = round(pre * (1 + rng.normal(0, 0.01)), 2)
            close = round(max(0.5, o * (1 + rng.normal(0, 0.015))), 2)
            rows.append(dict(date=d, code=code, open=o, high=max(o, close) * 1.005, low=min(o, close) * 0.995,
                             close=close, preclose=pre, volume=1e7, amount=close * 1e7, turn=1.0 + rng.random(),
                             tradestatus=1, isST=0))
    k = pd.DataFrame(rows)
    uni = pd.DataFrame({"snapshot": "2014-01-02", "code": k["code"].unique()})
    basic = pd.DataFrame({"code": k["code"].unique(), "code_name": "测试", "ipoDate": "2000-01-01"})
    return k, uni, basic


@pytest.fixture(scope="module")
def synth():
    return synthetic_kline()


def test_factors_have_no_lookahead(synth):
    k, uni, basic = synth
    cut = pd.Timestamp("2015-06-30")
    p1 = Panel(k)
    f1 = compute_factors(p1)
    k2 = k.copy()
    future = k2["date"] > cut
    k2.loc[future, ["open", "high", "low", "close"]] *= 3  # 篡改未来行情
    f2 = compute_factors(Panel(k2))
    for name in f1:
        a, b = f1[name].loc[:cut], f2[name].loc[:cut]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-5)
    e1, e2 = eligible(p1, uni, basic), eligible(Panel(k2), uni, basic)
    pd.testing.assert_frame_equal(e1.loc[:cut], e2.loc[:cut])


def test_limit_up_open_not_buyable(synth):
    k, uni, basic = synth
    k = k.copy()
    code = k["code"].iloc[0]
    T = pd.Timestamp("2015-03-03")
    i = k.index[(k["code"] == code) & (k["date"] == T)][0]
    pre = k.at[i, "preclose"]
    k.loc[i, ["open", "high", "low", "close"]] = round(pre * 1.1 + 1e-6, 2)
    p = Panel(k)
    out = trade_outcomes(p)
    t = p.dates[p.dates.get_loc(T) - 1]
    assert not out["buyable"].at[t, code]
    assert np.isnan(out["day_net"].at[t, code])


def test_costs_make_flat_trade_negative(synth):
    k, uni, basic = synth
    k = k.copy()
    code = k["code"].iloc[0]
    T = pd.Timestamp("2015-03-04")
    m = k["code"] == code
    i = k.index[m & (k["date"] == T)][0]
    k.loc[i, ["open", "close"]] = k.at[i, "preclose"]
    p = Panel(k)
    out = trade_outcomes(p)
    t = p.dates[p.dates.get_loc(T) - 1]
    assert out["day_ret"].at[t, code] < 0  # 买入滑点
    assert -0.004 < out["day_net"].at[t, code] < -0.002  # 双边滑点+佣金+印花税约 0.3%


def test_backtest_and_screen_end_to_end(synth, monkeypatch, tmp_path):
    k, uni, basic = synth
    monkeypatch.setattr(config, "TRAIN_START", "2014-06-01")
    monkeypatch.setattr(config, "TRAIN_END", "2015-12-31")
    monkeypatch.setattr(config, "TEST_START", "2016-01-01")
    monkeypatch.setattr(config, "MIN_AMOUNT_20D", 0)
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "model.json")
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "rep")
    monkeypatch.setattr(config, "PAGE_DIR", tmp_path / "page")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(screen, "LOG_FILE", tmp_path / "data" / "log.csv")
    res = backtest.run(k, uni, basic)
    backtest.write(res)
    md = (tmp_path / "rep" / "backtest.md").read_text()
    assert "样本外" in md
    # 随机数据上不应有显著的选股能力：样本外浮盈率接近全部股票
    t, b = res["chosen"]["test"], res["baseline"]["test"]
    if t.get("trades", 0) > 100:
        assert abs(t["hit_net"] - b["hit_net"]) < 0.1
    m = model.load_model()
    if not any(m["weights"].values()):
        m["weights"]["rev5"] = 1.0
    r = screen.screen(k, uni, basic, "2017-01-02", m=m)
    assert r["signal_date"] == "2016-12-30" and len(r["picks"]) == screen.WATCH_N
    log_df = screen.append_log(r)
    screen.write(r, screen.track(k, log_df))
