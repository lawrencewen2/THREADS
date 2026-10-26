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
    basic = pd.DataFrame({"code": k["code"].unique(), "code_name": "测试", "ipoDate": "2000-01-01",
                          "outDate": "", "type": "1", "status": "1"})
    return k, basic


@pytest.fixture(scope="module")
def synth():
    return synthetic_kline()


def test_factors_have_no_lookahead(synth):
    k, basic = synth
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
    e1, e2 = eligible(p1, basic), eligible(Panel(k2), basic)
    pd.testing.assert_frame_equal(e1.loc[:cut], e2.loc[:cut])


def test_limit_up_open_not_buyable(synth):
    k, basic = synth
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
    k, basic = synth
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
    k, basic = synth
    monkeypatch.setattr(config, "TRAIN_START", "2014-06-01")
    monkeypatch.setattr(config, "TRAIN_END", "2015-12-31")
    monkeypatch.setattr(config, "TEST_START", "2016-01-01")
    monkeypatch.setattr(config, "MIN_AMOUNT_20D", 0)
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "model.json")
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "rep")
    monkeypatch.setattr(config, "PAGE_DIR", tmp_path / "page")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(screen, "LOG_FILE", tmp_path / "data" / "log.csv")
    res = backtest.run(k, basic)
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
    r = screen.screen(k, basic, "2017-01-02", m=m)
    assert r["signal_date"] == "2016-12-30" and len(r["picks"]) == screen.WATCH_N
    log_df = screen.append_log(r)
    screen.write(r, screen.track(k, log_df))


def test_delisting_window_excluded(synth):
    k, basic = synth
    basic = basic.copy()
    code = basic["code"].iloc[0]
    basic.loc[basic["code"] == code, "outDate"] = "2015-06-30"
    e = eligible(Panel(k), basic)
    assert not e.loc["2015-06-10", code]
    assert e.loc["2015-04-01", code] or not e.loc["2015-04-01"].any()


def test_stock_codes_keeps_boards_and_delisted():
    from astock import data
    basic = pd.DataFrame({"code": ["sh.600000", "sz.300750", "sh.688981", "sz.000001", "sh.000001", "sz.200002", "sh.900901", "sz.002001"],
                          "type": ["1", "1", "1", "1", "2", "1", "1", "1"],
                          "status": ["1", "1", "1", "1", "1", "1", "1", "0"]})
    assert data.stock_codes(basic) == ["sh.600000", "sh.688981", "sz.000001", "sz.002001", "sz.300750"]
    assert "sz.002001" not in data.stock_codes(basic, listed_only=True)


def test_exit_rolls_past_suspension(synth):
    k, basic = synth
    k = k.copy()
    code = k["code"].iloc[0]
    T1 = pd.Timestamp("2015-03-05")  # T = 03-04，T+1 停牌，应在 03-06 开盘卖出
    k.loc[(k["code"] == code) & (k["date"] == T1), ["tradestatus", "volume"]] = 0
    p = Panel(k)
    out = trade_outcomes(p)
    t = pd.Timestamp("2015-03-03")
    T, T2 = pd.Timestamp("2015-03-04"), pd.Timestamp("2015-03-06")
    entry = p.open.at[T, code] * (1 + config.SLIPPAGE) * (1 + config.COMMISSION)
    sell = p.open.at[T2, code] * (1 - config.SLIPPAGE) * (1 - config.COMMISSION - config.STAMP_DUTY_OLD)
    assert out["next_open_net"].at[t, code] == pytest.approx(sell / entry - 1, rel=1e-4)


def test_hold_exit_uses_open_h_days_later(synth):
    k, basic = synth
    p = Panel(k)
    out = trade_outcomes(p)
    code = p.codes[0]
    i = p.dates.get_loc(pd.Timestamp("2015-03-03"))
    T, Th = p.dates[i + 1], p.dates[i + 1 + 5]
    entry = p.open.at[T, code] * (1 + config.SLIPPAGE) * (1 + config.COMMISSION)
    sell = p.open.at[Th, code] * (1 - config.SLIPPAGE) * (1 - config.COMMISSION - config.STAMP_DUTY_OLD)
    assert out["hold5_net"].iloc[i][code] == pytest.approx(sell / entry - 1, rel=1e-4)


def test_new_factors_have_no_lookahead(synth):
    k, basic = synth
    val = k[["date", "code"]].copy()
    val["peTTM"], val["pbMRQ"], val["psTTM"] = 15.0, 2.0, 3.0
    cut = pd.Timestamp("2015-06-30")
    f1 = compute_factors(Panel(k, val))
    assert {"ep", "bp", "sp", "beta60", "idiovol60"} <= set(f1)
    k2, v2 = k.copy(), val.copy()
    k2.loc[k2["date"] > cut, ["open", "high", "low", "close"]] *= 1.5
    v2.loc[v2["date"] > cut, "peTTM"] = -3.0
    f2 = compute_factors(Panel(k2, v2))
    for name in f1:
        pd.testing.assert_frame_equal(f1[name].loc[:cut], f2[name].loc[:cut], check_exact=False, rtol=1e-4)
