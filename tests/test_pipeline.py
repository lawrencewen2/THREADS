import numpy as np
import pandas as pd
import pytest

from fearindex import backtest, compute, config


def synthetic_series(seed=0, start="2010-01-01", end="2020-12-31"):
    """随机游走价格，在 2015-06 和 2020-02 制造两次暴跌，波动率类序列随之放大。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    vol = np.full(n, 0.01)
    drift = np.full(n, 0.0004)
    for crash in ("2015-06-15", "2020-02-20"):
        i = idx.get_indexer([pd.Timestamp(crash)], method="nearest")[0]
        vol[i:i + 40] = 0.035
        drift[i:i + 40] = -0.008
    ret = drift + vol * rng.standard_normal(n)
    price = pd.Series(100 * np.exp(np.cumsum(ret)), index=idx)
    stress = pd.Series(vol * 1500 + rng.normal(0, 0.5, n), index=idx)
    out = {}
    for key in config.SERIES:
        if key in ("vix", "qvix50", "move"):
            out[key] = stress
        elif key == "vix3m":
            out[key] = stress.rolling(60, min_periods=1).mean()
        elif key == "a_turnover":
            out[key] = price * 1e9
        else:
            out[key] = price * (1 + 0.01 * rng.standard_normal())
    return out


@pytest.fixture(scope="module")
def built():
    series = synthetic_series()
    return series, compute.build(series)


def test_scores_in_range(built):
    _, (idx, comp, _) = built
    vals = idx.drop(columns="common").stack().dropna()
    assert vals.between(0, 100).all()
    assert set(["global", "us", "eu", "jp", "cn", "hk"]).issubset(idx.columns)


def test_crash_produces_extreme_fear(built):
    """0 = 极度恐慌：暴跌期间分数应接近 0。"""
    _, (idx, _, _) = built
    low = idx.loc["2020-02-20":"2020-04-30", "global"].min()
    calm = idx.loc["2019-01-01":"2019-12-31", "global"].median()
    assert low < 25
    assert calm > 40


def test_no_lookahead(built):
    """截断未来数据后，历史分数不变。"""
    series, (idx, _, _) = built
    cut = pd.Timestamp("2018-06-29")
    truncated = {k: s[s.index <= cut] for k, s in series.items()}
    idx2, _, _ = compute.build(truncated)
    common = idx2.index[idx2.index <= cut]
    pd.testing.assert_series_equal(idx.loc[common, "global"], idx2.loc[common, "global"], check_names=False)


def test_missing_component_degrades_gracefully():
    series = synthetic_series()
    series.pop("hstech")
    series.pop("qvix50")
    idx, comp, _ = compute.build(series)
    assert "hk_tech" not in comp.columns and "cn_qvix" not in comp.columns
    assert idx["hk"].dropna().size > 1000 and idx["cn"].dropna().size > 1000


def test_rolling_rank_bounds():
    s = pd.Series(np.arange(1000, dtype=float), index=pd.bdate_range("2000-01-01", periods=1000))
    r = compute.rolling_pct_rank(s).dropna()
    assert (r == 100).all()


def test_backtest_runs(built):
    series, (idx, _, _) = built
    report = backtest.run(idx, series)
    assert report["us"]["capture_rate"] is not None
    md = backtest.to_markdown(report, idx)
    assert "回测报告" in md and "仓位模拟" in md


def test_drawdown_episodes():
    p = pd.Series([100, 110, 90, 80, 120, 100, 130], index=pd.bdate_range("2020-01-01", periods=7))
    eps = backtest.drawdown_episodes(p, 0.15)
    assert len(eps) == 2
    assert eps[0][2] == pytest.approx(1 - 80 / 110)
    # 未回到前高，但反弹超过 25% 后再跌，算两段
    p = pd.Series([100, 60, 80, 90, 70], index=pd.bdate_range("2020-01-01", periods=5))
    eps = backtest.drawdown_episodes(p, 0.15)
    assert [round(e[2], 3) for e in eps] == [0.4, round(1 - 70 / 90, 3)]


def test_fetch_falls_back_and_refetches_on_source_change(tmp_path, monkeypatch):
    from fearindex import fetch

    monkeypatch.setattr(fetch, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fetch, "SOURCES_FILE", tmp_path / "_sources.json")
    monkeypatch.setattr(config, "SERIES", {"x": [("yahoo", "A"), ("akshare", "B")]})
    idx = pd.bdate_range("2020-01-01", periods=300)
    calls = []

    def bad(code, start):
        calls.append(("yahoo", start))
        raise ConnectionError("blocked")

    def good(code, start):
        calls.append(("akshare", start))
        return pd.Series(range(len(idx)), index=idx, dtype=float)

    monkeypatch.setattr(fetch, "FETCHERS", {"yahoo": bad, "akshare": good})
    out, errors = fetch.update_all()
    assert not errors and len(out["x"]) == 300
    # 第二次运行优先用缓存来源 akshare，且为增量拉取
    calls.clear()
    fetch.update_all()
    assert calls[0][0] == "akshare" and calls[0][1] != config.START_DATE


def test_fetch_rejects_short_history(tmp_path, monkeypatch):
    from fearindex import fetch

    monkeypatch.setattr(fetch, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fetch, "SOURCES_FILE", tmp_path / "_sources.json")
    monkeypatch.setattr(config, "SERIES", {"x": [("yahoo", "A"), ("akshare", "B")]})
    one_row = pd.Series([1.0], index=pd.bdate_range("2026-10-23", periods=1))
    full = pd.Series(1.0, index=pd.bdate_range("2020-01-01", periods=300))
    monkeypatch.setattr(fetch, "FETCHERS", {"yahoo": lambda c, s: one_row, "akshare": lambda c, s: full})
    out, _ = fetch.update_all()
    assert len(out["x"]) == 300


def test_short_series_does_not_break_build():
    series = synthetic_series()
    series["hstech"] = series["hstech"].iloc[-1:]
    idx, comp, _ = compute.build(series)
    assert "hk_tech" not in comp.columns and idx["hk"].notna().any()


def test_drift_check(built):
    from fearindex import check

    series, (idx, _, _) = built
    tampered = idx.copy()
    tampered.loc[tampered.index[100], "global"] += 5
    assert not check.run_checks(series, {}, idx, tampered)["ok"]
    assert check.run_checks(series, {}, idx, tampered, method_changed=True)["ok"]
    assert check.run_checks(series, {}, idx, idx)["ok"]


def test_region_tables(built):
    from fearindex import publish

    _, (idx, comp, _) = built
    latest = publish.latest_snapshot(idx, comp, dict(warnings=[], failures=[], ok=True))
    md = publish.region_tables(idx, comp, latest)
    for name in config.REGIONS.values():
        assert f"### {name}：" in md
    assert md.count("| 成分 | 今日 | 5日前 | 20日前 | 权重 |") == 5
    assert md.count("](docs/regions/") == 5
    svg = publish.svg_recent(idx["us"], "#2563eb")
    assert svg.count("<circle") == 21


def test_sanitize_drops_weekend_and_fx_bad_ticks():
    from fearindex import fetch

    idx = pd.bdate_range("2020-01-01", periods=10)
    fx = pd.Series(100.0, index=idx)
    fx.iloc[5] = 110.0  # 单日跳出又回到原位：坏点
    fx.loc[pd.Timestamp("2020-01-04")] = 100.0  # 周六
    fx = fx.sort_index()
    clean = fetch.sanitize("usdjpy", fx)
    assert len(clean) == 9 and clean.max() == 100.0
    # 股指的单日暴跌是真实行情，保留
    eq = pd.Series(100.0, index=idx)
    eq.iloc[5] = 88.0
    assert fetch.sanitize("spx", eq).min() == 88.0


def test_verify_recomputes_match():
    from fearindex import verify

    series = synthetic_series()
    index_df, comp_df, _ = compute.build(series)
    assert max(verify.recompute_percentiles(series, comp_df).values()) < 0.1
    tail = index_df.index[-21:]
    assert (verify.recompute_index(comp_df).loc[tail] - index_df.loc[tail]).abs().max().max() < 0.2
    _, replay = verify.realtime_replay(series, index_df)
    assert replay.max() < 0.1


def test_risk_off_needs_two_days():
    s = pd.Series([60.0] * 25 + [35.0, 50.0, 35.0, 34.0], index=pd.bdate_range("2020-01-01", periods=29))
    on = config.risk_off(s)
    assert not on.iloc[25] and not on.iloc[27] and on.iloc[28]


def test_extreme_hints():
    idx = pd.bdate_range("2020-01-01", periods=60)
    s = pd.Series([50.0] * 10 + [15.0] * 10 + [30.0, 45.0, 55.0] + [55.0] * 15 + [90.0] * 5 + [70.0, 60.0] + [60.0] * 15,
                  index=idx)
    assert config.extreme_events(s, "bottom") == [idx[22]]
    assert config.extreme_events(s, "top") == [idx[44]]
    sig = config.signals(s)
    assert sig.iloc[22] == "bottom" and sig.iloc[26] == "bottom" and sig.iloc[27] == "normal"
    assert sig.iloc[44] == "top"


def test_short_bottom_hint():
    """短期触底：10 日内下跌至少 10 分后回升 8 分，不要求跌到 20 以下。"""
    idx = pd.bdate_range("2020-01-01", periods=40)
    s = pd.Series([60.0] * 10 + [52.0, 46.0, 42.0, 40.0, 44.0, 49.0] + [49.0] * 24, index=idx)
    assert config.extreme_events(s, "bottom") == []
    assert config.extreme_events(s, "short_bottom") == [idx[15]]
    sig = config.signals(s)
    assert sig.iloc[15] == "short_bottom" and sig.iloc[19] == "short_bottom" and sig.iloc[20] == "normal"
    # 低点高于 50 的回调不算
    assert config.extreme_events(s + 15, "short_bottom") == []


def test_sanitize_drops_unfinished_today():
    from fearindex import fetch

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    idx = pd.bdate_range(end=today - pd.Timedelta(days=7), periods=5).append(pd.DatetimeIndex([today]))
    s = pd.Series(1.0, index=idx)
    out = fetch.sanitize("spx", s)
    assert today not in out.index
    assert len(out) == 5


def test_intraday_apply_quotes():
    from datetime import datetime, timezone

    from fearindex import intraday

    idx = pd.bdate_range("2026-01-01", periods=300)
    series = {"n225": pd.Series(range(1, 301), index=idx, dtype=float),
              "csi300": pd.Series(100.0, index=idx), "usdjpy": pd.Series(150.0, index=idx),
              "a_turnover": pd.Series(1e9, index=idx), "sh_margin": pd.Series(1.0, index=idx)}
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    nxt = idx[-1] + pd.offsets.BDay()
    quotes = {"n225": dict(value=250.0, time=t, date=nxt),
              "a_turnover": dict(value=5e5, time=t, date=nxt, partial=True, minutes=120),  # 万元，半天
              "csi300": dict(value=99.0, time=t, date=nxt),
              "sh_margin": dict(value=2.0, time=t, date=nxt)}
    out, fresh = intraday.apply_quotes(series, quotes)
    assert sorted(fresh) == ["a_turnover", "csi300", "n225"]
    assert out["n225"].index[-1] == nxt and out["n225"].iloc[-1] == 250.0
    assert abs(out["a_turnover"].iloc[-1] - 1e9) < 1  # 折算全天并换算到历史单位
    # 不比历史新的行情不接
    stale = {"n225": dict(value=1.0, time=t, date=idx[-1])}
    assert intraday.apply_quotes(series, stale)[1] == []
    # 开盘不到 30 分钟的成交额不折算
    early = {"a_turnover": dict(value=1e5, time=t, date=nxt, partial=True, minutes=10),
             "csi300": dict(value=99.0, time=t, date=nxt)}
    assert intraday.apply_quotes(series, early)[1] == ["csi300"]
    # 主指数还没开盘时，24 小时交易的汇率不接
    fx = {"usdjpy": dict(value=149.0, time=t, date=nxt)}
    assert intraday.apply_quotes(series, fx)[1] == []
    # 北京时间过零点后，按北京日期标注的行情不能比主指数多出一天
    late = {"n225": dict(value=250.0, time=t, date=nxt), "usdjpy": dict(value=149.0, time=t, date=nxt + pd.offsets.BDay())}
    out, fresh = intraday.apply_quotes(series, late)
    assert sorted(fresh) == ["n225", "usdjpy"] and out["usdjpy"].index[-1] == nxt


def test_intraday_realtime_parsers():
    from fearindex import intraday

    hk = ["100", "恒生指数", "HSI", "22993.070"] + ["0"] * 26 + ["2026/10/29 15:40:38"]
    t, d = intraday._tencent_time(hk)
    assert d == pd.Timestamp("2026-10-29") and t.hour == 7 and t.minute == 40  # 北京 15:40 = UTC 07:40
    a = ["1"] * 30 + ["20261029154012"]
    assert intraday._tencent_time(a)[1] == pd.Timestamp("2026-10-29")
    nky = "日经225指数,64659.2500,705.06,1.10,2:12 AM,14:12:00,2026-10-29,14:30:01,64126.25".split(",")
    q = intraday._sina_quote("b_NKY", nky)
    assert q["value"] == 64659.25 and q["date"] == pd.Timestamp("2026-10-29") and q["time"].hour == 6
    gc = "4039.133,,4039.700,4039.900,4081.700,4033.400,15:40:40,4051.600,4081.400,0,1,1,2026-10-29,纽约黄金,0".split(",")
    assert intraday._sina_quote("hf_GC", gc)["value"] == 4039.133


def test_intraday_pick_rows(monkeypatch):
    from datetime import datetime

    from fearindex import intraday

    now = datetime.now(intraday.CN_TZ).strftime("%Y%m%d%H%M%S")

    def fields(price, prev, high, low):
        f = ["0"] * 40
        f[3], f[4], f[30], f[33], f[34] = str(price), str(prev), now, str(high), str(low)
        return f

    raw = {"sz002772": fields(8.0, 8.5, 8.6, 7.9), "sh603334": fields(30.0, 28.0, 31.2, 28.0),
           "sz301317": fields(33.0, 32.6, 33.5, 32.5)}
    monkeypatch.setattr(intraday, "_tencent_raw", lambda codes, session: {c: raw[c] for c in codes if c in raw})
    doc = {"trade_date": "2026-10-29", "picks": [
        dict(code="sz.002772", name="A", action="继续持有", entry_price=8.5, stop_price=7.95, take_profit_price=9.5),
        dict(code="sh.603334", name="B", action="继续持有", entry_price=28.0, stop_price=26.0, take_profit_price=31.0),
        dict(code="sz.301317", name="C", action="观察", stop_price_ref=31.0, take_profit_price_ref=36.0),
        dict(code="sz.000001", name="D", action="观察", stop_price_ref=9.0, take_profit_price_ref=11.0)]}
    rows, notes = intraday.pick_rows(doc, session=object())
    st = {r["code"]: r["status"] for r in rows}
    assert st == {"sz.002772": "触及止损", "sh.603334": "触及止盈", "sz.301317": "正常", "sz.000001": "无行情"}
    assert rows[0]["from_entry"] == -5.88 and rows[2]["ref"] is True and rows[0]["ref"] is False
    assert [k for _, k, _ in notes] == ["触及止损", "触及止盈"]  # 观察的股票不提醒
