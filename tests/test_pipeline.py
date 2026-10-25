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
