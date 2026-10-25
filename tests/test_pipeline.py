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
    for key, (source, _) in config.SERIES.items():
        if key in ("vix", "vstoxx", "qvix50", "move", "us_hy_oas", "eu_hy_oas"):
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


def test_crash_produces_high_fear(built):
    _, (idx, _, _) = built
    peak = idx.loc["2020-02-20":"2020-04-30", "global"].max()
    calm = idx.loc["2019-01-01":"2019-12-31", "global"].median()
    assert peak > 75
    assert calm < 60


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
    series.pop("vstoxx")
    series.pop("qvix50")
    idx, comp, _ = compute.build(series)
    assert "eu_vstoxx" not in comp.columns
    assert idx["eu"].dropna().size > 1000


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
