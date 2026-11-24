"""出场规则研究模块 exit_study.simulate_x 的单元测试（假数据，同 tests/test_astock_slots_refill.py 的写法）。

放进仓库后模块是 astock/exit_study.py；在 /tmp/team/research2 本地跑时从同目录按 astock.exit_study 载入。"""


import numpy as np
import pandas as pd
import pytest

from astock import exit_study as ex

from astock import config  # noqa: E402
from astock.slots import simulate  # noqa: E402

ATR = 0.02  # 止盈 2×ATR = +4%，止损 3×ATR = −6%


class FakePanel:
    def __init__(self, o):
        self.dates = pd.bdate_range("2024-01-01", periods=o.shape[0])
        self.codes = [f"c{j}" for j in range(o.shape[1])]
        self.open = pd.DataFrame(o, index=self.dates, columns=self.codes)
        self.traded = pd.DataFrame(True, index=self.dates, columns=self.codes)

    def limit_prices(self):
        up = self.open * 10
        return up, up * 0


def make(o, h=None, l=None, scores=None):
    T, k = o.shape
    h = o * 1.001 if h is None else h
    l = o * 0.999 if l is None else l
    atr = np.full((T, k), ATR)
    locked = np.zeros((T, k), dtype=bool)
    p = FakePanel(o)
    row = scores if scores is not None else list(range(k, 0, -1))
    sc = pd.DataFrame([row] * T, index=p.dates, columns=p.codes, dtype=float)
    gate = pd.Series(True, index=p.dates)
    return p, (o, h, l, atr, locked), sc, gate


def run(p, ohl, sc, gate, n=1, hold=10, rule=None, gapv=None):
    return ex.simulate_x(p, ohl, sc, gate, n, "atr2", "atr3", hold, p.dates[15], p.dates[-1], rule=rule, gapv=gapv)


@pytest.fixture(autouse=True)
def std_cost():
    s0 = config.SLIPPAGE
    config.SLIPPAGE = 0.001
    yield
    config.SLIPPAGE = s0


def test_empty_rule_matches_slots_simulate():
    rng = np.random.default_rng(0)
    T, k = 120, 15
    o = 10 * np.exp(np.cumsum(rng.normal(0, 0.03, (T, k)), axis=0))
    h = o * (1 + rng.uniform(0, 0.05, (T, k)))
    l = o * (1 - rng.uniform(0, 0.05, (T, k)))
    p, ohl, _, gate = make(o, h, l)
    sc = pd.DataFrame(rng.normal(size=(T, k)), index=p.dates, columns=p.codes)
    ohl[4][50:53, 3] = True  # 一字跌停卖不掉
    gate.iloc[40:45] = False
    d0, t0 = simulate(p, ohl, sc, gate, 5, "atr2", "atr3", 10, p.dates[15], p.dates[-1])
    d1, t1, held = ex.simulate_x(p, ohl, sc, gate, 5, "atr2", "atr3", 10, p.dates[15], p.dates[-1])
    pd.testing.assert_series_equal(d0, d1)
    assert len(t0) == len(t1) > 20
    for a, b in zip(t0, t1):
        assert (a["entry"], a["exit"], a["code"]) == (b["entry"], b["exit"], b["code"])
        assert a["net"] == pytest.approx(b["net"])
    assert held.index.equals(d1.index) and held.max() <= 5


def test_trail_lets_winner_run_and_uses_only_past_highs():
    T = 40
    o = np.full((T, 2), 10.0)
    o[16:21, 0] = [10.0, 10.3, 10.6, 11.0, 11.5]  # 第 17 天起上涨，第 17 天盘中碰到 +4%
    h, l = o * 1.001, o * 0.999
    h[17, 0] = 10.45
    h[21, 0], l[21, 0], o[21, 0] = 12.5, 11.3, 11.5  # 当天冲到 12.5 又跌到 11.3：按当天最高价算的 12.25 不能当天用
    o[22:, 0] = 11.6
    h[22:, 0], l[22:, 0] = 11.62, 11.58
    p, ohl, sc, gate = make(o, h, l)
    _, fixed, _ = run(p, ohl, sc, gate)
    _, tr, _ = run(p, ohl, sc, gate, hold=20, rule=dict(trail=1.0))
    assert fixed[0]["why"] == "tp" and fixed[0]["exit"] == p.dates[17]
    assert tr[0]["why"] == "trail" and tr[0]["exit"] == p.dates[22]  # 第 22 天开盘 11.6 < 12.25，开盘卖
    assert tr[0]["net"] > fixed[0]["net"]
    # 第 21 天止损价只用到第 20 天的最高价：11.5115×0.98 = 11.28 < 当天最低 11.3，不触发
    assert all(t["exit"] != p.dates[21] for t in tr)


def test_extend_keeps_top_ranked_at_expiry():
    T = 50
    o = np.full((T, 3), 10.0)
    p, ohl, sc, gate = make(o)
    _, base, _ = run(p, ohl, sc, gate)
    _, ext, held = run(p, ohl, sc, gate, rule=dict(extend=1))
    assert base[0]["why"] == "expiry" and base[0]["exit"] == p.dates[25]
    assert len(ext) == 0  # 一直排第一，一直续持，不付卖出成本
    assert (held == 1).all()
    _, ext3, _ = run(p, ohl, sc.iloc[:, ::-1].set_axis(p.codes, axis=1), gate, rule=dict(extend=1))
    # 排第一的换成 c2：先买 c2 并续持；无论如何续持的那只不会被卖出
    assert len(ext3) == 0


def test_extend_not_in_top_sells():
    T = 40
    o = np.full((T, 3), 10.0)
    p, ohl, sc, gate = make(o)
    sc.iloc[20:, :] = [1.0, 3.0, 2.0]  # 持有的 c0 跌出前 1 名
    _, ext, _ = run(p, ohl, sc, gate, rule=dict(extend=1))
    assert ext[0]["code"] == "c0" and ext[0]["why"] == "expiry" and ext[0]["exit"] == p.dates[25]


def test_early_exit_sells_flat_position_at_open_and_refills_same_day():
    T = 30
    o = np.full((T, 2), 10.0)
    p, ohl, sc, gate = make(o)
    d, t, held = run(p, ohl, sc, gate, rule=dict(early=3))
    assert t[0]["why"] == "early" and t[0]["exit"] == p.dates[18]
    # 开盘卖出腾出的名额当天就能买（c0 当天卖掉不能买回，买 c1）
    assert t[1]["entry"] == p.dates[18] and t[1]["code"] == "c1"
    assert (held == 1).all()


def test_early_exit_keeps_profitable_position():
    T = 30
    o = np.full((T, 1), 10.0)
    o[16:, 0] = 10.2  # 有浮盈（但没碰到 +4% 止盈）
    p, ohl, sc, gate = make(o)
    _, t, _ = run(p, ohl, sc, gate, rule=dict(early=3))
    assert t[0]["why"] == "expiry"


def test_partial_take_profit_half_then_trail():
    T = 40
    o = np.full((T, 1), 10.0)
    h, l = o * 1.001, o * 0.999
    h[17, 0] = 10.25  # 碰到 1×ATR = 10.2，卖一半
    o[18:22, 0] = [10.4, 10.8, 11.0, 11.0]
    h[18:22, 0] = [10.5, 10.9, 11.1, 11.05]
    l[18:22, 0] = [10.35, 10.75, 10.95, 10.95]
    o[22:, 0], h[22:, 0], l[22:, 0] = 10.7, 10.72, 10.6  # 第 22 天开盘 10.7 < 11.1×(1−1.5×0.02)=10.767
    p, ohl, sc, gate = make(o, h, l)
    d, t, held = run(p, ohl, sc, gate, hold=20, rule=dict(partial=1.0, trail=1.5))
    (tr,) = t
    assert tr["why"] == "trail" and tr["exit"] == p.dates[22]
    bc = config.SLIPPAGE + config.COMMISSION
    scost = config.SLIPPAGE + config.COMMISSION + config.STAMP_DUTY_NEW  # 假数据在 2024 年
    exp = 0.5 * (10.2 * (1 - scost) / (10 * (1 + bc)) - 1) + 0.5 * (10.7 * (1 - scost) / (10 * (1 + bc)) - 1)
    assert tr["net"] == pytest.approx(exp)
    assert held[p.dates[17]] == 0.5 and held[p.dates[16]] == 1.0
    # 逐日收益连乘与逐笔收益一致（只有一笔，仓位 n=1）
    assert (1 + d[: p.dates[22]]).prod() - 1 == pytest.approx(exp, abs=2e-3)


def test_partial_requires_trail():
    p, ohl, sc, gate = make(np.full((30, 1), 10.0))
    with pytest.raises(ValueError):
        run(p, ohl, sc, gate, rule=dict(partial=1.0))


def test_gap_filter_skips_high_open():
    T = 30
    o = np.full((T, 2), 10.0)
    p, ohl, sc, gate = make(o)
    gapv = np.zeros((T, 2))
    gapv[15, 0] = 0.06
    _, t, _ = run(p, ohl, sc, gate, rule=dict(gap=0.05), gapv=gapv)
    assert t[0]["code"] == "c1" and t[0]["entry"] == p.dates[15]
    _, t2, _ = run(p, ohl, sc, gate, rule=dict(gap=0.07), gapv=gapv)
    assert t2[0]["code"] == "c0"


def test_breakeven_moves_stop_to_entry():
    T = 30
    o = np.full((T, 1), 10.0)
    h, l = o * 1.001, o * 0.999
    h[16, 0] = 10.25  # 碰到 +1×ATR（未到 +4% 止盈）
    l[17, 0] = 9.9  # 第二天跌回买入价以下：按买入价保本卖
    p, ohl, sc, gate = make(o, h, l)
    _, t, _ = run(p, ohl, sc, gate, rule=dict(breakeven=1.0))
    assert t[0]["why"] == "be" and t[0]["exit"] == p.dates[17]
    assert -0.005 < t[0]["net"] < 0
    _, t0, _ = run(p, ohl, sc, gate)
    assert t0[0]["why"] == "expiry"


def test_grid_is_small():
    for fam, items in ex.GRID.items():
        assert len(items) <= 8
    assert sum(len(v) for v in ex.GRID.values()) <= 35


def test_run_end_to_end_with_fake_data(monkeypatch, tmp_path):
    """run() 全流程（打分、行情、ETF 换成假的）：保证 Actions 上跑到最后不会在汇总/写文件时出错。"""
    import json

    from astock import diversify as dv

    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2019-06-03", "2021-12-31")
    T, k = len(dates), 30
    o = 10 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, (T, k)), axis=0))
    h = o * (1 + rng.uniform(0, 0.05, (T, k)))
    l = o * (1 - rng.uniform(0, 0.05, (T, k)))
    p = FakePanel(o)
    p.dates = dates
    p.open = pd.DataFrame(o, index=dates, columns=p.codes)
    p.traded = pd.DataFrame(True, index=dates, columns=p.codes)
    p.preclose = p.open.shift(1) * (1 + rng.normal(0, 0.02, (T, k)))
    atr = np.full((T, k), ATR)
    ohl = (o, h, l, atr, np.zeros((T, k), dtype=bool))
    sc = pd.DataFrame(rng.normal(size=(T, k)), index=dates, columns=p.codes)
    sm = dict(take_profit="atr2", stop_loss="atr3", n=5, sleeve=dict(codes=["a"], rebalance=20))
    monkeypatch.setattr(ex, "scores", lambda *a: (p, None, sm, dict(accel=sc)))
    monkeypatch.setattr(ex, "emotion_regime", lambda p_, e: pd.Series(rng.random(T) > 0.3, index=dates))
    monkeypatch.setattr(ex, "adjusted_ohlc", lambda p_: ohl)
    monkeypatch.setattr(dv, "etf_returns", lambda: None)
    monkeypatch.setattr(dv, "sleeve_returns", lambda *a: pd.Series(0.0002, index=dates))
    monkeypatch.setattr(ex, "OUT_DAILY", tmp_path / "d.csv.gz")
    monkeypatch.setattr(ex, "OUT_JSON", tmp_path / "x.json")
    monkeypatch.setattr(ex, "OUT_MD", tmp_path / "x.md")
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(config, "PAGE_DIR", tmp_path)
    res = ex.run(None, None)
    ex.write(res)
    md = (tmp_path / "x.md").read_text()
    assert "出场规则研究" in md and "线上规则" in md
    assert json.loads((tmp_path / "x.json").read_text())["updated"] == "2021-12-31"
    for fam in ex.GRID:
        rows = res["families"][fam]
        ch = [r for r in rows if r.get("chosen")]
        assert len(ch) == 1 and {"std", "cons"} <= set(ch[0]["test"]) and "verdict" in ch[0]
        # 保守成本确实换了滑点（含 20 日均线空仓后路径不同，随机数据上不保证更差，只要求不同）
        assert ch[0]["test"]["cons"]["stock"]["ann_return"] != ch[0]["test"]["std"]["stock"]["ann_return"]
    assert config.SLIPPAGE == 0.001
    daily = pd.read_csv(tmp_path / "d.csv.gz", index_col=0)
    assert len(daily.columns) >= 2 * len(ex.GRID)
