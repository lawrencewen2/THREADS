"""2026-11-16：盘中止盈/止损腾出的仓位不能在同一天开盘价买回（红队发现的前视）。"""

import numpy as np
import pandas as pd

from astock.slots import simulate


class FakePanel:
    def __init__(self, o):
        self.dates = pd.bdate_range("2024-01-01", periods=o.shape[0])
        self.codes = [f"c{j}" for j in range(o.shape[1])]
        self.open = pd.DataFrame(o, index=self.dates, columns=self.codes)
        self.traded = pd.DataFrame(True, index=self.dates, columns=self.codes)

    def limit_prices(self):
        up = self.open * 10
        return up, up * 0


def setup():
    T, k = 25, 2
    o = np.full((T, k), 10.0)
    h, l = o * 1.001, o * 0.999
    h[20, 0] = 11.0  # 第 20 天股票 0 盘中冲高碰到止盈
    atr = np.full((T, k), 0.02)
    locked = np.zeros((T, k), dtype=bool)
    p = FakePanel(o)
    sc = pd.DataFrame([[2.0, 1.0]] * T, index=p.dates, columns=p.codes)
    gate = pd.Series(True, index=p.dates)
    return p, (o, h, l, atr, locked), sc, gate


def test_intraday_exit_slot_refilled_next_day_only():
    p, ohl, sc, gate = setup()
    _, trades = simulate(p, ohl, sc, gate, 1, "atr2", "atr3", 30, p.dates[15], p.dates[-1])
    entries = [t["entry"] for t in trades]
    _, old = simulate(p, ohl, sc, gate, 1, "atr2", "atr3", 30, p.dates[15], p.dates[-1], same_day_refill=True)
    st = {}
    simulate(p, ohl, sc, gate, 1, "atr2", "atr3", 30, p.dates[15], p.dates[-1], state=st)
    # 第 20 天盘中止盈卖出；修正后第 21 天才重新买入，旧口径第 20 天开盘就买回了同一只
    assert len(trades) == 1 and trades[0]["exit"] == p.dates[20]
    (entry, _, _, d0), = st["held"].values()
    assert d0 == 21
    st_old = {}
    simulate(p, ohl, sc, gate, 1, "atr2", "atr3", 30, p.dates[15], p.dates[-1], state=st_old, same_day_refill=True)
    assert list(st_old["held"].values())[0][3] == 20
    assert entries == [p.dates[15]] and len(old) == 1
