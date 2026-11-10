import numpy as np

from astock import t0


def bars(path):
    """48 根：每根开=高=低=收=path[i]（再加一点上下影）。"""
    c = np.array(path, dtype=float)
    return dict(open=c[None, :].copy(), high=(c + 0.01)[None, :], low=(c - 0.01)[None, :], close=c[None, :].copy())


def test_long_t_hits_target():
    path = [10.0] * 5 + [9.8] * 5 + [10.2] * 38
    arr = bars(path)
    done, net, longside = t0.simulate_t(arr, np.array([10.0]), np.array([0.02]), "正T", 1.0, 1.0, 1.0, slip=0, stamp=0)
    assert done[0] and longside[0]
    # 买在 10×0.98=9.8（开盘 9.8 不低于挂单价），止盈 9.8×1.02=9.996，下一根开盘 10.2 越过 → 按开盘价 10.2
    assert abs(net[0] - (10.2 * (1 - t0.config.COMMISSION) / (9.8 * (1 + t0.config.COMMISSION)) - 1)) < 1e-9


def test_long_t_stop_and_forced_exit():
    arr = bars([10.0] * 5 + [9.8] * 5 + [9.5] * 38)
    done, net, _ = t0.simulate_t(arr, np.array([10.0]), np.array([0.02]), "正T", 1.0, 1.0, 1.0, slip=0, stamp=0)
    assert done[0] and net[0] < -0.02  # 止损价 9.604，开盘 9.5 已跌破 → 按 9.5
    done, net, _ = t0.simulate_t(arr, np.array([10.0]), np.array([0.02]), "正T", 1.0, 1.0, None, slip=0, stamp=0)
    assert done[0] and net[0] < -0.02  # 不止损：14:55 收盘 9.5 了结


def test_short_t_and_no_fill():
    arr = bars([10.0] * 5 + [10.3] * 5 + [9.9] * 38)
    done, net, longside = t0.simulate_t(arr, np.array([10.0]), np.array([0.02]), "倒T", 1.0, 1.0, 1.0, slip=0, stamp=0)
    assert done[0] and not longside[0] and net[0] > 0  # 卖 10.3，买回 9.9
    done, net, _ = t0.simulate_t(bars([10.0] * 48), np.array([10.0]), np.array([0.02]), "双向", 1.0, 1.0, 1.0)
    assert not done[0] and net[0] == 0


def test_no_entry_after_cutoff():
    path = [10.0] * 40 + [9.0] * 8
    done, _, _ = t0.simulate_t(bars(path), np.array([10.0]), np.array([0.02]), "正T", 1.0, 1.0, 1.0)
    assert not done[0]
