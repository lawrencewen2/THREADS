"""带换手缓冲的持仓组合回测。

逐日轮动：信号日 t 收盘后按得分排名，T 日开盘调仓。目标持有前 n 只；已持有的股票只要排名还在前 n×buffer
就继续拿着，跌出才卖。这样只在名单真正变化时付交易成本。每只股票占 1/n 资金，空位放现金。

T 开盘到 T+1 开盘计一天收益（复权）；停牌时价格不变，复牌那天一次性计入。
卖出遇停牌或开盘跌停就继续拿着，买入遇开盘涨停就放弃。
"""

import numpy as np
import pandas as pd

from . import config


def adjusted_open(p):
    """复权开盘价：用 close / preclose 链出的复权因子，停牌日沿用前值。"""
    ret = (p.close / p.preclose).where(p.traded).fillna(1.0)
    factor = ret.cumprod() / p.close.ffill()  # 复权价 / 实际价
    return (p.open * factor.ffill()).ffill()


def context(p):
    """模拟要用的数组，算一次给所有参数组合共用。"""
    up, down = p.limit_prices()
    return dict(dates=p.dates, codes=p.codes, open=p.open.to_numpy(), up=up.to_numpy(), down=down.to_numpy(),
                traded=p.traded.to_numpy(), adj=adjusted_open(p).to_numpy())


def simulate(ctx, sc, fear, n, buffer, floor, start, end):
    """返回 (每日组合收益 Series, 交易记录 list)。sc 以信号日 t 为行。"""
    dates = ctx["dates"]
    i0 = dates.searchsorted(pd.Timestamp(start))
    i1 = dates.searchsorted(pd.Timestamp(end), side="right")
    stamp_change = pd.Timestamp(config.STAMP_DUTY_CHANGE)
    buy_cost = config.SLIPPAGE + config.COMMISSION
    codes = ctx["codes"]
    col = {c: j for j, c in enumerate(codes)}

    sc_v = sc.to_numpy()
    open_v, up_v, down_v = ctx["open"], ctx["up"], ctx["down"]
    traded_v, adj_v = ctx["traded"], ctx["adj"]
    fear_v = fear.to_numpy() if fear is not None else np.full(len(dates), np.nan)

    holdings = {}  # code -> (entry adj open, entry date)
    daily, trades = [], []
    for i in range(max(i0, 1), min(i1, len(dates) - 1)):
        t, T = i - 1, i  # 用 t 收盘的信号，在 T 开盘交易
        row = sc_v[t]
        valid = ~np.isnan(row)
        order = np.argsort(-np.where(valid, row, -np.inf))
        n_valid = int(valid.sum())
        rank = np.full(len(codes), np.inf)
        rank[order[:n_valid]] = np.arange(1, n_valid + 1)
        gate = floor is None or (not np.isnan(fear_v[t]) and fear_v[t] >= floor)
        stamp = config.STAMP_DUTY_NEW if dates[T] >= stamp_change else config.STAMP_DUTY_OLD
        cost = 0.0

        # 卖出：跌出缓冲区或市场开关关闭；当天不能卖就继续拿
        for c in list(holdings):
            j = col[c]
            keep = gate and rank[j] <= n * buffer
            if keep:
                continue
            can_sell = traded_v[T, j] and open_v[T, j] > down_v[T, j] + 0.005
            if not can_sell:
                continue
            entry_px, entry_date = holdings.pop(c)
            gross = adj_v[T, j] / entry_px - 1
            sell_cost = config.SLIPPAGE + config.COMMISSION + stamp
            trades.append(dict(code=c, entry=entry_date, exit=dates[T],
                               net=(1 + gross) * (1 - sell_cost) / (1 + buy_cost) - 1))
            cost += sell_cost / n

        # 买入：按排名补满空位，开盘涨停或停牌的跳过
        if gate:
            for j in order[:n * 3]:
                if len(holdings) >= n:
                    break
                c = codes[j]
                if c in holdings or not valid[j] or rank[j] > n:
                    continue
                if not (traded_v[T, j] and open_v[T, j] < up_v[T, j] - 0.005):
                    continue
                holdings[c] = (adj_v[T, j], dates[T])
                cost += buy_cost / n

        # T 开盘到 T+1 开盘的持仓收益
        r = 0.0
        for c in holdings:
            j = col[c]
            a0, a1 = adj_v[T, j], adj_v[T + 1, j]
            if a0 > 0 and a1 > 0:
                r += (a1 / a0 - 1) / n
        daily.append((dates[T], r - cost))
    s = pd.Series(dict(daily), dtype="float64")
    return s, trades


def stats(daily, trades):
    if daily.empty:
        return dict(days=0)
    equity = (1 + daily).cumprod()
    years = len(daily) / 244
    ann = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = daily.std() * np.sqrt(244)
    nets = np.array([t["net"] for t in trades]) if trades else np.array([])
    hold_days = [np.busday_count(t["entry"].date(), t["exit"].date()) for t in trades]
    return dict(
        days=len(daily), ann_return=float(ann), sharpe=float(daily.mean() * 244 / vol) if vol > 0 else 0.0,
        max_drawdown=float((equity / equity.cummax() - 1).min()),
        day_win=float((daily > 0).mean()),  # 组合当天（开盘到次日开盘）赚钱的天数占比
        trades=int(len(nets)), trade_win=float((nets > 0).mean()) if len(nets) else None,
        avg_trade=float(nets.mean()) if len(nets) else None,
        avg_hold_days=float(np.mean(hold_days)) if hold_days else None,
        _equity=equity,
    )


def choose(grid):
    """训练期挑参数：年化为正的组合里夏普最高的；都不为正就取夏普最高的。"""
    pos = [g for g in grid if g["train"].get("ann_return", -1) > 0]
    return max(pos or grid, key=lambda g: g["train"].get("sharpe", -9))
