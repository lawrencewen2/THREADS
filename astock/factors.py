"""把日线长表变成宽表面板，计算因子和交易结果。

约定：日期 t 的因子只用 t 日收盘及以前的数据，用来决定 t 的下一个交易日（记作 T）开盘买什么。
交易结果（T 日开盘买入后的收益）只在回测里用来打分，不进入因子。
"""

import numpy as np
import pandas as pd

from . import config

FACTORS = {
    "rev5": "近5日涨幅（短期反转）",
    "mom60": "近60日涨幅（剔除最近5日）",
    "vol20": "近20日波动率",
    "maxret20": "近20日最大单日涨幅",
    "intraday20": "近20日平均日内涨幅（开盘到收盘）",
    "overnight20": "近20日平均隔夜涨幅（昨收到今开）",
    "turn_ratio": "近5日换手率 / 近60日换手率",
    "clv": "昨日收盘在当日高低区间的位置",
    "ma20gap": "收盘价偏离20日均线",
    "ret1": "昨日涨幅",
    "size": "流通市值（对数）",
    "amount20": "近20日成交额（对数）",
    # 第二批（2026-10-26 加）
    "rev20": "近20日涨幅",
    "high250": "收盘价 / 近250日最高价",
    "skew20": "近20日收益偏度",
    "range20": "近20日平均振幅",
    "amihud20": "近20日非流动性（涨跌幅 / 成交额）",
    "turn20": "近20日平均换手率",
    "vol_spike": "昨日成交量 / 近20日均量",
    "gap1": "昨日开盘跳空",
    "limitup20": "近20日涨停次数",
    "beta60": "近60日相对大盘 beta",
    "idiovol60": "近60日特质波动率",
    "rel5": "近5日超额涨幅（减去全市场平均）",
    "ep": "盈利收益率（1 / 市盈率TTM）",
    "bp": "账面市值比（1 / 市净率）",
    "sp": "营收市值比（1 / 市销率TTM）",
    # 行为金融
    "cgo": "浮盈浮亏（处置效应）：收盘价相对换手加权的持仓成本",
    "downstreak": "连续下跌天数（过度反应）",
    "low250": "收盘价 / 近250日最低价（锚定低点）",
    "turn_rel": "昨日换手率 / 全市场中位数（散户注意力）",
}
VALUATION_FACTORS = ("ep", "bp", "sp")


class Panel:
    """各字段的宽表（行 = 交易日，列 = 股票代码）。"""

    def __init__(self, kline, valuation=None):
        k = kline[kline["code"].str[:5].isin(config.BOARD_PREFIXES)]
        self.dates = pd.DatetimeIndex(sorted(k["date"].unique()))
        self.codes = pd.Index(sorted(k["code"].unique()))

        def wide(col):
            w = k.pivot(index="date", columns="code", values=col)
            return w.reindex(index=self.dates, columns=self.codes).astype("float32")

        traded = wide("tradestatus").fillna(0) > 0
        traded &= wide("volume").fillna(0) > 0
        self.traded = traded
        self.is_st = wide("isST").fillna(0) > 0
        for col in ("open", "high", "low", "close", "preclose", "amount", "turn"):
            w = wide(col)
            if col in ("open", "high", "low", "close", "amount", "turn"):
                w = w.where(traded)
            setattr(self, col, w)
        self.volume = wide("volume").where(traded)
        self.valuation = {}
        if valuation is not None and len(valuation):
            v = valuation[valuation["code"].isin(self.codes)]
            for col in ("peTTM", "pbMRQ", "psTTM"):
                w = v.pivot(index="date", columns="code", values=col)
                self.valuation[col] = w.reindex(index=self.dates, columns=self.codes).astype("float32").where(traded)

    def limit_pct(self):
        """涨跌停幅度：科创板 20%；创业板 2020-08-24 起 20%，之前 10%；其余 10%（ST 已被剔除）。"""
        pct = pd.DataFrame(0.10, index=self.dates, columns=self.codes, dtype="float32")
        star = self.codes.str.startswith("sh.688")
        gem = self.codes.str.startswith("sz.300") | self.codes.str.startswith("sz.301")
        pct.loc[:, star] = 0.20
        pct.loc[self.dates >= pd.Timestamp("2020-08-24"), gem] = 0.20
        return pct

    def limit_prices(self):
        pct = self.limit_pct()
        up = (self.preclose * (1 + pct) + 1e-6).round(2)
        down = (self.preclose * (1 - pct) + 1e-6).round(2)
        return up, down


def compute_factors(p):
    return dict(iter_factors(p))


def iter_factors(p):
    """逐个产出 (因子名, 宽表)。全 A股时一次全放内存里太大，调用方可以算一个排一个名再丢掉。"""

    def fin(v):  # 停牌日不出因子，避免拿过期数据选股
        return v.where(p.traded).astype("float32")

    ret = (p.close / p.preclose - 1).where(p.traded)
    r0 = ret.fillna(0)
    adj = (1 + r0).cumprod()  # 复权价格指数：除权除息由 preclose 体现

    yield "rev5", fin(adj / adj.shift(5) - 1)
    yield "mom60", fin(adj.shift(5) / adj.shift(60) - 1)
    yield "vol20", fin(ret.rolling(20, min_periods=15).std())
    yield "maxret20", fin(ret.rolling(20, min_periods=15).max())
    yield "intraday20", fin((p.close / p.open - 1).rolling(20, min_periods=15).mean())
    yield "overnight20", fin((p.open / p.preclose - 1).rolling(20, min_periods=15).mean())
    turn = p.turn.where(p.turn > 0)
    yield "turn_ratio", fin(turn.rolling(5, min_periods=3).mean() / turn.rolling(60, min_periods=40).mean())
    rng = (p.high - p.low).where(p.high > p.low)
    yield "clv", fin((p.close - p.low) / rng)
    yield "ma20gap", fin(adj / adj.rolling(20, min_periods=15).mean() - 1)
    yield "ret1", fin(ret)
    float_shares = p.volume / (turn / 100)
    yield "size", fin(np.log((p.close * float_shares).ffill(limit=5)))
    yield "amount20", fin(np.log(p.amount.rolling(20, min_periods=15).mean()))

    yield "rev20", fin(adj / adj.shift(20) - 1)
    yield "high250", fin(adj / adj.rolling(250, min_periods=120).max())
    yield "skew20", fin(ret.rolling(20, min_periods=15).skew())
    yield "range20", fin(((p.high - p.low) / p.preclose).rolling(20, min_periods=15).mean())
    yield "amihud20", fin((ret.abs() / (p.amount / 1e8)).rolling(20, min_periods=15).mean())
    yield "turn20", fin(turn.rolling(20, min_periods=15).mean())
    yield "vol_spike", fin(p.volume / p.volume.rolling(20, min_periods=15).mean())
    yield "gap1", fin(p.open / p.preclose - 1)
    up, _ = p.limit_prices()
    yield "limitup20", fin((p.close >= up - 0.005).astype("float32").where(p.traded).rolling(20, min_periods=15).sum())
    mkt = ret.mean(axis=1)  # 等权全市场日收益
    rm = ret.mul(mkt, axis=0).rolling(60, min_periods=40).mean()
    r_mean = ret.rolling(60, min_periods=40).mean()
    m_mean = mkt.rolling(60, min_periods=40).mean()
    m_var = mkt.rolling(60, min_periods=40).var()
    beta = rm.sub(r_mean.mul(m_mean, axis=0)).div(m_var, axis=0)
    yield "beta60", fin(beta)
    total_var = ret.rolling(60, min_periods=40).var()
    yield "idiovol60", fin(np.sqrt((total_var - beta**2 * m_var.values[:, None]).clip(lower=0)))
    mkt_adj = (1 + mkt.fillna(0)).cumprod()
    yield "rel5", fin((adj / adj.shift(5) - 1).sub(mkt_adj / mkt_adj.shift(5) - 1, axis=0))
    # 行为金融因子
    # 持仓成本：每天按换手率把当天价格换进成本，R_t = (1 - v_t) R_{t-1} + v_t P_t（Grinblatt-Han 的递推近似）
    v = (turn / 100).clip(upper=1).fillna(0).to_numpy()
    price = adj.to_numpy()
    cost = np.empty_like(price)
    cost[0] = price[0]
    for i in range(1, len(price)):
        prev = np.where(np.isnan(cost[i - 1]), price[i], cost[i - 1])
        cost[i] = (1 - v[i]) * prev + v[i] * price[i]
    yield "cgo", fin(adj / pd.DataFrame(cost, index=adj.index, columns=adj.columns) - 1)
    del v, price, cost
    down = (ret < 0).to_numpy(dtype="float32")
    traded = p.traded.to_numpy()
    streak = np.zeros_like(down)
    for i in range(1, len(down)):  # 停牌日计数不变
        streak[i] = np.where(traded[i], (streak[i - 1] + 1) * down[i], streak[i - 1])
    yield "downstreak", fin(pd.DataFrame(streak, index=adj.index, columns=adj.columns))
    del down, traded, streak
    yield "low250", fin(adj / adj.rolling(250, min_periods=120).min())
    yield "turn_rel", fin(turn.div(turn.median(axis=1), axis=0))
    # 估值：取倒数，亏损（负值）也保留排序含义；数据缺失时这三个因子为空
    for name, col in (("ep", "peTTM"), ("bp", "pbMRQ"), ("sp", "psTTM")):
        v = p.valuation.get(col)
        if v is not None:
            yield name, fin((1 / v.where(v.abs() > 1e-6)).ffill(limit=5))


def eligible(p, basic):
    """t 日收盘后已知的可选条件：非 ST、上市满半年、不在退市整理期、流动性够、近期正常交易。

    股票池是全 A股（沪深主板、创业板、科创板），包括后来退市的股票，避免幸存者偏差。
    """
    b = basic.set_index("code").reindex(p.codes)
    ipo = pd.to_datetime(b["ipoDate"], errors="coerce")
    out = pd.to_datetime(b["outDate"], errors="coerce")
    d = p.dates.values[:, None]
    old_enough = d - ipo.values[None, :] >= pd.Timedelta(days=config.MIN_LISTED_DAYS)
    # 退市整理期（约 15 个交易日）提前公告，开盘前已知
    delisting = d >= (out - pd.Timedelta(days=config.DELIST_EXCLUDE_DAYS)).values[None, :]
    flags = pd.DataFrame(old_enough & ~delisting, index=p.dates, columns=p.codes)
    liquid = p.amount.rolling(20, min_periods=15).mean() >= config.MIN_AMOUNT_20D
    active = p.traded.rolling(60, min_periods=1).sum() >= config.MIN_TRADED_DAYS_60
    return flags & ~p.is_st & liquid & active & p.traded


def trade_outcomes(p):
    """以信号日 t 为行：T = t 的下一个交易日开盘买入的结果。

    返回 dict：
      buyable  T 日正常交易且开盘价低于涨停价（一字涨停或开盘涨停买不进）
      day_ret  T 日收盘 / 买入价 - 1，买入价含滑点，不含费用
      day_net  T 日收盘按卖出价计算的扣费收益，即“当天账面是否赚钱”（实际要到 T+1 才能卖）
      next_open_net  T+1 开盘卖出（遇停牌或跌停开盘顺延）的扣费收益，真实可实现
      next_close_net T+1 收盘卖出（遇停牌或收盘跌停顺延）的扣费收益
    """
    up, down = p.limit_prices()
    n = len(p.dates)
    stamp = np.where(p.dates >= pd.Timestamp(config.STAMP_DUTY_CHANGE),
                     config.STAMP_DUTY_NEW, config.STAMP_DUTY_OLD)
    stamp = pd.Series(stamp, index=p.dates)

    entry_ok = p.traded & (p.open < up - 0.005)
    entry = p.open * (1 + config.SLIPPAGE)
    cost_in = 1 + config.COMMISSION

    def forward_exit(price, ok):
        """从每个日期起向后找第一个可卖出的价格（最多 MAX_HOLD_SCAN 天）；返回价格和对应日期序号。"""
        val = price.where(ok).to_numpy(dtype="float32")
        out = np.full(val.shape, np.nan, dtype="float32")
        when = np.full(val.shape, -1, dtype="int32")
        for k in range(config.MAX_HOLD_SCAN, -1, -1):
            v = np.full(val.shape, np.nan, dtype="float32")
            v[: n - k] = val[k:]
            hit = ~np.isnan(v)
            out[hit] = v[hit]
            when[hit] = np.broadcast_to(np.arange(n, dtype="int32")[:, None] + k, val.shape)[hit]
        return (pd.DataFrame(out, index=p.dates, columns=p.codes),
                pd.DataFrame(when, index=p.dates, columns=p.codes))

    # T+1 开盘卖：T+1 正常交易且开盘不在跌停价
    open_ok = p.traded & (p.open > down + 0.005)
    close_ok = p.traded & (p.close > down + 0.005)
    open_exit, open_when = forward_exit(p.open, open_ok)
    close_exit, close_when = forward_exit(p.close, close_ok)

    def net(exit_px, when):
        w = when.to_numpy()
        sv = stamp.values[np.clip(w, 0, n - 1)]
        value = exit_px * (1 - config.SLIPPAGE) * (1 - config.COMMISSION - sv)
        return (value / (entry * cost_in) - 1).where(w >= 0).astype("float32")

    # 以 T 为行的结果
    day_ret_T = p.close / entry - 1
    day_net_T = (p.close * (1 - config.SLIPPAGE) * (1 - config.COMMISSION - stamp.values[:, None])) / (entry * cost_in) - 1
    next_open_T = net(open_exit.shift(-1), open_when.shift(-1, fill_value=-1))
    next_close_T = net(close_exit.shift(-1), close_when.shift(-1, fill_value=-1))

    # 平移到信号日 t（T 的前一个交易日）
    out = {
        "buyable": entry_ok.shift(-1, fill_value=False),
        "day_ret": day_ret_T.where(entry_ok).shift(-1),
        "day_net": day_net_T.where(entry_ok).shift(-1),
        "next_open_net": next_open_T.where(entry_ok).shift(-1),
        "next_close_net": next_close_T.where(entry_ok).shift(-1),
    }
    # 持有 h 个交易日：T 开盘买，T+h 开盘卖（停牌或跌停开盘顺延），用于比较持有期长短
    for h in config.HOLD_DAYS_GRID:
        held = net(open_exit.shift(-h), open_when.shift(-h, fill_value=-1))
        out[f"hold{h}_net"] = held.where(entry_ok).shift(-1)
    return out
