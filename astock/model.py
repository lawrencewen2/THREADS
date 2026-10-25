"""因子打分、训练期定权重、按选股数量和恐慌指数开关回测。"""

import json

import numpy as np
import pandas as pd

from . import config
from .factors import FACTORS


def xs_rank(f, mask):
    """横截面百分位排名，居中到 [-0.5, 0.5]；不在可选范围的为 NaN。"""
    return f.where(mask).rank(axis=1, pct=True) - 0.5


def daily_ic(rank_f, target, mask):
    """逐日 Spearman IC：因子排名与收益排名的横截面相关。"""
    t = xs_rank(target, mask & target.notna())
    x = rank_f.where(t.notna())
    t = t.where(x.notna())
    x = x.sub(x.mean(axis=1), axis=0)
    t = t.sub(t.mean(axis=1), axis=0)
    num = (x * t).sum(axis=1)
    den = np.sqrt((x**2).sum(axis=1) * (t**2).sum(axis=1))
    n = x.notna().sum(axis=1)
    return (num / den).where(n >= 30)


def fit_weights(ranks, target, mask, start, end):
    """训练期内每个因子的 IC 均值和 t 值；|t| < 2 的因子权重为 0，其余按 IC 信息比率加权。"""
    stats = {}
    for name, r in ranks.items():
        ic = daily_ic(r, target, mask).loc[start:end].dropna()
        mean, std = float(ic.mean()), float(ic.std())
        t = mean / std * np.sqrt(len(ic)) if std > 0 else 0.0
        stats[name] = dict(ic_mean=mean, ic_ir=mean / std if std > 0 else 0.0, t=t, days=len(ic))
    w = {k: (v["ic_ir"] if abs(v["t"]) >= 2 else 0.0) for k, v in stats.items()}
    total = sum(abs(x) for x in w.values()) or 1.0
    return {k: v / total for k, v in w.items()}, stats


def score(ranks, weights, mask):
    total = None
    present = None
    for name, w in weights.items():
        if w == 0:
            continue
        r = ranks[name]
        part = r.fillna(0) * w
        total = part if total is None else total + part
        cnt = r.notna().astype(int)
        present = cnt if present is None else present + cnt
    need = max(1, int(0.7 * sum(1 for w in weights.values() if w != 0)))
    return total.where(mask & (present >= need))


def load_fear(dates):
    try:
        df = pd.read_csv(config.FEAR_INDEX_FILE, parse_dates=["date"]).set_index("date")
    except FileNotFoundError:
        return pd.Series(np.nan, index=dates)
    return df["cn"].reindex(dates, method="ffill", tolerance=pd.Timedelta(days=7))


def pick_mask(sc, top_n, fear=None, floor=None):
    m = sc.rank(axis=1, ascending=False, method="first") <= top_n
    if floor is not None and fear is not None:
        m = m & (fear >= floor).values[:, None]
    return m & sc.notna()


def evaluate(picks, out, start, end):
    """一组选股在一段时间内的表现。picks/out 都以信号日 t 为行。"""
    sel = picks.loc[start:end]
    buy = sel & out["buyable"].loc[start:end]
    n_days = len(sel)
    trades = int(buy.values.sum())
    if trades == 0:
        return dict(days=n_days, trade_days=0, trades=0)

    def by_trade(key):
        return out[key].loc[start:end].where(buy)

    day_ret, day_net = by_trade("day_ret"), by_trade("day_net")
    nxt_open, nxt_close = by_trade("next_open_net"), by_trade("next_close_net")
    trade_day = buy.any(axis=1)
    daily_open = nxt_open.mean(axis=1).where(trade_day, 0.0).fillna(0.0)
    daily_daynet = day_net.mean(axis=1)[trade_day]
    equity = (1 + daily_open).cumprod()
    years = max(n_days / 244, 1e-9)
    return dict(
        days=n_days,
        trade_days=int(trade_day.sum()),
        trade_day_share=float(trade_day.mean()),
        trades=trades,
        hit_gross=float((day_ret > 0).sum().sum() / trades),  # 当日收盘高于买入价
        hit_net=float((day_net > 0).sum().sum() / trades),  # 扣掉买卖费用后当日仍浮盈
        day_win_net=float((daily_daynet > 0).mean()),  # 当日组合整体扣费后浮盈的天数占比
        avg_day_net=float(np.nanmean(day_net.values)),
        avg_next_open_net=float(np.nanmean(nxt_open.values)),
        avg_next_close_net=float(np.nanmean(nxt_close.values)),
        hit_next_open_net=float((nxt_open > 0).sum().sum() / nxt_open.notna().sum().sum()),
        ann_return=float(equity.iloc[-1] ** (1 / years) - 1),
        max_drawdown=float((equity / equity.cummax() - 1).min()),
        _equity=equity,
        _daily_hit=(day_net > 0).sum(axis=1)[trade_day] / buy.sum(axis=1)[trade_day],
    )


def public(stats):
    return {k: v for k, v in stats.items() if not k.startswith("_")}


def choose(grid_train):
    """在训练期结果里挑参数：有足够交易日、平均可实现收益为正的组合中，扣费当日浮盈率最高的。"""
    ok = [g for g in grid_train if g["stats"].get("trade_day_share", 0) >= config.MIN_TRADE_DAY_SHARE]
    pos = [g for g in ok if g["stats"].get("avg_next_open_net", -1) > 0]
    pool = pos or ok or grid_train
    return max(pool, key=lambda g: g["stats"].get("hit_net", 0))


def save_model(weights, stats, chosen, extra=None):
    m = dict(weights=weights, factor_stats=stats, top_n=chosen["top_n"], fear_floor=chosen["fear_floor"],
             train=[config.TRAIN_START, config.TRAIN_END], target="T日开盘买入到T日收盘的涨幅")
    if extra:
        m.update(extra)
    config.MODEL_FILE.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    return m


def load_model():
    return json.loads(config.MODEL_FILE.read_text())


__all__ = ["FACTORS", "xs_rank", "fit_weights", "score", "load_fear", "pick_mask", "evaluate", "choose"]
