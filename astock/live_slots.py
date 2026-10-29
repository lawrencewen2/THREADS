"""每日清单：业绩预期 + 情绪开关，最多 20 只，每只买入前定好止盈价、止损价和最长持有天数，
组合净值跌破 20 日均线时空仓（2026-10-29 用户在决策卡上选定；建议只用一半资金，每只占总资金 2.5%）。

和回测（slots.simulate + stability.overlay 的 trend20）是同一套规则：
- “影子组合”从 START 起按规则逐日买卖（持仓由行情重新推出来，不靠手工记录）；
- 空仓开关看影子组合到上一个开盘为止的净值是否低于 20 日均线（回测里错开两行，也就是开盘前已知）；
- 开关关的时候，实际账户卖光、不买，影子组合照常跑，开关再开时按影子组合当时的持仓买回。
"""

import json

import numpy as np
import pandas as pd

from . import config, model
from .alt_study import emotion_regime, iter_alt_factors
from .factors import Panel, eligible
from .slots import levels, simulate, stats
from .winrate import adjusted_ohlc

MODEL_FILE = config.ROOT / "astock" / "slot_model.json"
PANEL_DAYS = 400
SPARES = 5  # 补位时多列几只候补：开盘涨停买不进就按顺序换下一只


def load_model():
    return json.loads(MODEL_FILE.read_text()) if MODEL_FILE.exists() else None


def _pct(sl_or_tp, a):
    if sl_or_tp is None:
        return None
    return float(sl_or_tp[3:]) * a if isinstance(sl_or_tp, str) else float(sl_or_tp)


def screen(kline, basic, trade_date, sm, valuation=None, forecast=None):
    start = pd.Timestamp(sm["start"])
    recent = kline[kline["date"] >= min(kline["date"].max() - pd.Timedelta(days=PANEL_DAYS),
                                        start - pd.Timedelta(days=200))]
    if valuation is not None:
        valuation = valuation[valuation["date"] >= recent["date"].min()]
    p = Panel(recent, valuation)
    elig = eligible(p, basic)
    weights = {k: v for k, v in sm["weights"].items() if v}
    ranks = {}
    for k, v in iter_alt_factors(p, forecast):
        if k in weights:
            ranks[k] = model.xs_rank(v, elig)
    sc = model.score(ranks, weights, elig)
    gate = emotion_regime(p, elig)
    ohl = adjusted_ohlc(p)
    o_adj, _, _, atr, _ = ohl
    n, tp, sl, hold = sm["n"], sm["take_profit"], sm["stop_loss"], sm["max_hold"]
    dates, t = p.dates, len(p.dates) - 1

    # 空仓开关：从面板开头跑一遍影子组合，看到 t-1 行（t 开盘为止）的净值和 20 日均线
    full, _ = simulate(p, ohl, sc, gate, n, tp, sl, hold, dates[15], dates[-1])
    eq = (1 + full).cumprod()
    ma = eq.rolling(20, min_periods=10).mean()
    invest = bool(eq.iloc[-1] >= ma.iloc[-1]) if len(eq) else True

    # 实际照做的影子组合：从 START 起
    st = {}
    daily, trades = simulate(p, ohl, sc, gate, n, tp, sl, hold, start, dates[-1], state=st)
    held = st["held"]
    names = basic.set_index("code")["code_name"]
    raw_open = p.open.to_numpy()
    close_t = p.close.to_numpy()[t]
    score_t = sc.to_numpy()[t]
    rank = pd.Series(score_t, index=p.codes).dropna().rank(ascending=False, method="first")
    weight = sm.get("capital_share", 1.0) / n
    T = pd.Timestamp(trade_date)
    T_idx = len(dates)  # T 还不在面板里

    def base(j):
        c = p.codes[j]
        return dict(code=c, name=names.get(c, ""), rank=None if c not in rank.index else int(rank[c]),
                    score=None if np.isnan(score_t[j]) else round(float(score_t[j]), 4), close=round(float(close_t[j]), 2))

    picks, sells = [], []
    for j, (entry, target, stop, d0) in sorted(held.items(), key=lambda kv: kv[1][3]):
        row = base(j)
        k = raw_open[d0, j] / o_adj[d0, j]  # 复权价换回买入那天的实际价格（之后除权的话要按比例调）
        row.update(entry_date=str(dates[d0].date()), entry_price=round(float(entry * k), 2),
                   stop_price=round(float(stop * k), 2),
                   take_profit_price=None if not np.isfinite(target) else round(float(target * k), 2),
                   days_held=int(T_idx - d0), weight=weight)
        if T_idx - d0 >= hold:
            row.update(action="到期卖出", selected=False)
            sells.append(row)
        elif not invest:
            row.update(action="空仓卖出", selected=False)
            sells.append(row)
        else:
            row.update(action="继续持有", selected=True)
            picks.append(row)

    buys_open = invest and bool(gate.iloc[-1])
    # 空仓或情绪开关关时不买，但仍列出排名靠前的几只作“观察”，方便看
    free = n - len(picks) if buys_open else 0
    want = free + SPARES if buys_open else (0 if picks else SPARES * 2)
    cands = []
    if want > 0:
        traded_t = p.traded.to_numpy()[t]
        for j in np.argsort(-np.where(np.isnan(score_t), -np.inf, score_t))[:n * 3]:
            if len(cands) >= want or np.isnan(score_t[j]):
                break
            if j in held or not traded_t[j] or not np.isfinite(atr[t, j]):
                continue
            a = float(atr[t, j])
            row = base(j)
            sl_pct, tp_pct = _pct(sl, a), _pct(tp, a)
            target, stop = levels(1.0, a, tp, sl)
            act = ("买入" if len(cands) < free else "候补") if buys_open else "观察"
            row.update(action=act, selected=act == "买入", weight=weight,
                       stop_pct=round(-sl_pct, 4), take_profit_pct=None if tp_pct is None else round(tp_pct, 4),
                       stop_price_ref=round(float(close_t[j] * stop), 2),
                       take_profit_price_ref=None if tp is None else round(float(close_t[j] * target), 2))
            cands.append(row)
    picks += cands

    sel = [r for r in picks if r["selected"]]
    result = dict(
        strategy=sm["strategy"], hold_note=sm["hold_note"], signal_date=str(dates[t].date()), trade_date=str(T.date()),
        fear_cn=None, fear_date=model.fear_date(), fear_floor=None,
        gate_open=buys_open, invest=invest, emotion_on=bool(gate.iloc[-1]),
        equity_vs_ma20=None if not len(eq) else round(float(eq.iloc[-1] / ma.iloc[-1] - 1), 4),
        equity_20d=None if len(eq) < 21 else round(float(eq.iloc[-1] / eq.iloc[-21] - 1), 4),
        gate="情绪开关（近 5 日平均涨停家数高于近 20 日平均）开才买新的；组合净值跌破 20 日均线时空仓",
        top_n=n, capital_share=sm.get("capital_share", 1.0), per_name_weight=weight,
        universe_size=int(elig.iloc[-1].sum()), picks=picks, sells=sells, n_selected=len(sel),
        backtest=sm.get("backtest"),
        ranking=pd.DataFrame(dict(rank=rank.sort_values().astype(int).values, code=rank.sort_values().index,
                                  name=[names.get(c, "") for c in rank.sort_values().index],
                                  score=[round(float(sc.iloc[-1][c]), 4) for c in rank.sort_values().index])),
    )
    result["ranking"]["pct"] = (1 - (result["ranking"]["rank"] - 1) / max(len(rank), 1)).round(4)
    fear = model.load_fear(dates)
    if len(fear) and not np.isnan(fear.iloc[-1]):
        result["fear_cn"] = round(float(fear.iloc[-1]), 1)

    # 跟踪：START 以来影子组合的结果（叠加空仓开关，按一半资金折算总仓位）
    pos = (eq >= ma).astype(float).shift(2).fillna(1.0).reindex(daily.index).fillna(1.0)
    real = daily * pos - pos.diff().abs().fillna(0) * (config.SLIPPAGE * 2 + config.COMMISSION * 2 + config.STAMP_DUTY_NEW) / 2
    share = sm.get("capital_share", 1.0)
    summ = stats(real * share, trades) if len(real) else dict(days=0)
    summ.update(mode="slots", capital_share=share, closed_trades=len(trades),
                daily=[dict(trade_date=str(d.date()), net=round(float(v * share), 5)) for d, v in real.items()][-250:])
    rows = [dict(code=tr["code"], name=names.get(tr["code"], ""), entry_date=str(tr["entry"].date()),
                 exit_date=str(tr["exit"].date()), net=round(float(tr["net"]), 4),
                 status="赚" if tr["net"] > 0 else "亏") for tr in trades][-300:]
    return result, dict(summary=summ, rows=rows)


def write(result, live):
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    result = dict(result)
    ranking = result.pop("ranking", None)
    if ranking is not None:
        ranking.to_csv(config.PAGE_DIR / "scores.csv", index=False)
    (config.PAGE_DIR / "picks.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float))
    (config.PAGE_DIR / "live.json").write_text(json.dumps(live, ensure_ascii=False, default=float))
