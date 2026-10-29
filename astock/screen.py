"""每日开盘前选股，并跟踪过去每天选股的真实结果。"""

import json

import numpy as np
import pandas as pd

from . import config, model
from .backtest import prepare
from .factors import FACTORS
from .fix_study import knife_mask, market_gate

LOG_FILE = config.DATA_DIR / "picks_log.csv"
WATCH_N = 10  # 面板上列出的候选数量；只有前 top_n 只算入选
# 2026-10-28 起线上用“业绩预期 + 涨停情绪开关”的 20 只持仓组合（见 alt_study.py）。旧的单日清单记录留在 picks_log.csv
PORT_LOG = config.DATA_DIR / "portfolio_log.csv"
STATE_FILE = config.DATA_DIR / "holdings.json"
PANEL_DAYS = 400  # 业绩预告最多生效 120 个交易日，近 60 日盈利变化要往前 60 天，所以取长一点


def screen(kline, basic, trade_date, m=None, valuation=None):
    """用最后一个交易日 t 收盘后的数据，给 trade_date（T）开盘买入出候选清单。"""
    m = m or model.load_model()
    recent = kline[kline["date"] >= kline["date"].max() - pd.Timedelta(days=200)]
    if valuation is not None:
        valuation = valuation[valuation["date"] >= recent["date"].min()]
    p, f, elig, ranks = prepare(recent, basic, valuation)
    # 2026-10-27 对照实验后加的两道筛：剔除前一天大跌、连跌 3 天、收在当日最低点附近的股票；全市场近 5 日下跌时空仓
    pool = elig & knife_mask(p) if m.get("knife_filter") else elig
    sc = model.score(ranks, m["weights"], pool)
    t = p.dates[-1]
    market = m.get("market_gate")
    market_ok = True if not market else bool(market_gate(p, elig, market).loc[t])
    fear = model.load_fear(p.dates)
    fear_t = fear.iloc[-1]
    fear_date = model.fear_date()
    floor = m.get("fear_floor")
    gate_open = (floor is None or (not np.isnan(fear_t) and fear_t >= floor)) and market_ok

    full = sc.loc[t].dropna().sort_values(ascending=False)
    row = full.head(WATCH_N)
    names = basic.set_index("code")["code_name"]
    ranking = pd.DataFrame(dict(rank=np.arange(1, len(full) + 1), code=full.index,
                                name=[names.get(c, "") for c in full.index], score=full.round(4).values))
    ranking["pct"] = (1 - (ranking["rank"] - 1) / len(full)).round(4)  # 1 = 全市场最高
    up_pct = p.limit_pct().loc[t]
    picks = []
    for rank, (code, s) in enumerate(row.items(), 1):
        close = float(p.close.at[t, code])
        picks.append(dict(
            rank=rank, code=code, name=names.get(code, ""), score=round(float(s), 4),
            selected=bool(gate_open and rank <= m["top_n"]),
            close=round(close, 2),
            limit_up=round(close * (1 + float(up_pct[code])) + 1e-6, 2),
            factors={k: (None if np.isnan(v) else round(float(v), 4))
                     for k, v in ((k, f[k].at[t, code]) for k in FACTORS if k in f) if m["weights"].get(k)},
        ))
    return dict(
        signal_date=str(t.date()), trade_date=str(pd.Timestamp(trade_date).date()),
        fear_cn=None if np.isnan(fear_t) else round(float(fear_t), 1), fear_date=fear_date, fear_floor=floor,
        gate_open=bool(gate_open), market_gate=market, market_ok=market_ok, knife_filter=bool(m.get("knife_filter")),
        top_n=m["top_n"], universe_size=int(elig.loc[t].sum()), picks=picks,
        ranking=ranking,
    )


def load_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else None


def reconcile(state, kline):
    """上一份清单在交易日 T 的实际成交：开盘涨停或停牌没买进的去掉，开盘跌停或停牌没卖掉的留下。"""
    if not state:
        return []
    T = pd.Timestamp(state["trade_date"])
    bars = kline[kline["date"] == T].set_index("code")
    held = []
    for c in state["codes"]:
        b = bars.loc[c] if c in bars.index else None
        if c in state.get("buys", []) and (b is None or b["tradestatus"] <= 0 or _at_limit(c, b, up=True)):
            continue
        held.append(c)
    for c in state.get("sells", []):
        b = bars.loc[c] if c in bars.index else None
        if b is not None and (b["tradestatus"] <= 0 or _at_limit(c, b, up=False)):
            held.append(c)
    return held


def _limit(code, preclose, up=True):
    pct = 0.2 if code.startswith(("sh.688", "sz.300", "sz.301")) else 0.1
    return round(float(preclose) * (1 + pct if up else 1 - pct) + 1e-6, 2)


def _at_limit(code, bar, up=True):
    lim = _limit(code, bar["preclose"], up)
    return float(bar["open"]) >= lim - 0.005 if up else float(bar["open"]) <= lim + 0.005


def screen_forecast(kline, basic, trade_date, fm, valuation=None, forecast=None):
    """业绩预期因子打分、情绪开关择时的持仓组合：信号日 t 收盘后出 T 日开盘要持有的 n 只。

    和回测（portfolio.simulate）同一规则：已持有的排名还在前 n×buffer 就拿着，空位按排名从前 n 里补；
    情绪开关关（近 5 日平均涨停家数不高于近 20 日平均）就全部卖出。
    """
    from .alt_study import emotion_regime, iter_alt_factors
    from .factors import Panel, eligible

    recent = kline[kline["date"] >= kline["date"].max() - pd.Timedelta(days=PANEL_DAYS)]
    if valuation is not None:
        valuation = valuation[valuation["date"] >= recent["date"].min()]
    p = Panel(recent, valuation)
    elig = eligible(p, basic)
    t = p.dates[-1]
    weights = {k: v for k, v in fm["weights"].items() if v}
    ranks, raw = {}, {}
    for k, v in iter_alt_factors(p, forecast):
        if k in weights:
            ranks[k] = model.xs_rank(v, elig)
            raw[k] = v.loc[t]
    sc = model.score(ranks, weights, elig).loc[t].dropna().sort_values(ascending=False)
    n, buffer = fm["n"], fm["buffer"]
    gate = bool(emotion_regime(p, elig).loc[t])

    state = load_state()
    if state and state["signal_date"] == str(t.date()):
        prev = state["prev"]  # 同一天重跑：从前一天的持仓重新算
    else:
        prev = reconcile(state, kline)
    rank = {c: i for i, c in enumerate(sc.index, 1)}
    keep = [c for c in prev if gate and rank.get(c, np.inf) <= n * buffer]
    buys = []
    if gate:
        for c in sc.index[:n]:
            if len(keep) + len(buys) >= n:
                break
            if c not in keep:
                buys.append(c)
    sells = [c for c in prev if c not in keep]
    hold = sorted(keep + buys, key=lambda c: rank.get(c, np.inf))

    names = basic.set_index("code")["code_name"]
    ranking = pd.DataFrame(dict(rank=np.arange(1, len(sc) + 1), code=sc.index,
                                name=[names.get(c, "") for c in sc.index], score=sc.round(4).values))
    ranking["pct"] = (1 - (ranking["rank"] - 1) / max(len(sc), 1)).round(4)
    up_pct = p.limit_pct().loc[t]

    def row(c, action):
        close = float(p.close.at[t, c])
        r = rank.get(c)
        return dict(rank=r, code=c, name=names.get(c, ""), score=None if r is None else round(float(sc[c]), 4),
                    selected=action != "卖出", action=action, close=round(close, 2),
                    limit_up=round(close * (1 + float(up_pct[c])) + 1e-6, 2),
                    factors={k: (None if np.isnan(raw[k].get(c, np.nan)) else round(float(raw[k][c]), 4)) for k in raw})

    picks = [row(c, "买入" if c in buys else "继续持有") for c in hold]
    if not gate:
        # 空仓时也列出排名前几的，方便看
        picks = [dict(row(c, "观察"), selected=False) for c in sc.index[:WATCH_N]]
    fear = model.load_fear(p.dates)
    fear_t = fear.iloc[-1]
    result = dict(
        strategy=fm["strategy"], hold_note=fm.get("hold_note"),
        signal_date=str(t.date()), trade_date=str(pd.Timestamp(trade_date).date()),
        fear_cn=None if np.isnan(fear_t) else round(float(fear_t), 1), fear_date=model.fear_date(), fear_floor=None,
        gate_open=gate, gate="情绪开关：近 5 日平均涨停家数高于近 20 日平均", top_n=n,
        universe_size=int(elig.loc[t].sum()), picks=picks,
        sells=[row(c, "卖出") for c in sells if c in p.codes],
        backtest=fm.get("backtest"), ranking=ranking,
    )
    new_state = dict(signal_date=str(t.date()), trade_date=result["trade_date"], prev=prev, codes=hold,
                     buys=buys, sells=sells)
    return result, new_state


def save_state(state):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1))


def append_port_log(result):
    rows = [dict(signal_date=result["signal_date"], trade_date=result["trade_date"], rank=pk["rank"], code=pk["code"],
                 name=pk["name"], score=pk["score"], action=pk["action"], selected=int(pk["selected"]),
                 fear_cn=result["fear_cn"]) for pk in result["picks"] + result.get("sells", [])]
    new = pd.DataFrame(rows, columns=["signal_date", "trade_date", "rank", "code", "name", "score", "action",
                                      "selected", "fear_cn"])
    if PORT_LOG.exists():
        old = pd.read_csv(PORT_LOG, dtype={"code": str})
        old = old[old["signal_date"] != result["signal_date"]]
        new = pd.concat([old, new], ignore_index=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    new.to_csv(PORT_LOG, index=False)
    return new


def track_portfolio(kline, log_df, n):
    """按实际行情算组合每天的结果：T 开盘到 T+1 开盘，新买的扣买入费，卖出的扣卖出费和印花税，每只占 1/n。"""
    if log_df is None or log_df.empty:
        return [], []
    k = kline.set_index(["code", "date"]).sort_index()
    buy_cost = config.SLIPPAGE + config.COMMISSION
    rows, days = [], []
    for T, g in log_df.groupby("trade_date", sort=True):
        T = pd.Timestamp(T)
        stamp = config.STAMP_DUTY_NEW if T >= pd.Timestamp(config.STAMP_DUTY_CHANGE) else config.STAMP_DUTY_OLD
        total, complete, cost = 0.0, True, 0.0
        for _, r in g.iterrows():
            rec = dict(r)
            try:
                bars = k.loc[r["code"]]
            except KeyError:
                bars = None
            if bars is None or T not in bars.index:
                rec.update(status="还没开盘" if bars is not None else "没有行情")
                complete = False
                rows.append(rec)
                continue
            d = bars.loc[T]
            if r["action"] == "卖出":
                if d["tradestatus"] <= 0 or _at_limit(r["code"], d, up=False):
                    rec.update(status="开盘跌停或停牌，没卖掉")
                else:
                    rec.update(status="已卖出", entry_open=float(d["open"]))
                    cost += (buy_cost + stamp) / n
                rows.append(rec)
                continue
            if r["action"] == "观察":
                rows.append(rec)
                continue
            if d["tradestatus"] <= 0 and r["action"] == "继续持有":
                rec.update(status="停牌，收益按 0 计")
                rows.append(rec)
                continue
            if d["tradestatus"] <= 0 or (r["action"] == "买入" and _at_limit(r["code"], d, up=True)):
                rec.update(status="开盘涨停或停牌，买不进")
                rows.append(rec)
                continue
            fee = buy_cost if r["action"] == "买入" else 0.0
            if r["action"] == "买入":
                cost += buy_cost / n
            rec.update(entry_open=float(d["open"]), close=float(d["close"]),
                       day_net=float(d["close"]) / float(d["open"]) - 1 - fee, status="当日收盘")
            after = bars[bars.index > T]
            if len(after):
                nxt = float(after.iloc[0]["open"])
                gross = nxt / float(d["open"]) - 1
                rec.update(next_open=nxt, next_open_net=gross - fee, status="已到 T+1 开盘")
                total += gross / n
            else:
                complete = False
            rows.append(rec)
        if complete:
            days.append(dict(trade_date=str(T.date()), net=total - cost))
    return rows, days


def append_log(result):
    rows = [dict(signal_date=result["signal_date"], trade_date=result["trade_date"], rank=pk["rank"],
                 code=pk["code"], name=pk["name"], score=pk["score"], selected=int(pk["selected"]),
                 fear_cn=result["fear_cn"]) for pk in result["picks"]]
    new = pd.DataFrame(rows)
    if LOG_FILE.exists():
        old = pd.read_csv(LOG_FILE, dtype={"code": str})
        old = old[old["signal_date"] != result["signal_date"]]
        new = pd.concat([old, new], ignore_index=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    new.to_csv(LOG_FILE, index=False)
    return new


def track(kline, log_df):
    """用实际行情给历史清单算结果：T 日开盘买入价、T 日收盘、T+1 开盘卖出，扣费口径和回测一致。"""
    if log_df is None or log_df.empty:
        return []
    k = kline.set_index(["code", "date"]).sort_index()
    out = []
    for _, r in log_df.iterrows():
        T = pd.Timestamp(r["trade_date"])
        try:
            g = k.loc[r["code"]]
        except KeyError:
            continue
        g = g[g["tradestatus"] > 0]
        if T not in g.index:
            continue
        d = g.loc[T]
        up = round(float(d["preclose"]) * (1 + (0.2 if r["code"].startswith(("sh.688", "sz.300", "sz.301")) else 0.1)) + 1e-6, 2)
        rec = dict(r)
        if float(d["open"]) >= up - 0.005:
            rec.update(status="开盘涨停，买不进")
            out.append(rec)
            continue
        entry = float(d["open"]) * (1 + config.SLIPPAGE) * (1 + config.COMMISSION)
        stamp = config.STAMP_DUTY_NEW if T >= pd.Timestamp(config.STAMP_DUTY_CHANGE) else config.STAMP_DUTY_OLD
        sell = 1 - config.SLIPPAGE
        sell *= 1 - config.COMMISSION - stamp
        rec.update(entry_open=float(d["open"]), close=float(d["close"]),
                   day_net=float(d["close"]) * sell / entry - 1, status="当日收盘")
        after = g[g.index > T]
        if len(after):
            nxt = after.iloc[0]
            rec.update(next_open=float(nxt["open"]), next_open_net=float(nxt["open"]) * sell / entry - 1,
                       status="已到 T+1 开盘")
        out.append(rec)
    return out


def rolling(days):
    """线上组合的稳定性：近 3 个月、近 12 个月收益，最大回撤，赚钱月份占比。"""
    if not days:
        return {}
    d = pd.Series({pd.Timestamp(x["trade_date"]): x["net"] for x in days}).sort_index()
    eq = (1 + d).cumprod()
    monthly = (1 + d).resample("ME").prod() - 1
    return dict(ret_63d=float((1 + d.iloc[-63:]).prod() - 1), ret_244d=float((1 + d.iloc[-244:]).prod() - 1),
                max_drawdown=float((eq / eq.cummax() - 1).min()), month_win=float((monthly > 0).mean()),
                months=int(len(monthly)))


def write(result, tracked, days=None):
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    sel = [t for t in tracked if t.get("selected") and "day_net" in t]
    if days is not None:
        nets = np.array([d["net"] for d in days])
        # 组合口径：每天 T 开盘到 T+1 开盘，换手的费用已扣
        summary = dict(mode="portfolio", days=len(nets), trades=len(sel),
                       hit_net=float((nets > 0).mean()) if len(nets) else None,
                       avg_next_open_net=float(nets.mean()) if len(nets) else None,
                       cum_net=float(np.prod(1 + nets) - 1) if len(nets) else None, daily=days[-250:],
                       **rolling(days))
    else:
        summary = dict(
            trades=len(sel),
            hit_net=float(np.mean([t["day_net"] > 0 for t in sel])) if sel else None,
            avg_next_open_net=float(np.mean([t["next_open_net"] for t in sel if "next_open_net" in t]))
            if any("next_open_net" in t for t in sel) else None,
        )
    clean = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in t.items()} for t in tracked]
    result = dict(result)
    ranking = result.pop("ranking", None)
    if ranking is not None:
        # 全部可选股票的当日得分和排名，方便查任意一只
        ranking.to_csv(config.PAGE_DIR / "scores.csv", index=False)
    (config.PAGE_DIR / "picks.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    (config.PAGE_DIR / "live.json").write_text(json.dumps(dict(summary=summary, rows=clean[-300:]),
                                                          ensure_ascii=False, default=str))
