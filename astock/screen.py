"""每日开盘前选股，并跟踪过去每天选股的真实结果。"""

import json

import numpy as np
import pandas as pd

from . import config, model
from .backtest import prepare
from .factors import FACTORS

LOG_FILE = config.DATA_DIR / "picks_log.csv"
WATCH_N = 10  # 面板上列出的候选数量；只有前 top_n 只算入选


def screen(kline, basic, trade_date, m=None, valuation=None):
    """用最后一个交易日 t 收盘后的数据，给 trade_date（T）开盘买入出候选清单。"""
    m = m or model.load_model()
    recent = kline[kline["date"] >= kline["date"].max() - pd.Timedelta(days=200)]
    if valuation is not None:
        valuation = valuation[valuation["date"] >= recent["date"].min()]
    p, f, elig, ranks = prepare(recent, basic, valuation)
    sc = model.score(ranks, m["weights"], elig)
    t = p.dates[-1]
    fear = model.load_fear(p.dates)
    fear_t = fear.iloc[-1]
    fear_date = model.fear_date()
    floor = m.get("fear_floor")
    gate_open = floor is None or (not np.isnan(fear_t) and fear_t >= floor)

    row = sc.loc[t].dropna().sort_values(ascending=False).head(WATCH_N)
    names = basic.set_index("code")["code_name"]
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
        gate_open=bool(gate_open), top_n=m["top_n"], universe_size=int(elig.loc[t].sum()), picks=picks,
    )


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


def write(result, tracked):
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    sel = [t for t in tracked if t.get("selected") and "day_net" in t]
    summary = dict(
        trades=len(sel),
        hit_net=float(np.mean([t["day_net"] > 0 for t in sel])) if sel else None,
        avg_next_open_net=float(np.mean([t["next_open_net"] for t in sel if "next_open_net" in t]))
        if any("next_open_net" in t for t in sel) else None,
    )
    clean = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in t.items()} for t in tracked]
    (config.PAGE_DIR / "picks.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    (config.PAGE_DIR / "live.json").write_text(json.dumps(dict(summary=summary, rows=clean[-300:]),
                                                          ensure_ascii=False, default=str))
