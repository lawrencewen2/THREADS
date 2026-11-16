"""实盘留痕账本（2026-11-16 量化小组：先自用，以后可能对外，业绩记录要能经得起查）。

docs/astock/ledger/ 下四个 CSV，只追加、不修改已写的行（git 历史就是时间戳证明）：
- signals.csv：每天开盘前的清单（买入 / 继续持有 / 卖出，含止盈价、止损价），logged_at 是写入时间（UTC），
  证明信号在开盘前就定好了；
- plan.csv：每天开盘前定下的资金安排（股票占用、ETF 分仓比例、是否空仓、规则版本）；
- nav.csv：模型组合的逐日收益和净值。选股部分取 live.json 当天第一次出现的数（开盘到下一个开盘，已按资金比例折算），
  ETF 分仓部分用三只 ETF 收盘到收盘等权收益 × 当天 plan.csv 里的分仓比例。一天的数一旦写入就不再改，
  即使以后回测口径或数据修正让 live.json 的历史变了；
- trades.csv：影子组合（按规则逐日买卖）每笔平仓交易，第一次出现时记下。

这是“模型组合”的记录，不是券商账户的真实成交；以后要对外展示，还需要把券商对账单的真实净值记进来对照。
"""

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone

from . import config

log = logging.getLogger(__name__)

LEDGER_DIR = config.PAGE_DIR / "ledger"
FILES = {
    "signals": ["trade_date", "code", "name", "action", "weight", "entry_price", "ref_close", "stop_price",
                "take_profit_price", "rule_version", "logged_at"],
    "plan": ["trade_date", "invest", "gate_open", "n_selected", "stock_used", "sleeve_weight", "rule_version",
             "source", "logged_at"],
    "nav": ["date", "stock_net", "sleeve_weight", "sleeve_ret", "total_net", "nav", "rule_version", "source",
            "logged_at"],
    "trades": ["code", "name", "entry_date", "exit_date", "net", "kind", "logged_at"],
}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rule_version(sm):
    """规则版本 = slot_model.json 里决定交易的字段的短哈希，改了规则就会变。"""
    keys = ("weights", "blend", "n", "take_profit", "stop_loss", "max_hold", "overlay", "capital_share", "sleeve")
    blob = json.dumps({k: sm.get(k) for k in keys}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def read(name, d=None):
    path = (d or LEDGER_DIR) / f"{name}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append(name, rows, d=None):
    """只追加。文件不存在时先写表头。"""
    if not rows:
        return 0
    d = d or LEDGER_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FILES[name], extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def _r(x, k=6):
    return "" if x is None else round(float(x), k)


def log_signals(result, version, d=None, ts=None):
    td = result["trade_date"]
    if any(r["trade_date"] == td for r in read("signals", d)):
        return 0  # 同一天重跑不重复记（以第一次、开盘前的为准）
    ts = ts or now()
    rows = []
    for r in result.get("picks", []) + result.get("sells", []):
        if r.get("action") not in ("买入", "继续持有", "到期卖出", "空仓卖出"):
            continue
        rows.append(dict(trade_date=td, code=r["code"], name=r.get("name", ""), action=r["action"],
                         weight=_r(r.get("weight"), 4), entry_price=_r(r.get("entry_price"), 3),
                         ref_close=_r(r.get("close"), 3),
                         stop_price=_r(r.get("stop_price", r.get("stop_price_ref")), 3),
                         take_profit_price=_r(r.get("take_profit_price", r.get("take_profit_price_ref")), 3),
                         rule_version=version, logged_at=ts))
    return append("signals", rows, d)


def log_plan(result, sleeve, version, d=None, ts=None):
    td = result["trade_date"]
    if any(r["trade_date"] == td for r in read("plan", d)):
        return 0
    row = dict(trade_date=td, invest=int(bool(result.get("invest", True))), gate_open=int(bool(result.get("gate_open"))),
               n_selected=result.get("n_selected", 0),
               stock_used=_r(sleeve.get("stock_used") if sleeve else None, 4),
               sleeve_weight=_r(sleeve.get("sleeve_weight") if sleeve else 0.0, 4),
               rule_version=version, source="live", logged_at=ts or now())
    return append("plan", [row], d)


def sleeve_daily(closes):
    """三只 ETF 收盘到收盘的等权收益：{日期字符串: 收益}。只用三只都有数的日子。"""
    import pandas as pd
    if not closes:
        return {}
    df = pd.DataFrame(closes).sort_index()
    r = df.pct_change(fill_method=None).dropna(how="any")
    return {str(k.date()): float(v) for k, v in r.mean(axis=1).items()}


def log_nav(live, closes, version, d=None, ts=None):
    """把 live.json 里新出现的日子记进 nav.csv。ETF 收益缺的日子先不记，下次再补（不写半截数）。"""
    have = read("nav", d)
    done = {r["date"] for r in have}
    nav = float(have[-1]["nav"]) if have else 1.0
    weights = {r["trade_date"]: float(r["sleeve_weight"] or 0) for r in read("plan", d)}
    srets = sleeve_daily(closes)
    rows, ts = [], ts or now()
    for x in live.get("summary", {}).get("daily", []):
        day = x["trade_date"]
        if day in done or (have and day <= have[-1]["date"]):
            continue
        w = weights.get(day, 0.0)
        if w > 0 and day not in srets:
            break  # 按日期顺序记，缺一天就停，保证净值连续
        sr = srets.get(day, 0.0)
        total = float(x["net"]) + w * sr
        nav *= 1 + total
        rows.append(dict(date=day, stock_net=_r(x["net"]), sleeve_weight=_r(w, 4), sleeve_ret=_r(sr),
                         total_net=_r(total), nav=_r(nav), rule_version=version, source="live", logged_at=ts))
    return append("nav", rows, d)


def log_trades(live, d=None, ts=None):
    seen = {(r["code"], r["entry_date"]) for r in read("trades", d)}
    ts = ts or now()
    rows = [dict(code=t["code"], name=t.get("name", ""), entry_date=t["entry_date"], exit_date=t["exit_date"],
                 net=_r(t["net"], 4), kind="影子", logged_at=ts)
            for t in live.get("rows", []) if (t["code"], t["entry_date"]) not in seen]
    return append("trades", rows, d)


def update(result, live, sleeve, sm, closes=None, d=None):
    version = rule_version(sm)
    ts = now()
    if closes is None and sm.get("sleeve"):
        from .live_sleeve import fetch_closes
        closes = fetch_closes(sm["sleeve"]["codes"])
    n = dict(signals=log_signals(result, version, d, ts), plan=log_plan(result, sleeve, version, d, ts),
             nav=log_nav(live, closes or {}, version, d, ts), trades=log_trades(live, d, ts))
    log.info("ledger appended %s", n)
    return n
