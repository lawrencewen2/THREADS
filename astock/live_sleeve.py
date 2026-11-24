"""每天的 ETF 分仓目标（2026-11-10 起，见 astock/diversify.py 和 reports/astock/diversify.md）。

选股用 capital_share 的资金（每只 capital_share / 20）；其余资金和选股闲着的钱放进 slot_model.json 里 sleeve 的几只 ETF，等权：
- 当天要买股票、钱不够时，先卖 ETF 腾出现金（ETF 分仓比例马上降到 1 − 股票占用）；
- 闲钱（卖出股票、空仓开关关着）攒到总资金的 park_band（10%）以上才买回 ETF；
- 每 rebalance（20）个交易日把几只 ETF 调回等权。
写 docs/astock/sleeve.json，状态存 data/astock/sleeve_state.json；picks.json 里加一个 sleeve 摘要。
"""

import json
import logging
import urllib.request

import pandas as pd

from . import config

log = logging.getLogger(__name__)

STATE_FILE = config.DATA_DIR / "sleeve_state.json"
OUT_FILE = config.PAGE_DIR / "sleeve.json"
UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}


def fetch_closes(codes, count=40, timeout=15):
    """腾讯前复权日线最近 count 根收盘价。返回 {code: Series}；失败的不在里面。"""
    out = {}
    for c in codes:
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={c},day,,,{count},qfq"
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read().decode("utf-8", errors="replace"))["data"][c]
            k = body.get("qfqday") or body.get("day") or []
            out[c] = pd.Series([float(x[2]) for x in k], index=pd.to_datetime([x[0] for x in k]))
        except Exception as exc:  # 行情拉不到不影响选股清单
            log.warning("etf %s quote failed: %s", c, exc)
    return out


def stock_used(result, sm):
    """交易日开盘后股票占总资金的比例：继续持有 + 买入的只数 × 每只资金。"""
    if not result.get("invest", True):
        return 0.0
    n_sel = sum(1 for r in result.get("picks", []) if r.get("selected"))
    return min(1.0, n_sel * sm.get("capital_share", 1.0) / sm.get("n", 20))


def plan(used, prev, cfg, trade_date, days_since_rebalance):
    """返回 (新的 ETF 分仓比例, 动作说明, 是否调回等权)。prev 是上一个交易日的分仓比例（第一次用 None）。"""
    target = max(0.0, 1.0 - used)
    band = cfg.get("park_band", 0.10)
    if prev is None:
        return target, "开始：按目标比例买入 ETF 分仓", True
    if target < prev - 1e-9:
        return target, "卖出部分 ETF，腾出现金买股票", True
    if target - prev > band:
        return target, "闲钱超过总资金 10%，买回 ETF", True
    if days_since_rebalance >= cfg.get("rebalance", 20):
        return prev, "满 20 个交易日，调回等权", True
    return prev, "不动", False


def update(result, sm, closes=None):
    cfg = sm.get("sleeve")
    if not cfg:
        return None
    trade_date = result["trade_date"]
    state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    if state.get("trade_date") == trade_date:  # 同一天重跑：从前一天的状态重新算
        prev, rebalanced = state.get("prev_weight"), state.get("prev_rebalanced")
    else:
        prev, rebalanced = state.get("weight"), state.get("rebalanced")
    closes = fetch_closes(cfg["codes"]) if closes is None else closes
    ref = next((s for s in closes.values() if len(s)), None)
    if rebalanced and ref is not None:
        days = int((ref.index > pd.Timestamp(rebalanced)).sum()) + 1  # 交易日 T 也算一天
    else:
        days = 0 if rebalanced else 10 ** 6
    used = stock_used(result, sm)
    weight, action, reb = plan(used, prev, cfg, trade_date, days)
    names = dict(zip(cfg["codes"], cfg.get("names", cfg["codes"])))
    k = len(cfg["codes"])
    etfs = []
    for c in cfg["codes"]:
        s = closes.get(c)
        etfs.append(dict(code=c, name=names[c], weight=round(weight / k, 4),
                         close=None if s is None or not len(s) else round(float(s.iloc[-1]), 3),
                         close_date=None if s is None or not len(s) else str(s.index[-1].date())))
    out = dict(trade_date=trade_date, signal_date=result.get("signal_date"), capital_share=sm.get("capital_share"),
               stock_used=round(used, 4), sleeve_weight=round(weight, 4),
               prev_weight=None if prev is None else round(prev, 4), cash=round(max(0.0, 1 - used - weight), 4),
               action=action, rebalance=reb, days_since_rebalance=None if days >= 10 ** 6 else days, etfs=etfs,
               note=cfg.get("note", ""))
    bt = config.PAGE_DIR / "diversify.json"
    if bt.exists():
        try:
            d = json.loads(bt.read_text())
            out["backtest"] = dict(test=d["chosen"]["test"], train=d["chosen"]["train"], updated=d.get("updated"))
        except Exception:
            pass
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    STATE_FILE.write_text(json.dumps(dict(trade_date=trade_date, weight=weight, prev_weight=prev,
                                          rebalanced=trade_date if reb else rebalanced, prev_rebalanced=rebalanced),
                                     ensure_ascii=False, indent=1))
    picks = config.PAGE_DIR / "picks.json"
    if picks.exists():
        p = json.loads(picks.read_text())
        p["sleeve"] = {k2: out[k2] for k2 in ("sleeve_weight", "stock_used", "action")} | dict(
            etfs=[dict(code=e["code"], name=e["name"], weight=e["weight"]) for e in etfs])
        picks.write_text(json.dumps(p, ensure_ascii=False, indent=1, default=float))
    log.info("sleeve: used %.3f weight %.3f (%s)", used, weight, action)
    return out
