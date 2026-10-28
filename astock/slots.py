"""能实际照做的版本：最多持有 n 只，每只买入前定好止盈价、止损价和最长持有天数（2026-10-28 用户要求）。

winrate.py 里每天新开 20 笔、资金按平均持有天数分摊，只是估算；实际操作时仓位有限。这里按真实账户模拟：
- 资金分 n 份，每只占一份；每天开盘先处理卖出，再按信号日得分从前 n×3 名里补满空位（情绪开关关时不买新的）。
- 卖出：持有满 H 天的开盘卖；否则 T+1 起最低价碰到止损价按止损价卖（开盘已低于止损价按开盘价），
  最高价碰到止盈价按止盈价卖（开盘已高于止盈价按开盘价），同一天都碰到按止损算；一字跌停或停牌卖不掉，顺延。
- 止损、止盈价按买入价和买入前一天的 14 日 ATR（相对价格）定，买入后不再改。
- 每天的收益按复权开盘价到下一天开盘计；卖出那天按卖出价计。扣佣金、双边滑点和印花税。
规则只在训练期（2015–2020）挑：每笔胜率 ≥ 50% 里夏普最高的；样本外（2021 至今）只检验。
"""

import itertools
import json
import logging

import numpy as np
import pandas as pd

from . import config, model, portfolio
from .alt_study import LIVE_FILE, emotion_regime, iter_alt_factors
from .factors import Panel, eligible
from .stability import overlay
from .winrate import adjusted_ohlc

log = logging.getLogger(__name__)

N = 20
TAKE_PROFIT = (None, "atr2", "atr3", "atr4")
STOP_LOSS = ("atr2", "atr3", 0.08)
MAX_HOLD = (10, 20)
GOAL_WIN = 0.50
GOAL_DD = -0.15
OVERLAYS = [
    ("波动率目标年化 15%", "vol15"),
    ("波动率目标年化 10%", "vol10"),
    ("净值低于 20 日均线时空仓", "trend20"),
    ("回撤超过 10% 时半仓，回到 20 日均线上方恢复", "dd10"),
]


def levels(entry, atr, tp, sl):
    """买入价 entry、前一天相对 ATR 下的止盈价和止损价（复权口径，实际价格同比例）。"""
    stop = entry * (1 - (float(sl[3:]) * atr if isinstance(sl, str) else sl))
    target = np.inf if tp is None else entry * (1 + float(tp[3:]) * atr)
    return target, stop


def simulate(p, ohl, sc, gate, n, tp, sl, hold, start, end):
    o, h, l, atr, locked = ohl
    up, _ = p.limit_prices()
    up_v, raw_open, traded = up.to_numpy(), p.open.to_numpy(), p.traded.to_numpy()
    sc_v = sc.to_numpy()
    gate_v = gate.reindex(p.dates).fillna(False).to_numpy(dtype=bool)
    dates = p.dates
    i0 = max(dates.searchsorted(pd.Timestamp(start)), 15)
    i1 = min(dates.searchsorted(pd.Timestamp(end), side="right"), len(dates) - 1)
    buy_cost = config.SLIPPAGE + config.COMMISSION
    stamp_change = pd.Timestamp(config.STAMP_DUTY_CHANGE)
    held = {}  # 列号 -> [买入价, 止盈价, 止损价, 买入日序号]
    daily, trades = [], []
    for T in range(i0, i1):
        stamp = config.STAMP_DUTY_NEW if dates[T] >= stamp_change else config.STAMP_DUTY_OLD
        sell_cost = config.SLIPPAGE + config.COMMISSION + stamp
        r = 0.0
        for j in list(held):
            entry, target, stop, d0 = held[j]
            px = None
            if not locked[T, j] and np.isfinite(o[T, j]):
                if T - d0 >= hold:
                    px = o[T, j]
                elif l[T, j] <= stop:
                    px = min(o[T, j], stop)
                elif h[T, j] >= target:
                    px = max(o[T, j], target)
            if px is None:
                continue
            r += (px / o[T, j] - 1 - sell_cost) / n  # 当天开盘到卖出价
            trades.append(dict(entry=dates[d0], exit=dates[T],
                               net=px * (1 - sell_cost) / (entry * (1 + buy_cost)) - 1))
            del held[j]
        if gate_v[T - 1] and len(held) < n:
            row = sc_v[T - 1]
            ok = ~np.isnan(row)
            for j in np.argsort(-np.where(ok, row, -np.inf))[:n * 3]:
                if len(held) >= n or not ok[j]:
                    break
                if j in held or not traded[T, j] or raw_open[T, j] >= up_v[T, j] - 0.005:
                    continue
                a = atr[T - 1, j]
                if not np.isfinite(a) or not np.isfinite(o[T, j]):
                    continue
                target, stop = levels(o[T, j], a, tp, sl)
                held[j] = [o[T, j], target, stop, T]
                r -= buy_cost / n
        for j in held:  # 当天开盘（或买入价）到下一天开盘
            if np.isfinite(o[T + 1, j]) and o[T, j] > 0:
                r += (o[T + 1, j] / o[T, j] - 1) / n
        daily.append((dates[T], r))
    return pd.Series(dict(daily), dtype="float64"), trades


def stats(daily, trades):
    st = portfolio.stats(daily, trades)
    st.pop("_equity", None)
    nets = np.array([t["net"] for t in trades])
    if len(nets) and (nets > 0).any() and (nets < 0).any():
        st["payoff"] = float(nets[nets > 0].mean() / -nets[nets < 0].mean())
        st["avg_loss"] = float(nets[nets <= 0].mean())
        st["worst5"] = float(np.quantile(nets, 0.05))
    yearly = (1 + daily).resample("YE").prod() - 1
    st["yearly"] = {str(d.year): round(float(v), 4) for d, v in yearly.items()}
    return st


def run(kline, basic, valuation=None, forecast=None):
    kline = kline() if callable(kline) else kline
    valuation = valuation() if callable(valuation) else valuation
    forecast = forecast() if callable(forecast) else forecast
    fm = json.loads(LIVE_FILE.read_text())
    weights = {k: v for k, v in fm["weights"].items() if v}
    p = Panel(kline, valuation)
    del kline, valuation
    elig = eligible(p, basic)
    ranks = {}
    for k, v in iter_alt_factors(p, forecast):
        if k in weights:
            ranks[k] = model.xs_rank(v, elig)
        del v
    sc = model.score(ranks, weights, elig)
    del ranks
    gate = emotion_regime(p, elig)
    ohl = adjusted_ohlc(p)
    end = str(p.dates[-1].date())
    periods = dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, end))
    rows = []
    for tp, sl, hold in itertools.product(TAKE_PROFIT, STOP_LOSS, MAX_HOLD):
        row = dict(n=N, take_profit=tp, stop_loss=sl, max_hold=hold)
        for k, (s, e) in periods.items():
            row[k] = stats(*simulate(p, ohl, sc, gate, N, tp, sl, hold, s, e))
        row["meets"] = {k: bool((row[k].get("trade_win") or 0) >= GOAL_WIN and row[k].get("max_drawdown", -1) >= GOAL_DD
                                and row[k].get("ann_return", -1) > 0) for k in periods}
        log.info("%s %s %s: train %s test %s", tp, sl, hold, row["train"].get("ann_return"), row["test"].get("ann_return"))
        rows.append(row)
    pool = [r for r in rows if (r["train"].get("trade_win") or 0) >= GOAL_WIN] or rows
    chosen = max(pool, key=lambda r: r["train"].get("sharpe", -9))
    # 控回撤：在挑出的规则上叠加仓位（波动率目标、净值跌破均线空仓、回撤半仓），胜率不变，只改仓位
    overlays = []
    for name, kind in OVERLAYS:
        row = dict(overlay=name, kind=kind)
        for k, (s, e) in periods.items():
            d, t = simulate(p, ohl, sc, gate, N, chosen["take_profit"], chosen["stop_loss"], chosen["max_hold"], s, e)
            row[k] = stats(overlay(d, kind), t)
        row["meets"] = {k: bool((row[k].get("trade_win") or 0) >= GOAL_WIN and row[k].get("max_drawdown", -1) >= GOAL_DD
                                and row[k].get("ann_return", -1) > 0) for k in periods}
        overlays.append(row)
    return dict(updated=end, n=N, rows=rows, chosen=chosen, chosen_meets_win=pool is not rows, overlays=overlays)


def _p(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def rule(r):
    tp = "不止盈" if r["take_profit"] is None else f"止盈 {r['take_profit'][3:]} 倍 ATR"
    sl = r["stop_loss"]
    sl = f"止损 {sl[3:]} 倍 ATR" if isinstance(sl, str) else f"止损 {sl * 100:.0f}%"
    return f"最多 {r['n']} 只，{tp}，{sl}，最长 {r['max_hold']} 天"


def to_markdown(res):
    c = res["chosen"]
    lines = ["# 带止盈止损的真实账户模拟：业绩预期 + 情绪开关", "",
             f"数据截至 {res['updated']}。最多持有 {res['n']} 只，每只一份资金；规则只用 2015–2020 挑（每笔胜率 ≥ 50% 里夏普最高"
             + ("" if res["chosen_meets_win"] else "；没有规则达到 50%，只按夏普挑") + "），2021 至今只检验。", "",
             f"**训练期挑出：{rule(c)}**。样本外年化 {_p(c['test'].get('ann_return'))}、最大回撤 {_p(c['test'].get('max_drawdown'))}、"
             f"每笔胜率 {_p(c['test'].get('trade_win'), 0)}、盈亏比 {c['test'].get('payoff') or 0:.2f}。", "",
             "| 规则 | 训练期年化 | 训练期回撤 | 训练期胜率 | 样本外年化 | 样本外夏普 | 样本外回撤 | 样本外胜率 | 盈亏比 | 亏的一笔平均 | 最差 5% 的一笔 | 平均持有天数 | 训练期达标 | 样本外达标 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(res["rows"], key=lambda r: -r["train"].get("sharpe", -9)):
        a, t = r["train"], r["test"]
        lines.append(f"| {rule(r)} | {_p(a.get('ann_return'))} | {_p(a.get('max_drawdown'))} | {_p(a.get('trade_win'), 0)} | "
                     f"{_p(t.get('ann_return'))} | {t.get('sharpe', 0):.2f} | {_p(t.get('max_drawdown'))} | {_p(t.get('trade_win'), 0)} | "
                     f"{t.get('payoff') or 0:.2f} | {_p(t.get('avg_loss'), 2)} | {_p(t.get('worst5'))} | {t.get('avg_hold_days') or 0:.1f} | "
                     f"{'是' if r['meets']['train'] else '否'} | {'是' if r['meets']['test'] else '否'} |")
    lines += ["", f"## 挑出规则叠加仓位控制（目标：每笔胜率 ≥ 50%、最大回撤不超过 {-GOAL_DD * 100:.0f}%）", "",
              "| 仓位控制 | 训练期年化 | 训练期回撤 | 样本外年化 | 样本外夏普 | 样本外回撤 | 胜率 | 训练期达标 | 样本外达标 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in res.get("overlays", []):
        a, t = r["train"], r["test"]
        lines.append(f"| {r['overlay']} | {_p(a.get('ann_return'))} | {_p(a.get('max_drawdown'))} | {_p(t.get('ann_return'))} | "
                     f"{t.get('sharpe', 0):.2f} | {_p(t.get('max_drawdown'))} | {_p(t.get('trade_win'), 0)} | "
                     f"{'是' if r['meets']['train'] else '否'} | {'是' if r['meets']['test'] else '否'} |")
    years = sorted(c["test"].get("yearly", {}))
    lines += ["", "## 挑出规则的样本外分年收益", "", "| " + " | ".join(years) + " |", "|" + " --- |" * len(years),
              "| " + " | ".join(_p(c["test"]["yearly"][y]) for y in years) + " |", "", "仅供个人研究参考，不构成投资建议。"]
    return "\n".join(lines) + "\n"


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "slots.md").write_text(to_markdown(res))
    (config.PAGE_DIR / "slots.json").write_text(json.dumps(res, ensure_ascii=False, default=str))
