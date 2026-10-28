"""胜率能做到多高（2026-10-28 用户：选股胜率要在 80% 以上）。

线上的“业绩预期 + 情绪开关”每天按得分取前 N 只，T 日开盘买入。卖出规则换成止盈、止损和最长持有天数：
- T+1 起（A股当天买的不能卖），某天最高价碰到止盈价就按止盈价卖（开盘就高于止盈价的按开盘价）；
- 有止损时，最低价碰到止损价就按止损价卖（开盘就低于止损价的按开盘价）；同一天两者都碰到按止损算（保守）；
- 都没碰到，持有满 H 天后的开盘卖。开盘涨停买不进；价格都用复权价。扣佣金、双边滑点和印花税。
每种规则在训练期（2015–2020）和样本外（2021 至今）分别算每笔胜率、每笔平均、最差 5% 的一笔和估算年化。
估算年化：每天新开的一批平均收益，资金分成“平均持有天数”份轮流用。规则只在训练期挑，样本外只检验。
"""

import itertools
import json
import logging

import numpy as np
import pandas as pd

from . import config, model, portfolio
from .alt_study import LIVE_FILE, emotion_regime, iter_alt_factors
from .factors import Panel, eligible

log = logging.getLogger(__name__)

TAKE_PROFIT = (0.01, 0.02, 0.03, 0.05)
STOP_LOSS = (None, 0.05, 0.08)
MAX_HOLD = (5, 10, 20)
TOP_N = (5, 20)
TARGET_WIN = 0.80
GOAL_WIN = 0.50  # 用户 2026-10-28 改定的目标：每笔胜率 50–60%


def adjusted_ohlc(p):
    adj_open = portfolio.adjusted_open(p)
    k = adj_open / p.open
    return adj_open.to_numpy(), (p.high * k).to_numpy(), (p.low * k).to_numpy()


def entries(p, sc, gate, n):
    """(信号日序号, 股票列) 列表：gate 开的信号日按得分取前 n 只，T = 信号日 + 1 开盘买。"""
    up, _ = p.limit_prices()
    up_v, open_v, traded = up.to_numpy(), p.open.to_numpy(), p.traded.to_numpy()
    sc_v = sc.to_numpy()
    gate_v = gate.reindex(p.dates).fillna(False).to_numpy(dtype=bool)
    out = []
    for t in range(len(p.dates) - 1):
        if not gate_v[t]:
            continue
        row = sc_v[t]
        ok = ~np.isnan(row)
        if not ok.any():
            continue
        order = np.argsort(-np.where(ok, row, -np.inf))[:n * 3]
        T, got = t + 1, 0
        for j in order:
            if got >= n or not ok[j]:
                break
            if traded[T, j] and open_v[T, j] < up_v[T, j] - 0.005:
                out.append((T, j))
                got += 1
    return np.array(out, dtype=int).reshape(-1, 2)


def simulate_exits(ohl, ent, dates, tp, sl, hold):
    """每笔的扣费收益和持有天数。"""
    o, h, l = ohl
    T, j = ent[:, 0], ent[:, 1]
    last = len(dates) - 1
    keep = T + hold <= last
    T, j = T[keep], j[keep]
    entry = o[T, j]
    buy_cost = config.SLIPPAGE + config.COMMISSION
    exit_px = o[T + hold, j].copy()
    exit_day = np.full(len(T), hold)
    done = np.zeros(len(T), dtype=bool)
    for d in range(1, hold):
        od, hd, ld = o[T + d, j], h[T + d, j], l[T + d, j]
        live = ~done & np.isfinite(od) & np.isfinite(hd)
        if sl is not None:
            stop = entry * (1 - sl)
            hit = live & (ld <= stop)
            exit_px[hit] = np.minimum(od[hit], stop[hit])
            exit_day[hit] = d
            done |= hit
            live &= ~hit
        target = entry * (1 + tp)
        hit = live & (hd >= target)
        exit_px[hit] = np.maximum(od[hit], target[hit])
        exit_day[hit] = d
        done |= hit
    stamp = np.where(dates[T] >= pd.Timestamp(config.STAMP_DUTY_CHANGE), config.STAMP_DUTY_NEW, config.STAMP_DUTY_OLD)
    sell_cost = config.SLIPPAGE + config.COMMISSION + stamp
    net = exit_px * (1 - sell_cost) / (entry * (1 + buy_cost)) - 1
    ok = np.isfinite(net)
    return T[ok], net[ok], exit_day[ok]


def summarize(T, net, days, dates, start, end):
    m = (dates[T] >= pd.Timestamp(start)) & (dates[T] <= pd.Timestamp(end))
    T, net, days = T[m], net[m], days[m]
    if not len(net):
        return dict(trades=0)
    cohort = pd.Series(net).groupby(T).mean()
    hold = float(days.mean())
    n_days = int(((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))).sum())
    # 资金分成 hold 份，每天一份去买当天那一批；没开仓的日子那份空着
    per_day = cohort.sum() / n_days / hold
    return dict(trades=int(len(net)), win=float((net > 0).mean()), avg=float(net.mean()),
                worst5=float(np.quantile(net, 0.05)), avg_loss=float(net[net <= 0].mean()) if (net <= 0).any() else 0.0,
                avg_win=float(net[net > 0].mean()) if (net > 0).any() else 0.0, hold_days=hold,
                ann_est=float((1 + per_day) ** 244 - 1))


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
    dates = p.dates
    end = str(dates[-1].date())
    periods = dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, end))
    rows = []
    for n in TOP_N:
        ent = entries(p, sc, gate, n)
        for tp, sl, hold in itertools.product(TAKE_PROFIT, STOP_LOSS, MAX_HOLD):
            T, net, days = simulate_exits(ohl, ent, dates, tp, sl, hold)
            row = dict(top_n=n, take_profit=tp, stop_loss=sl, max_hold=hold,
                       **{k: summarize(T, net, days, dates, s, e) for k, (s, e) in periods.items()})
            rows.append(row)
        log.info("top %d: %d entries", n, len(ent))
    # 训练期挑：每笔平均为正的规则里胜率最高的
    pos = [r for r in rows if r["train"].get("avg", -1) > 0]
    best_win = max(pos or rows, key=lambda r: r["train"].get("win", 0))
    best_ret = max(rows, key=lambda r: r["train"].get("ann_est", -9))
    goal = [r for r in rows if r["train"].get("win", 0) >= GOAL_WIN and r["train"].get("avg", -1) > 0]
    best_goal = max(goal, key=lambda r: r["train"].get("ann_est", -9)) if goal else None
    return dict(updated=end, rows=rows, best_win=best_win, best_ret=best_ret, best_goal=best_goal, target_win=TARGET_WIN,
                n_reach_target_test=sum(1 for r in rows if r["test"].get("win", 0) >= TARGET_WIN),
                n_reach_target_test_positive=sum(1 for r in rows if r["test"].get("win", 0) >= TARGET_WIN
                                                 and r["test"].get("avg", -1) > 0))


def _p(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def _rule(r):
    sl = "不止损" if r["stop_loss"] is None else f"止损 {r['stop_loss'] * 100:.0f}%"
    return f"前 {r['top_n']} 只，止盈 {r['take_profit'] * 100:.0f}%，{sl}，最长 {r['max_hold']} 天"


def to_markdown(res):
    lines = ["# 胜率能做到多高：业绩预期 + 情绪开关，换成止盈止损卖出", "",
             f"数据截至 {res['updated']}。训练期 2015–2020，样本外 2021 至今。每笔扣佣金、双边滑点和印花税；"
             "T+1 才能卖；开盘涨停买不进；同一天碰到止盈和止损按止损算。", "",
             f"共 {len(res['rows'])} 种规则。样本外每笔胜率达到 {res['target_win'] * 100:.0f}% 的有 "
             f"{res['n_reach_target_test']} 种，其中每笔平均还赚钱的有 {res['n_reach_target_test_positive']} 种。", ""]
    for label, key in (("训练期挑出的胜率最高（每笔平均为正）", "best_win"), ("训练期挑出的估算年化最高", "best_ret"),
                       (f"训练期胜率 ≥ {GOAL_WIN * 100:.0f}% 且每笔平均为正里估算年化最高", "best_goal")):
        r = res[key]
        if r is None:
            lines += [f"**{label}**：训练期没有规则达标。", ""]
            continue
        a, t = r["train"], r["test"]
        lines += [f"**{label}**：{_rule(r)}。训练期胜率 {_p(a.get('win'), 0)}、每笔平均 {_p(a.get('avg'), 2)}；"
                  f"样本外胜率 {_p(t.get('win'), 0)}、每笔平均 {_p(t.get('avg'), 2)}、最差 5% 的一笔 {_p(t.get('worst5'))}、"
                  f"估算年化 {_p(t.get('ann_est'))}。", ""]
    lines += ["## 全部规则（按样本外胜率排序）", "",
              "| 规则 | 训练期胜率 | 训练期每笔平均 | 样本外胜率 | 样本外每笔平均 | 赚的一笔平均 | 亏的一笔平均 | 最差 5% 的一笔 | 平均持有天数 | 样本外估算年化 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(res["rows"], key=lambda r: -r["test"].get("win", 0)):
        a, t = r["train"], r["test"]
        lines.append(f"| {_rule(r)} | {_p(a.get('win'), 0)} | {_p(a.get('avg'), 2)} | {_p(t.get('win'), 0)} | {_p(t.get('avg'), 2)} | "
                     f"{_p(t.get('avg_win'), 2)} | {_p(t.get('avg_loss'), 2)} | {_p(t.get('worst5'))} | "
                     f"{t.get('hold_days', 0):.1f} | {_p(t.get('ann_est'))} |")
    lines += ["", "仅供个人研究参考，不构成投资建议。"]
    return "\n".join(lines) + "\n"


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "winrate.md").write_text(to_markdown(res))
    (config.PAGE_DIR / "winrate.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
