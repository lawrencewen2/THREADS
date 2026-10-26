"""完整回测：训练期定权重和参数，样本外检验，生成报告和面板数据。"""

import json
import logging

import numpy as np
import pandas as pd

from . import config, diagnose, model, portfolio
from .factors import FACTORS, Panel, compute_factors, eligible, trade_outcomes

log = logging.getLogger(__name__)


def prepare(kline, basic, valuation=None, keep_raw=True):
    """keep_raw=False 时算完排名就丢掉原始因子值（只留市值），全 A股回测省一半内存。"""
    p = Panel(kline, valuation)
    f = compute_factors(p)
    elig = eligible(p, basic)
    ranks = {}
    for k in list(f):
        ranks[k] = model.xs_rank(f[k], elig)
        if not keep_raw and k != "size":
            del f[k]
    return p, f, elig, ranks


def run(kline, basic, valuation=None):
    p, f, elig, ranks = prepare(kline, basic, valuation, keep_raw=False)
    size = f.get("size")
    del f, kline, valuation  # 全 A股面板很大，回测只用排名
    out = trade_outcomes(p)
    label_mask = elig & out["buyable"]
    weights, stats = model.fit_weights(ranks, out["day_ret"], label_mask, config.TRAIN_START, config.TRAIN_END)
    # 样本外 IC 只用于报告，不参与任何选择
    for name, r in ranks.items():
        ic = model.daily_ic(r, out["day_ret"], label_mask).loc[config.TEST_START:].dropna()
        stats[name]["test_ic_mean"] = float(ic.mean()) if len(ic) else None
    sc = model.score(ranks, weights, elig)
    order = sc.rank(axis=1, ascending=False, method="first")
    fear = model.load_fear(p.dates)
    holding = holding_study(p, ranks, elig, out, fear)
    diagnosis = diagnose.run(p, size, elig, ranks, out, sc)
    port = portfolio_study(p, ranks, elig, out, fear)
    del ranks

    end = p.dates[-1]
    grid = []
    for n in config.TOP_N_GRID:
        for floor in config.FEAR_FLOOR_GRID:
            picks = model.pick_mask(sc, n, fear, floor, order)
            grid.append(dict(top_n=n, fear_floor=floor,
                             stats=model.evaluate(picks, out, config.TRAIN_START, config.TRAIN_END),
                             test=model.evaluate(picks, out, config.TEST_START, end)))
    chosen = model.choose(grid)
    base_mask = elig & sc.notna()
    baseline = dict(train=model.evaluate(base_mask, out, config.TRAIN_START, config.TRAIN_END),
                    test=model.evaluate(base_mask, out, config.TEST_START, end))
    m = model.save_model(weights, stats, chosen, extra=dict(
        fitted_on=str(end.date()),
        test=model.public(chosen["test"]), train_stats=model.public(chosen["stats"])))
    return dict(p=p, out=out, sc=sc, order=order, fear=fear, grid=grid, chosen=chosen, baseline=baseline, model=m,
                holding=holding, diagnosis=diagnosis, portfolio=port)


def portfolio_study(p, ranks, elig, out, fear):
    """改进版：按“持有 5 天”的可实现收益定因子权重，分散持有多只，排名跌出缓冲区才卖，降低换手。"""
    end = p.dates[-1]
    train_end = p.dates[p.dates <= pd.Timestamp(config.TRAIN_END)][-6]
    w, _ = model.fit_weights(ranks, out["hold5_net"], elig & out["buyable"], config.TRAIN_START, train_end)
    sc = model.score(ranks, w, elig)
    ctx = portfolio.context(p)
    grid = []
    for n in config.PORT_N_GRID:
        for buf in config.PORT_BUFFER_GRID:
            for floor in config.PORT_FLOOR_GRID:
                d_tr, t_tr = portfolio.simulate(ctx, sc, fear, n, buf, floor, config.TRAIN_START, config.TRAIN_END)
                d_te, t_te = portfolio.simulate(ctx, sc, fear, n, buf, floor, config.TEST_START, end)
                grid.append(dict(n=n, buffer=buf, fear_floor=floor,
                                 train=portfolio.stats(d_tr, t_tr), test=portfolio.stats(d_te, t_te)))
                log.info("portfolio n=%d buf=%d floor=%s train %.3f test %.3f", n, buf, floor,
                         grid[-1]["train"].get("ann_return", np.nan), grid[-1]["test"].get("ann_return", np.nan))
    chosen = portfolio.choose(grid)
    # 对照：全部可选股票等权、每天开盘到次日开盘，不扣费（近似全市场等权指数）
    adj = pd.DataFrame(ctx["adj"], index=p.dates, columns=p.codes)
    mkt = (adj.shift(-1) / adj - 1).where(elig.shift(1, fill_value=False)).mean(axis=1)
    bench = {k: portfolio.stats(mkt.loc[s:e].dropna(), []) for k, (s, e) in
             dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, end)).items()}
    return dict(weights=w, grid=grid, chosen=chosen, bench=bench)


def holding_study(p, ranks, elig, out, fear):
    """多日持有对照：每个持有期单独在训练期定因子权重和参数，样本外检验。"""
    end = p.dates[-1]
    rows = []
    for h in config.HOLD_DAYS_GRID:
        target = out[f"hold{h}_net"]
        mask = elig & out["buyable"]
        # 训练期最后 h 天的标签会用到 2021 年的价格，剔掉，避免把样本外信息带进训练
        train_end = p.dates[p.dates <= pd.Timestamp(config.TRAIN_END)][-(h + 1)]
        w, _ = model.fit_weights(ranks, target, mask, config.TRAIN_START, train_end)
        if not any(w.values()):
            rows.append(dict(hold=h, weights=w, train={}, test={}, baseline={}))
            continue
        sc = model.score(ranks, w, elig)
        order = sc.rank(axis=1, ascending=False, method="first")
        grid = []
        for n in config.TOP_N_GRID:
            for floor in config.FEAR_FLOOR_GRID:
                picks = model.pick_mask(sc, n, fear, floor, order)
                grid.append(dict(top_n=n, fear_floor=floor,
                                 train=model.evaluate_hold(picks, out, h, config.TRAIN_START, train_end),
                                 test=model.evaluate_hold(picks, out, h, config.TEST_START, end)))
        c = model.choose_hold(grid)
        base = model.evaluate_hold(elig & sc.notna(), out, h, config.TEST_START, end)
        rows.append(dict(hold=h, weights=w, top_n=c["top_n"], fear_floor=c["fear_floor"],
                         train=c["train"], test=c["test"], baseline=base))
        log.info("hold %d: top %s floor %s test %s", h, c["top_n"], c["fear_floor"], model.public(c["test"]))
    return rows


def _pct(x, digits=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{digits}f}%"


def yearly(picks, out, start, end):
    rows = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        s = model.evaluate(picks, out, f"{y}-01-01", f"{y}-12-31")
        if s.get("trades"):
            rows.append(dict(year=y, **model.public(s)))
    return rows


def to_markdown(res):
    c, b, m = res["chosen"], res["baseline"], res["model"]
    end = res["p"].dates[-1].date()
    floor = "不设开关" if c["fear_floor"] is None else f"A股恐慌贪婪指数 ≥ {c['fear_floor']} 才选股"
    lines = [
        "# A股开盘买入选股回测",
        "",
        f"数据截至 {end}。股票池：全 A股（沪深主板、创业板、科创板，含已退市股票）。"
        f"训练期 {config.TRAIN_START[:4]}–{config.TRAIN_END[:4]}（定权重和参数），"
        f"样本外 {config.TEST_START[:4]} 至今（只检验，不调参）。",
        "",
        "规则：前一交易日收盘后打分，T 日开盘买入（开盘涨停买不进则放弃），T+1 开盘卖出（停牌或跌停开盘顺延）。"
        f"成本：佣金单边 {config.COMMISSION * 1e4:.1f}‱，滑点单边 {config.SLIPPAGE * 1e3:.0f}‰，"
        "卖出印花税 0.1%（2023-08-28 起 0.05%）。剔除 ST、上市不满半年、停牌、近 20 日日均成交额低于 5000 万的股票。",
        "",
        "> A股 T+1，T 日买入当天不能卖。“当日浮盈率”是 T 日收盘按卖出价扣完全部费用后仍高于买入成本的比例，"
        "是账面结果；真正落袋的是“T+1 开盘卖出”那一列。仅供个人研究参考，不构成投资建议。",
        "",
        f"## 训练期选定：每天前 {c['top_n']} 只，{floor}",
        "",
        "| 区间 | 有选股的天数占比 | 交易笔数 | 当日收盘高于买入价 | 当日扣费浮盈率 | 当日组合浮盈天数 | T+1开盘卖出胜率 | 每笔平均（T+1开盘卖） | 年化 | 最大回撤 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def row(name, s):
        if not s.get("trades"):
            return f"| {name} | 0 | 0 | — | — | — | — | — | — | — |"
        return (f"| {name} | {_pct(s['trade_day_share'], 0)} | {s['trades']} | {_pct(s['hit_gross'])} | "
                f"{_pct(s['hit_net'])} | {_pct(s['day_win_net'])} | {_pct(s['hit_next_open_net'])} | "
                f"{_pct(s['avg_next_open_net'], 2)} | {_pct(s['ann_return'])} | {_pct(s['max_drawdown'])} |")

    lines += [row("训练期", c["stats"]), row("**样本外**", c["test"]),
              row("对照：股票池全部可买股票（训练期）", b["train"]),
              row("对照：股票池全部可买股票（样本外）", b["test"]), ""]

    lines += ["## 选得越少，成功率能高多少（样本外，当日扣费浮盈率）", "",
              "| 每天选股数 | " + " | ".join("不设开关" if f is None else f"指数≥{f}" for f in config.FEAR_FLOOR_GRID) + " |",
              "| --- |" + " --- |" * len(config.FEAR_FLOOR_GRID)]
    for n in config.TOP_N_GRID:
        cells = []
        for f in config.FEAR_FLOOR_GRID:
            g = next(x for x in res["grid"] if x["top_n"] == n and x["fear_floor"] == f)
            t = g["test"]
            cells.append(f"{_pct(t.get('hit_net'))}（{_pct(t.get('trade_day_share'), 0)}天）" if t.get("trades") else "—")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "括号里是有选股的交易日占比。", ""]

    picks = model.pick_mask(res["sc"], c["top_n"], res["fear"], c["fear_floor"], res["order"])
    lines += ["## 样本外逐年", "", "| 年份 | 交易笔数 | 当日扣费浮盈率 | T+1开盘卖出胜率 | 每笔平均 | 年化 |", "| --- | --- | --- | --- | --- | --- |"]
    for r in yearly(picks, res["out"], config.TEST_START, res["p"].dates[-1]):
        lines.append(f"| {r['year']} | {r['trades']} | {_pct(r['hit_net'])} | {_pct(r['hit_next_open_net'])} | "
                     f"{_pct(r['avg_next_open_net'], 2)} | {_pct(r['ann_return'])} |")

    lines += ["", "## 对照：持有更久，费用占比更小", "",
              "同样的因子、同样的成本，每个持有期单独在训练期定权重和参数，下表是样本外。T 开盘买，T+h 开盘卖。", "",
              "| 持有 | 训练期选定 | 交易笔数 | 胜率（扣费） | 每笔平均 | 年化 | 最大回撤 | 对照：全部可买股票每笔平均 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in res.get("holding", []):
        t, bb = r["test"], r["baseline"]
        if not t.get("trades"):
            lines.append(f"| {r['hold']} 天 | 没有显著因子 | 0 | — | — | — | — | — |")
            continue
        gate = "不设开关" if r["fear_floor"] is None else f"指数≥{r['fear_floor']}"
        lines.append(f"| {r['hold']} 天 | 前 {r['top_n']} 只，{gate} | {t['trades']} | {_pct(t['hit'])} | "
                     f"{_pct(t['avg_net'], 2)} | {_pct(t['ann_return'])} | {_pct(t['max_drawdown'])} | "
                     f"{_pct(bb.get('avg_net'), 2)} |")

    pf = res.get("portfolio")
    if pf:
        ch = pf["chosen"]
        lines += ["", "## 改进版：分散持有 + 换手缓冲", "",
                  "按“持有 5 天”的可实现收益在训练期定因子权重；每天目标持有前 n 只，已持有的只要排名还在前 n×缓冲倍数就不卖。"
                  "参数只在训练期挑（年化为正的组合里夏普最高的）。对照组是全部可选股票等权、不扣费。", "",
                  f"训练期选定：持有 {ch['n']} 只，缓冲 {ch['buffer']} 倍，"
                  + ("不设开关" if ch["fear_floor"] is None else f"A股恐慌贪婪指数低于 {ch['fear_floor']} 时空仓") + "。", "",
                  "| 区间 | 年化 | 夏普 | 最大回撤 | 组合赚钱的天数 | 交易笔数 | 每笔胜率 | 每笔平均 | 平均持有天数 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]

        def prow(name, s):
            if not s.get("days"):
                return f"| {name} | — | — | — | — | — | — | — | — |"
            hd = s.get("avg_hold_days")
            return (f"| {name} | {_pct(s['ann_return'])} | {s['sharpe']:.2f} | {_pct(s['max_drawdown'])} | "
                    f"{_pct(s['day_win'])} | {s['trades']} | {_pct(s.get('trade_win'))} | {_pct(s.get('avg_trade'), 2)} | "
                    f"{'—' if hd is None else f'{hd:.1f}'} |")

        lines += [prow("训练期", ch["train"]), prow("**样本外**", ch["test"]),
                  prow("对照：全部可选股票等权（训练期）", pf["bench"]["train"]),
                  prow("对照：全部可选股票等权（样本外）", pf["bench"]["test"]), "",
                  "全部参数组合（样本外，仅供对照，没有参与挑选）：", "",
                  "| 持有只数 | 缓冲 | 开关 | 训练期年化 | 训练期夏普 | 样本外年化 | 样本外夏普 | 样本外最大回撤 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for g in pf["grid"]:
            a, t = g["train"], g["test"]
            lines.append(f"| {g['n']} | {g['buffer']} | {'不设' if g['fear_floor'] is None else '≥' + str(g['fear_floor'])} | "
                         f"{_pct(a.get('ann_return'))} | {a.get('sharpe', 0):.2f} | {_pct(t.get('ann_return'))} | "
                         f"{t.get('sharpe', 0):.2f} | {_pct(t.get('max_drawdown'))} |")

    lines += ["", "## 因子（训练期 IC 定权重，|t| < 2 的不用）", "",
              "| 因子 | 含义 | 训练期 IC | t 值 | 权重 | 样本外 IC |", "| --- | --- | --- | --- | --- | --- |"]
    for k, desc in FACTORS.items():
        if k not in m["factor_stats"]:
            continue
        s = m["factor_stats"][k]
        tic = s.get("test_ic_mean")
        lines.append(f"| {k} | {desc} | {s['ic_mean']:.3f} | {s['t']:.1f} | {m['weights'][k]:+.2f} | "
                     f"{'—' if tic is None else f'{tic:.3f}'} |")
    return "\n".join(lines) + "\n"


def page_data(res):
    """面板用的折线数据：收益曲线、滚动胜率、选股数量与成功率。"""
    c, b = res["chosen"], res["baseline"]
    picks = model.pick_mask(res["sc"], c["top_n"], res["fear"], c["fear_floor"], res["order"])
    full = model.evaluate(picks, res["out"], config.TRAIN_START, res["p"].dates[-1])
    base = model.evaluate(res["sc"].notna(), res["out"], config.TRAIN_START, res["p"].dates[-1])
    eq = pd.DataFrame({"strategy": full["_equity"], "baseline": base["_equity"]})
    eq = eq.iloc[::5]  # 每周一个点，够画折线
    hit = full["_daily_hit"]
    roll = hit.rolling(60, min_periods=20).mean().iloc[::5]
    base_hit = base["_daily_hit"].rolling(60, min_periods=20).mean().reindex(roll.index)
    curve = []
    for n in config.TOP_N_GRID:
        for f in config.FEAR_FLOOR_GRID:
            g = next(x for x in res["grid"] if x["top_n"] == n and x["fear_floor"] == f)
            curve.append(dict(top_n=n, fear_floor=f, hit_net=g["test"].get("hit_net"),
                              trade_day_share=g["test"].get("trade_day_share"),
                              avg_next_open_net=g["test"].get("avg_next_open_net")))
    return dict(
        updated=str(res["p"].dates[-1].date()), test_start=config.TEST_START,
        chosen=dict(top_n=c["top_n"], fear_floor=c["fear_floor"], train=model.public(c["stats"]),
                    test=model.public(c["test"])),
        baseline=dict(train=model.public(b["train"]), test=model.public(b["test"])),
        equity=[dict(date=str(d.date()), strategy=round(float(r.strategy), 4), baseline=round(float(r.baseline), 4))
                for d, r in eq.iterrows()],
        rolling_hit=[dict(date=str(d.date()), strategy=None if np.isnan(v) else round(float(v), 4),
                          baseline=None if np.isnan(bv) else round(float(bv), 4))
                     for d, v, bv in zip(roll.index, roll.values, base_hit.values)],
        selectivity=curve,
        portfolio=None if not res.get("portfolio") else dict(
            n=res["portfolio"]["chosen"]["n"], buffer=res["portfolio"]["chosen"]["buffer"],
            fear_floor=res["portfolio"]["chosen"]["fear_floor"],
            train=model.public(res["portfolio"]["chosen"]["train"]), test=model.public(res["portfolio"]["chosen"]["test"]),
            bench=dict(train=model.public(res["portfolio"]["bench"]["train"]), test=model.public(res["portfolio"]["bench"]["test"])),
            equity=[dict(date=str(d.date()), strategy=round(float(v), 4))
                    for d, v in pd.concat([res["portfolio"]["chosen"]["train"].get("_equity", pd.Series(dtype=float)),
                                           res["portfolio"]["chosen"]["test"].get("_equity", pd.Series(dtype=float))]).iloc[::5].items()],
            grid=[dict(n=g["n"], buffer=g["buffer"], fear_floor=g["fear_floor"],
                       train=model.public(g["train"]), test=model.public(g["test"])) for g in res["portfolio"]["grid"]]),
        holding=[dict(hold=r["hold"], top_n=r.get("top_n"), fear_floor=r.get("fear_floor"),
                      train=model.public(r["train"]), test=model.public(r["test"]),
                      baseline=model.public(r["baseline"])) for r in res.get("holding", [])],
    )


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "backtest.md").write_text(to_markdown(res))
    if res.get("diagnosis"):
        (config.REPORT_DIR / "diagnosis.md").write_text(res["diagnosis"])
    (config.PAGE_DIR / "backtest.json").write_text(json.dumps(page_data(res), ensure_ascii=False))
