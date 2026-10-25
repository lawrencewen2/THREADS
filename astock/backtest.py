"""完整回测：训练期定权重和参数，样本外检验，生成报告和面板数据。"""

import json
import logging

import numpy as np
import pandas as pd

from . import config, model
from .factors import FACTORS, Panel, compute_factors, eligible, trade_outcomes

log = logging.getLogger(__name__)


def prepare(kline, universe, basic):
    p = Panel(kline)
    f = compute_factors(p)
    elig = eligible(p, universe, basic)
    ranks = {k: model.xs_rank(v, elig) for k, v in f.items()}
    return p, f, elig, ranks


def run(kline, universe, basic):
    p, f, elig, ranks = prepare(kline, universe, basic)
    out = trade_outcomes(p)
    label_mask = elig & out["buyable"]
    weights, stats = model.fit_weights(ranks, out["day_ret"], label_mask, config.TRAIN_START, config.TRAIN_END)
    # 样本外 IC 只用于报告，不参与任何选择
    for name, r in ranks.items():
        ic = model.daily_ic(r, out["day_ret"], label_mask).loc[config.TEST_START:].dropna()
        stats[name]["test_ic_mean"] = float(ic.mean()) if len(ic) else None
    sc = model.score(ranks, weights, elig)
    fear = model.load_fear(p.dates)

    end = p.dates[-1]
    grid = []
    for n in config.TOP_N_GRID:
        for floor in config.FEAR_FLOOR_GRID:
            picks = model.pick_mask(sc, n, fear, floor)
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
    return dict(p=p, out=out, sc=sc, fear=fear, grid=grid, chosen=chosen, baseline=baseline, model=m)


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
        f"数据截至 {end}。股票池：沪深300 + 中证500 历史成分（每半年快照）。"
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

    picks = model.pick_mask(res["sc"], c["top_n"], res["fear"], c["fear_floor"])
    lines += ["## 样本外逐年", "", "| 年份 | 交易笔数 | 当日扣费浮盈率 | T+1开盘卖出胜率 | 每笔平均 | 年化 |", "| --- | --- | --- | --- | --- | --- |"]
    for r in yearly(picks, res["out"], config.TEST_START, res["p"].dates[-1]):
        lines.append(f"| {r['year']} | {r['trades']} | {_pct(r['hit_net'])} | {_pct(r['hit_next_open_net'])} | "
                     f"{_pct(r['avg_next_open_net'], 2)} | {_pct(r['ann_return'])} |")

    lines += ["", "## 因子（训练期 IC 定权重，|t| < 2 的不用）", "",
              "| 因子 | 含义 | 训练期 IC | t 值 | 权重 | 样本外 IC |", "| --- | --- | --- | --- | --- | --- |"]
    for k, desc in FACTORS.items():
        s = m["factor_stats"][k]
        tic = s.get("test_ic_mean")
        lines.append(f"| {k} | {desc} | {s['ic_mean']:.3f} | {s['t']:.1f} | {m['weights'][k]:+.2f} | "
                     f"{'—' if tic is None else f'{tic:.3f}'} |")
    return "\n".join(lines) + "\n"


def page_data(res):
    """面板用的折线数据：收益曲线、滚动胜率、选股数量与成功率。"""
    c, b = res["chosen"], res["baseline"]
    picks = model.pick_mask(res["sc"], c["top_n"], res["fear"], c["fear_floor"])
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
    )


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "backtest.md").write_text(to_markdown(res))
    (config.PAGE_DIR / "backtest.json").write_text(json.dumps(page_data(res), ensure_ascii=False))
