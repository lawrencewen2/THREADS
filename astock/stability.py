"""线上规则的稳定性检验（2026-10-28 用户：要稳定的盈利模式，胜率也要保持在一定水平）。

线上规则是“业绩预期 + 涨停情绪开关”的 20 只持仓（权重见 astock/forecast_model.json，只用 2015–2020 定）。
这里不改因子权重，只比较几种事先写死的组合做法：持股更分散、换手更低、贪婪时不持仓。
每种做法都报训练期和样本外的年化、最大回撤、每笔胜率、赚钱月份占比、分年收益和最差的滚动 12 个月。
挑选只看训练期：先要求训练期每笔胜率 ≥ 50%，再取夏普最高的；没有达标的就只取夏普最高的。样本外只检验。
"""

import json
import logging

import numpy as np
import pandas as pd

from . import config, model, portfolio
from .alt_study import LIVE_FILE, emotion_regime, iter_alt_factors
from .factors import Panel, eligible

log = logging.getLogger(__name__)

MIN_TRADE_WIN = 0.50
GREED = 70  # 恐慌指数 ≥ 70 视为贪婪
# (名称, 持股数, 缓冲倍数, 开关)
VARIANTS = [
    ("线上规则：20 只，跌出前 80 卖，情绪开关", 20, 4, "emotion"),
    ("30 只，跌出前 120 卖，情绪开关", 30, 4, "emotion"),
    ("50 只，跌出前 200 卖，情绪开关", 50, 4, "emotion"),
    ("20 只，跌出前 160 卖（换手更低），情绪开关", 20, 8, "emotion"),
    ("20 只，跌出前 80 卖，情绪开关且不贪婪", 20, 4, "emotion_nogreed"),
    ("30 只，跌出前 240 卖，情绪开关且不贪婪", 30, 8, "emotion_nogreed"),
    ("对照：20 只，跌出前 80 卖，不择时", 20, 4, None),
]


# 控回撤：在线上规则的每日收益上叠加仓位规则，都只用前一天收盘已知的信息
OVERLAYS = [
    ("线上规则 + 波动率目标（年化 15%，最高满仓）", "vol15"),
    ("线上规则 + 自身净值低于 20 日均线时空仓", "trend20"),
    ("线上规则 + 回撤超过 10% 时半仓，回到 20 日均线上方恢复", "dd10"),
]
SWITCH_COST = config.SLIPPAGE * 2 + config.COMMISSION * 2 + config.STAMP_DUTY_NEW  # 整体进出一次的成本


def overlay(daily, kind):
    """返回叠加仓位后的每日收益。仓位按 t 日收盘后的信息定、作用在下一天；仓位变化按比例扣一次进出成本。"""
    eq = (1 + daily).cumprod()
    if kind == "vol15":
        vol = daily.rolling(20, min_periods=10).std() * np.sqrt(244)
        pos = (0.15 / vol).clip(upper=1.0).fillna(1.0)
    elif kind == "trend20":
        pos = (eq >= eq.rolling(20, min_periods=10).mean()).astype(float)
    elif kind == "dd10":
        dd = eq / eq.cummax() - 1
        above = eq >= eq.rolling(20, min_periods=10).mean()
        state, out = 1.0, []
        for d, a in zip(dd.to_numpy(), above.to_numpy()):
            if d < -0.10:
                state = 0.5
            elif a:
                state = 1.0
            out.append(state)
        pos = pd.Series(out, index=daily.index)
    else:
        raise ValueError(kind)
    pos = pos.shift(1).fillna(1.0)
    return daily * pos - pos.diff().abs().fillna(0) * SWITCH_COST / 2


def period_stats(daily, trades):
    st = portfolio.stats(daily, trades)
    eq = st.pop("_equity", None)
    if eq is None or daily.empty:
        return st
    monthly = (1 + daily).resample("ME").prod() - 1
    yearly = (1 + daily).resample("YE").prod() - 1
    roll = eq / eq.shift(244) - 1
    st.update(month_win=float((monthly > 0).mean()), worst_month=float(monthly.min()),
              worst_12m=float(roll.min()) if roll.notna().any() else None,
              yearly={str(d.year): round(float(v), 4) for d, v in yearly.items()})
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
    ctx = portfolio.context(p)
    fear = model.load_fear(p.dates)
    emotion = emotion_regime(p, elig)
    gates = {"emotion": emotion, "emotion_nogreed": emotion & ~(fear >= GREED), None: None}
    end = p.dates[-1]
    periods = dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, end))

    rows = []
    for name, n, buffer, gate in VARIANTS:
        row = dict(variant=name, n=n, buffer=buffer, gate=gate)
        for k, (s, e) in periods.items():
            d, t = portfolio.simulate(ctx, sc, fear, n, buffer, None, s, e, regime=gates[gate])
            row[k] = period_stats(d, t)
            if k == "test":
                eq = (1 + d).cumprod()
                row["equity"] = {str(i.date()): round(float(v), 4) for i, v in eq.iloc[::5].items()}
        log.info("%s train %s test %s", name, row["train"].get("ann_return"), row["test"].get("ann_return"))
        rows.append(row)
        if name.startswith("线上规则"):
            for oname, kind in OVERLAYS:
                orow = dict(variant=oname, n=n, buffer=buffer, gate=gate, overlay=kind)
                for k, (s, e) in periods.items():
                    d, t = portfolio.simulate(ctx, sc, fear, n, buffer, None, s, e, regime=gates[gate])
                    orow[k] = period_stats(overlay(d, kind), t)  # 胜率仍是每笔的（仓位不改变买卖哪只）
                rows.append(orow)
    ok = [r for r in rows if r["gate"] and (r["train"].get("trade_win") or 0) >= MIN_TRADE_WIN]
    pool = ok or [r for r in rows if r["gate"]]
    best = max(pool, key=lambda r: r["train"].get("sharpe", -9))
    # 控回撤：训练期最大回撤最小、且年化为正的
    calm_pool = [r for r in rows if r["gate"] and r["train"].get("ann_return", -1) > 0]
    calm = max(calm_pool or rows, key=lambda r: r["train"].get("max_drawdown", -9))
    bench = {k: period_stats(portfolio.market(ctx, elig, None, s, e), []) for k, (s, e) in periods.items()}
    return dict(updated=str(end.date()), rows=rows, chosen=best["variant"], chosen_meets_win=bool(ok), calmest=calm["variant"],
                bench=bench, min_trade_win=MIN_TRADE_WIN)


def _p(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def to_markdown(res):
    lines = ["# 线上规则稳定性检验：业绩预期 + 情绪开关", "",
             f"数据截至 {res['updated']}。因子权重只用 2015–2020 定；下面的做法都是事先写死的，"
             f"只在训练期挑：先要求每笔胜率 ≥ {res['min_trade_win'] * 100:.0f}%，再取夏普最高。样本外 2021 至今只检验。"
             "收益扣佣金、双边滑点和印花税；开盘涨停买不进、跌停卖不掉。", "",
             f"训练期挑出的做法：**{res['chosen']}**"
             + ("" if res["chosen_meets_win"] else f"（没有做法在训练期达到 {res['min_trade_win'] * 100:.0f}% 胜率，只按夏普挑）"), "",
             f"训练期最大回撤最小（且年化为正）的做法：**{res['calmest']}**", "",
             "控回撤的几种是在线上规则的每日收益上调仓位，不改变买卖哪只，所以每笔胜率和线上规则一样。", "",
             "| 做法 | 训练期年化 | 训练期胜率 | 样本外年化 | 样本外夏普 | 样本外最大回撤 | 样本外每笔胜率 | 赚钱月份占比 | 最差月份 | 最差滚动 12 个月 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in res["rows"]:
        a, t = r["train"], r["test"]
        lines.append(f"| {r['variant']} | {_p(a.get('ann_return'))} | {_p(a.get('trade_win'), 0)} | {_p(t.get('ann_return'))} | "
                     f"{t.get('sharpe', 0):.2f} | {_p(t.get('max_drawdown'))} | {_p(t.get('trade_win'), 0)} | "
                     f"{_p(t.get('month_win'), 0)} | {_p(t.get('worst_month'))} | {_p(t.get('worst_12m'))} |")
    b = res["bench"]["test"]
    lines.append(f"| 对照：等权持有全市场（不扣费） | {_p(res['bench']['train'].get('ann_return'))} | — | {_p(b.get('ann_return'))} | "
                 f"{b.get('sharpe', 0):.2f} | {_p(b.get('max_drawdown'))} | — | {_p(b.get('month_win'), 0)} | "
                 f"{_p(b.get('worst_month'))} | {_p(b.get('worst_12m'))} |")
    years = sorted({y for r in res["rows"] for y in r["test"].get("yearly", {})} | set(b.get("yearly", {})))
    lines += ["", "## 样本外分年收益", "", "| 做法 | " + " | ".join(years) + " |", "| --- |" + " --- |" * len(years)]
    for r in res["rows"] + [dict(variant="等权持有全市场", test=b)]:
        y = r["test"].get("yearly", {})
        lines.append(f"| {r['variant']} | " + " | ".join(_p(y.get(k)) for k in years) + " |")
    lines += ["", "仅供个人研究参考，不构成投资建议。"]
    return "\n".join(lines) + "\n"


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "stability.md").write_text(to_markdown(res))
    (config.PAGE_DIR / "stability.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
