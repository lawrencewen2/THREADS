"""改进选股的对照实验（2026-10-27 用户看到 5 只候选全跌后要求改正）。

现有模型的问题和对应的改法，每种都只在训练期（2015–2020）挑参数，样本外（2021 以后）只看不调：
1. 目标不对：现在按“T 日开盘买到 T 日收盘”定权重，但 T+1 才能卖，真正拿到的是 T+1 开盘价。
   改法：按 T+1 开盘卖出的扣费收益（next_open_net）定权重。
2. 不看大盘：现在几乎每天都买。改法：全市场等权指数在 20 日线下方（或近 5 日下跌）时空仓。
3. 接飞刀：权重偏向近期跌得多、收在最低点附近的小票。改法：剔除前一天跌超 5%、连跌 3 天以上、
   或收盘在当日最低 10% 区间的股票。
"""

import json
import logging

import numpy as np
import pandas as pd

from . import config, model
from .factors import Panel, eligible, iter_factors, trade_outcomes

log = logging.getLogger(__name__)

TARGETS = ("day_ret", "next_open_net")
GATES = ("none", "ma20", "ret5")
KNIFE = (False, True)
TOP_N = (3, 5, 10)


def market_gate(p, elig, kind):
    """信号日 t 收盘后可知：全市场等权指数的状态。True = 当天可以买。"""
    ret = (p.close / p.preclose - 1).where(p.traded & elig)
    idx = (1 + ret.mean(axis=1).fillna(0)).cumprod()
    if kind == "ma20":
        ok = idx >= idx.rolling(20, min_periods=15).mean()
    elif kind == "ret5":
        ok = idx / idx.shift(5) - 1 >= 0
    else:
        ok = pd.Series(True, index=p.dates)
    return ok.fillna(False)


def knife_mask(p):
    """True = 不是飞刀，可以选。"""
    ret = p.close / p.preclose - 1
    rng = (p.high - p.low).where(p.high > p.low)
    clv = (p.close - p.low) / rng
    down = (ret < 0).astype("int8")
    streak3 = down.rolling(3, min_periods=3).sum() >= 3
    return (ret > -0.05) & ~streak3 & ~(clv < 0.1)


def run(kline, basic, valuation=None):
    """kline / valuation 可以传读取函数，建好面板就释放原始行情。"""
    kline = kline() if callable(kline) else kline
    valuation = valuation() if callable(valuation) else valuation
    p = Panel(kline, valuation)
    del kline, valuation
    elig = eligible(p, basic)
    ranks = {k: model.xs_rank(v, elig) for k, v in iter_factors(p)}
    out = trade_outcomes(p)
    label_mask = elig & out["buyable"]
    end = p.dates[-1]
    train_end = p.dates[p.dates <= pd.Timestamp(config.TRAIN_END)][-3]
    knife_ok = knife_mask(p)
    gates = {g: market_gate(p, elig, g) for g in GATES}
    base = model.evaluate(elig & out["buyable"], out, config.TEST_START, end)
    rows = []
    for target in TARGETS:
        w, _ = model.fit_weights(ranks, out[target], label_mask, config.TRAIN_START, train_end)
        if not any(w.values()):
            log.warning("%s: no factor passes |t|>=2, skipped", target)
            continue
        for knife in KNIFE:
            m = elig & knife_ok if knife else elig
            sc = model.score(ranks, w, m)
            order = sc.rank(axis=1, ascending=False, method="first")
            for g in GATES:
                for n in TOP_N:
                    picks = (order <= n) & sc.notna() & gates[g].values[:, None]
                    tr = model.evaluate(picks, out, config.TRAIN_START, train_end)
                    te = model.evaluate(picks, out, config.TEST_START, end)
                    rows.append(dict(target=target, knife=knife, gate=g, top_n=n,
                                     weights={k: round(v, 4) for k, v in w.items() if v},
                                     train=model.public(tr), test=model.public(te)))
                    log.info("%s knife=%s gate=%s n=%d train %.4f test %.4f", target, knife, g, n,
                             tr.get("avg_next_open_net", np.nan), te.get("avg_next_open_net", np.nan))
    ok = [r for r in rows if r["train"].get("trade_day_share", 0) >= config.MIN_TRADE_DAY_SHARE] or rows
    chosen = max(ok, key=lambda r: r["train"].get("avg_next_open_net", -1))
    current = next((r for r in rows if r["target"] == "day_ret" and not r["knife"] and r["gate"] == "none"
                    and r["top_n"] == 5), chosen)
    return dict(end=str(end.date()), rows=rows, chosen=chosen, current=current, baseline=model.public(base))


def _p(x, d=2):
    return "—" if x is None else f"{x * 100:.{d}f}%"


def to_markdown(res):
    lines = ["# 选股改进对照实验", "",
             f"数据截至 {res['end']}。参数只在 2015–2020 训练期挑，2021 年以后是样本外。收益是 T 开盘买、T+1 开盘卖、扣费后。", "",
             "| 目标 | 剔除飞刀 | 大盘开关 | 只数 | 训练期每笔 | 样本外每笔 | 样本外胜率 | 样本外年化 | 样本外最大回撤 | 交易天数占比 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(res["rows"], key=lambda r: -(r["test"].get("avg_next_open_net") or -1)):
        tr, te = r["train"], r["test"]
        lines.append(f"| {r['target']} | {'是' if r['knife'] else '否'} | {r['gate']} | {r['top_n']} | "
                     f"{_p(tr.get('avg_next_open_net'))} | {_p(te.get('avg_next_open_net'))} | "
                     f"{_p(te.get('hit_next_open_net'), 1)} | {_p(te.get('ann_return'), 1)} | "
                     f"{_p(te.get('max_drawdown'), 1)} | {_p(te.get('trade_day_share'), 0)} |")
    c, cur, b = res["chosen"], res["current"], res["baseline"]
    lines += ["", f"训练期挑出的组合：目标 {c['target']}，剔除飞刀 {'是' if c['knife'] else '否'}，大盘开关 {c['gate']}，"
              f"每天 {c['top_n']} 只。样本外每笔 {_p(c['test'].get('avg_next_open_net'))}，"
              f"年化 {_p(c['test'].get('ann_return'), 1)}，最大回撤 {_p(c['test'].get('max_drawdown'), 1)}。",
              f"现在线上的组合样本外每笔 {_p(cur['test'].get('avg_next_open_net'))}，年化 {_p(cur['test'].get('ann_return'), 1)}。",
              f"全部可买股票平均每笔 {_p(b.get('avg_next_open_net'))}。"]
    return "\n".join(lines)


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "fix_study.md").write_text(to_markdown(res))
    (config.PAGE_DIR / "fix_study.json").write_text(json.dumps(res, ensure_ascii=False, default=str))
