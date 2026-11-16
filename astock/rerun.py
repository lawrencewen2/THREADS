"""修正同日腾位前视后的全量重跑（2026-11-16 量化小组第 2 轮，用户批准“修复”）。

对照四种口径（训练 2015–2020，样本外 2021 至今）：
- 旧口径（same_day_refill=True）vs 修正后；
- 标准成本（滑点 0.1%/边）vs 保守成本（滑点 0.2%/边，小组以后统一同时报告）；
另外在修正后重新检查“业绩加速”因子是否还有用（不加 vs 加 0.3）。
选股层按线上口径（净值跌破 20 日均线空仓）统计；总组合按线上方案（选股 65% + ETF 分仓闲钱停泊）用
redline.load_part / redline.simulate 合成。逐日收益写 data/research/rerun_daily.csv.gz，修正后的逐笔交易写
data/research/rerun_trades.csv.gz，供风控、研究员后续使用。
"""

import json
import logging

import pandas as pd

from . import config, diversify as dv, model, redline
from .alt_study import emotion_regime, iter_alt_factors
from .factor_hunt import NEW, _summ, forecast_extras
from .factors import Panel, eligible
from .live_slots import MODEL_FILE
from .slots import simulate
from .winrate import adjusted_ohlc

log = logging.getLogger(__name__)

OUT_DAILY = dv.RESEARCH_DIR / "rerun_daily.csv.gz"
OUT_TRADES = dv.RESEARCH_DIR / "rerun_trades.csv.gz"
OUT_JSON = config.PAGE_DIR / "rerun.json"
OUT_MD = config.REPORT_DIR / "rerun.md"
VARIANTS = [  # 名字, 打分, 旧口径?, 滑点
    ("old_std", "accel", True, 0.001),
    ("fixed_std", "accel", False, 0.001),
    ("fixed_cons", "accel", False, 0.002),
    ("fixed_std_noaccel", "base", False, 0.001),
    ("fixed_cons_noaccel", "base", False, 0.002),
]
LABELS = {
    "old_std": "修正前（旧口径），标准成本",
    "fixed_std": "修正后，标准成本",
    "fixed_cons": "修正后，保守成本（滑点 0.2%/边）",
    "fixed_std_noaccel": "修正后，不加业绩加速，标准成本",
    "fixed_cons_noaccel": "修正后，不加业绩加速，保守成本",
}


def scores(kline, basic, valuation, forecast):
    sm = json.loads(MODEL_FILE.read_text())
    p = Panel(kline, valuation)
    elig = eligible(p, basic)
    weights = {k: v for k, v in sm["weights"].items() if v}
    ranks = {k: model.xs_rank(v, elig) for k, v in iter_alt_factors(p, forecast) if k in weights}
    base = model.score(ranks, weights, elig)
    ex = forecast_extras(p, forecast)
    accel = base.copy()
    for k, w in (sm.get("blend") or {}).items():
        accel = accel + w * (model.xs_rank(ex[k], elig) * NEW[k][1]).fillna(0)
    accel = accel.where(base.notna())
    return p, elig, sm, dict(base=base, accel=accel)


def run(kline, basic, valuation=None, forecast=None):
    kline = kline() if callable(kline) else kline
    valuation = valuation() if callable(valuation) else valuation
    forecast = forecast() if callable(forecast) else forecast
    p, elig, sm, sc = scores(kline, basic, valuation, forecast)
    del kline
    gate = emotion_regime(p, elig)
    ohl = adjusted_ohlc(p)
    tp, sl, hold, n = sm["take_profit"], sm["stop_loss"], sm["max_hold"], sm["n"]
    end = str(p.dates[-1].date())
    periods = dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, end))
    cols, rows, trades_out = {}, [], []
    slip0 = config.SLIPPAGE
    rets = dv.etf_returns()
    sleeve = dv.sleeve_returns(rets, sm["sleeve"]["codes"], sm["sleeve"].get("rebalance", 20))
    try:
        for name, key, old, slip in VARIANTS:
            config.SLIPPAGE = slip
            row = dict(name=name, label=LABELS[name])
            parts = []
            for per, (s, e) in periods.items():
                d, t = simulate(p, ohl, sc[key], gate, n, tp, sl, hold, s, e, same_day_refill=old)
                row[per] = _summ(d, t)
                held = pd.Series(0, index=d.index)
                for x in t:
                    held[(held.index >= x["entry"]) & (held.index < x["exit"])] += 1
                parts.append(pd.DataFrame({name + "|raw": d, name + "|held": held}))
                if name == "fixed_std":
                    trades_out += [dict(period=per, code=x["code"], entry=x["entry"].date(), exit=x["exit"].date(),
                                        net=round(float(x["net"]), 5)) for x in t]
            df = pd.concat(parts)
            for c in df:
                cols[c] = df[c]
            tot = redline.load_part(name, frame=df)
            for per in ("train", "test"):
                r, _, _ = redline.simulate(tot[per], sleeve, None)
                row[per + "_total"] = redline.stats(r)
                row[per + "_total_yearly"] = {str(y): float((1 + g).prod() - 1) for y, g in r.groupby(r.index.year)}
            rows.append(row)
            log.info("%s: %s / %s", name, row["test"], row["test_total"])
    finally:
        config.SLIPPAGE = slip0
    out = pd.DataFrame(cols)
    out.index.name = "date"
    OUT_DAILY.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DAILY, float_format="%.6g")
    pd.DataFrame(trades_out).to_csv(OUT_TRADES, index=False)
    return dict(updated=end, rows=rows)


def _p(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def to_markdown(res):
    L = [f"# 修正同日腾位后的全量重跑（数据到 {res['updated']}）", "",
         "修正：盘中止盈/止损腾出的仓位第二天开盘才能买，当天卖掉的股票当天不买回（astock/slots.py）。"
         "训练期 2015–2020，样本外 2021 至今。已扣成本。", "",
         "## 总组合（选股 65% + ETF 分仓闲钱停泊）", "",
         "| 口径 | 训练年化 | 训练回撤 | 样本外年化 | 样本外回撤 | 样本外夏普 |", "|---|---|---|---|---|---|"]
    for r in res["rows"]:
        a, b = r["train_total"], r["test_total"]
        L.append(f"| {r['label']} | {_p(a['ann_return'])} | {_p(a['max_drawdown'])} | {_p(b['ann_return'])} | "
                 f"{_p(b['max_drawdown'])} | {b['sharpe']:.2f} |")
    L += ["", "## 选股层（100% 资金口径，含 20 日均线空仓）", "",
          "| 口径 | 训练年化 | 训练回撤 | 训练胜率 | 样本外年化 | 样本外回撤 | 样本外胜率 | 样本外笔数 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        a, b = r["train"], r["test"]
        L.append(f"| {r['label']} | {_p(a['ann_return'])} | {_p(a['max_drawdown'])} | {_p(a['trade_win'])} | "
                 f"{_p(b['ann_return'])} | {_p(b['max_drawdown'])} | {_p(b['trade_win'])} | {b['trades']} |")
    by = {r["name"]: r for r in res["rows"]}
    if "fixed_std" in by:
        L += ["", "## 修正后逐年（总组合，标准成本）", "", "| 年份 | 收益 |", "|---|---|"]
        yr = by["fixed_std"]["train_total_yearly"] | by["fixed_std"]["test_total_yearly"]
        L += [f"| {y} | {_p(v)} |" for y, v in sorted(yr.items())]
    return "\n".join(L) + "\n"


def write(res):
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    OUT_MD.write_text(to_markdown(res))
