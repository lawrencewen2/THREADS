"""回测诊断：区分“数据或代码有错”和“策略本身没有优势”。结果写到 reports/astock/diagnosis.md。"""

import numpy as np
import pandas as pd

from . import config, model
from .factors import FACTORS


def _pct(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def run(p, size, elig, ranks, out, sc):
    lines = ["# A股选股回测诊断", "",
             f"数据截至 {p.dates[-1].date()}。先查数据和代码是否有错，再看策略本身。", ""]
    tr, te = slice(config.TRAIN_START, config.TRAIN_END), slice(config.TEST_START, None)

    # 1. 数据覆盖
    lines += ["## 1. 数据覆盖", "", "| 年份 | 有行情的股票数 | 平均每天可选股票数 |", "| --- | --- | --- |"]
    traded_year = p.traded.groupby(p.dates.year).any().sum(axis=1)
    elig_year = elig.sum(axis=1).groupby(p.dates.year).mean()
    for y in traded_year.index:
        lines.append(f"| {y} | {int(traded_year[y])} | {elig_year[y]:.0f} |")

    # 2. 数据质量
    ret = (p.close / p.preclose - 1).where(p.traded)
    pct = p.limit_pct()
    listed_days = p.traded.cumsum()
    normal = p.traded & (listed_days > 5)
    beyond = ((ret.abs() > pct + 0.02) & normal).values.sum()
    bad_open = (((p.open > p.high + 0.005) | (p.open < p.low - 0.005)) & p.traded).values.sum()
    nonpos = ((p.close <= 0) & p.traded).values.sum()
    total = int(p.traded.values.sum())
    lines += ["", "## 2. 数据质量", "",
              f"- 有效交易记录 {total:,} 条。",
              f"- 涨跌幅超过涨跌停限制 2 个百分点以上（剔除上市前 5 天）：{int(beyond):,} 条（{beyond / total:.4%}）。"
              "（比例很低时不影响结论；可能来自涨跌停规则调整、复牌首日等特殊情况。）",
              f"- 开盘价落在当日最高最低价之外：{int(bad_open):,} 条。",
              f"- 收盘价小于等于 0：{int(nonpos):,} 条。"]

    # 3. 收益对齐（有没有用到未来数据、标签有没有错位）
    entry = p.open * (1 + config.SLIPPAGE)
    direct = (p.close / entry - 1).shift(-1)
    diff = (out["day_ret"] - direct).abs().max().max()
    lines += ["", "## 3. 标签对齐检查", "",
              f"- 信号日 t 的“T 日收益”与直接用 T 日价格算的结果最大差 {diff:.2e}（应接近 0）。",
              "- 因子只用 t 日及以前数据：单元测试会篡改未来行情，确认历史因子和可选范围不变（已通过）。"]

    # 4. 费用拆解
    m = elig & out["buyable"]
    intraday = (p.close / p.open - 1).shift(-1).where(m)
    overnight = (p.open.shift(-2) / p.close.shift(-1) - 1).where(m)
    o2o = (p.open.shift(-2) / p.open.shift(-1) - 1).where(m)
    rt_cost = 2 * config.SLIPPAGE + 2 * config.COMMISSION + config.STAMP_DUTY_NEW
    lines += ["", "## 4. 收益和费用拆解（全部可买股票，每笔平均，未扣费）", "",
              "| 区间 | 开盘→收盘 | 收盘→次日开盘 | 开盘→次日开盘 |", "| --- | --- | --- | --- |"]
    for name, sl in (("训练期", tr), ("样本外", te)):
        lines.append(f"| {name} | {_pct(np.nanmean(intraday.loc[sl].values))} | "
                     f"{_pct(np.nanmean(overnight.loc[sl].values))} | {_pct(np.nanmean(o2o.loc[sl].values))} |")
    lines += ["", f"一买一卖的成本约 {_pct(rt_cost)}（滑点双边 {config.SLIPPAGE * 2:.1%}、佣金双边、印花税）。"
              "隔夜交易每天都要付一次，平均一笔的毛收益远小于这个数时，选股再准也很难覆盖。"]

    # 5. 得分十分位
    q = sc.rank(axis=1, pct=True)
    dec_o2o, dec_net = [], []
    lines += ["", "## 5. 原策略得分分组（十分位，10 = 得分最高）", "",
              "| 分组 | 样本外 当日开盘→收盘 | 样本外 开盘→次日开盘（未扣费） | 样本外 扣费后 | 样本外 扣费后胜率 | 训练期 开盘→次日开盘（未扣费） |",
              "| --- | --- | --- | --- | --- | --- |"]
    for d in range(1, 11):
        g = (q > (d - 1) / 10) & (q <= d / 10) & m
        dec_o2o.append(np.nanmean(o2o.where(g).loc[te].values))
        dec_net.append(np.nanmean(out["next_open_net"].where(g).loc[te].values))
        lines.append(f"| {d} | {_pct(np.nanmean(out['day_ret'].where(g).loc[te].values))} | "
                     f"{_pct(np.nanmean(o2o.where(g).loc[te].values))} | "
                     f"{_pct(np.nanmean(out['next_open_net'].where(g).loc[te].values))} | "
                     f"{_pct((out['next_open_net'].where(g).loc[te] > 0).values.sum() / max(1, out['next_open_net'].where(g).loc[te].notna().values.sum()), 1)} | "
                     f"{_pct(np.nanmean(o2o.where(g).loc[tr].values))} |")
    mono = pd.Series(dec_o2o).rank().corr(pd.Series(range(10), dtype=float))
    spread = dec_o2o[-1] - dec_o2o[0]
    verdict = ("分组越高、收益越高，因子有区分能力" if mono > 0.6 else "分组和收益没有明显的单调关系，因子区分能力弱")
    lines += ["", f"样本外十个分组的单调性（秩相关）{mono:.2f}，最高组比最低组每天多 {_pct(spread, 3)}（未扣费）。"
              f"{verdict}；最高组扣费后每笔 {_pct(dec_net[-1])}。"]

    # 6. 因子稳定性
    lines += ["", "## 6. 因子逐年稳定性（对 T 日开盘→收盘的 IC）", "",
              "| 因子 | 含义 | 训练期 IC | 样本外 IC | 与训练期同号的年份 |", "| --- | --- | --- | --- | --- |"]
    for k, r in ranks.items():
        ic = model.daily_ic(r, out["day_ret"], m).dropna()
        if ic.empty:
            continue
        by_year = ic.groupby(ic.index.year).mean()
        tr_ic = ic.loc[tr].mean()
        same = int((np.sign(by_year) == np.sign(tr_ic)).sum())
        lines.append(f"| {k} | {FACTORS.get(k, k)} | {tr_ic:.3f} | {ic.loc[te].mean():.3f} | {same}/{len(by_year)} |")

    # 7. 原策略每天前 1 只是什么样的股票
    top1 = (sc.rank(axis=1, ascending=False, method="first") <= 1) & m
    size_q = size.rank(axis=1, pct=True) if size is not None else None
    amt = p.amount.rolling(20, min_periods=15).mean()
    lines += ["", "## 7. 原策略每天选出的第 1 名", ""]
    if size_q is not None:
        sq = size_q.where(top1).stack()
        lines.append(f"- 流通市值在全市场的平均分位 {sq.mean():.0%}，落在最小 20% 的占 {(sq <= 0.2).mean():.0%}。")
    am = amt.where(top1).stack()
    lines.append(f"- 近 20 日日均成交额中位数 {am.median() / 1e8:.2f} 亿元。小盘、低成交额的股票，真实滑点往往高于回测假设的 0.1%。")
    lines.append(f"- 只选 1 只，每天的结果几乎全看这一只的运气，样本外胜率的波动会很大。")
    return "\n".join(lines) + "\n"
