"""主升浪选股 + 按市场阶段控仓位的回测（2026-11-01 用户：学习主升浪的选股原理，加入规则回测；
主升浪的时候满仓，震荡期、埋伏期低仓位试仓；学习网上出名的金融书籍，加入策略）。

把几本经典书里能用日线数据写死的规则拿来检验（都只用信号日 t 收盘前的数据，T 开盘买）：
- 米内尔维尼《股票魔法师》趋势模板：收盘在 50/150/200 日均线上方且均线多头、200 日线上行一个月、
  离 52 周低点至少 +30%、离 52 周高点不超过 25%、相对强度（RS）排名前 30%；
- 米内尔维尼 VCP（波动收缩）：趋势模板内，近 10 日振幅明显小于近 50 日、成交缩量，然后放量突破；
- 达瓦斯箱体 / 利弗莫尔关键点：站上 50 日均线、收盘突破前 20 日箱体高点，成交（换手）放大 1.5 倍以上；
- 温斯坦《笨人在股市赚钱的秘密》第二阶段：收盘在上行的 30 周（150 日）均线上方，突破 60 日高点，换手放大 2 倍以上；
- 欧奈尔 CANSLIM：C/A 用线上的业绩预期得分代替（排名前一半），N 离 52 周高点 15% 以内，
  S 换手放大，L 相对强度前 20%，M 交给市场阶段开关；I（机构持仓）没有数据，略过；
- A股常说的主升浪：均线多头（5>10>20>60 日，60 日线上行），放量突破 60 日平台，近 20 日涨幅不超过 40%（不追太高）。

卖出（每笔买入前定好止损价和止盈价）：
- 线上：止盈 2 倍 ATR、止损 3 倍 ATR、最长 10 天；
- 欧奈尔：止损 8%、止盈 25%、最长 60 天；
- 米内尔维尼：止损 7%、止盈 20%，涨过 10% 后止损提到成本价，最长 40 天；
- 趋势跟踪（让利润奔跑）：止损 3 倍 ATR、最高价回落 3 倍 ATR 移动止损、止盈 8 倍 ATR，最长 60 天。

市场阶段（t 收盘可知，T 开盘用）：主升浪满仓（20 只都可以买），震荡/埋伏期只用一部分仓位试仓，下跌期空仓或极少。
三种判断办法：大盘均线、市场广度（站上 50 日线的股票占比）、欧奈尔派发日。

规则只在训练期（2015–2020）挑：先要每笔胜率 ≥ 50%，再比夏普；样本外（2021 至今）只检验。
"""

import json
import logging

import numpy as np
import pandas as pd

from . import config, model
from .alt_study import emotion_regime, iter_alt_factors
from .factors import Panel, eligible
from .live_slots import MODEL_FILE
from .slots import GOAL_WIN, stats
from .stability import overlay
from .winrate import adjusted_ohlc

log = logging.getLogger(__name__)

N = 20
SELECTIONS = [
    ("线上规则：业绩预期 + 情绪开关（对照）", "live"),
    ("线上规则 + 只买符合趋势模板的", "live_tt"),
    ("米内尔维尼趋势模板，按相对强度排", "tt"),
    ("米内尔维尼 VCP 收缩后放量突破", "vcp"),
    ("达瓦斯箱体 / 利弗莫尔关键点突破", "darvas"),
    ("温斯坦第二阶段突破", "stage2"),
    ("欧奈尔 CANSLIM（业绩 + 新高 + 强势 + 放量）", "canslim"),
    ("A股主升浪：均线多头 + 放量突破平台", "zsl"),
]
# (名称, 代号, 止盈, 止损, 移动止损, 保本线, 最长天数)
EXITS = [
    ("线上：止盈 2 倍 ATR、止损 3 倍 ATR、最长 10 天", "live", "atr2", "atr3", None, None, 10),
    ("欧奈尔：止损 8%、止盈 25%、最长 60 天", "oneil", 0.25, 0.08, None, None, 60),
    ("米内尔维尼：止损 7%、止盈 20%、涨 10% 后保本，最长 40 天", "minervini", 0.20, 0.07, None, 0.10, 40),
    ("趋势跟踪：止损 3 倍 ATR、回落 3 倍 ATR 移动止损、止盈 8 倍 ATR，最长 60 天", "trend", "atr8", "atr3", 3.0, None, 60),
]
REGIMES = [
    ("大盘均线：等权指数在 20 日线和上行的 60 日线上方为主升浪，在下行的 60 日线下方为下跌", "ma"),
    ("市场广度：站上 50 日线的股票 ≥ 60% 为主升浪，≤ 30% 为下跌", "breadth"),
    ("欧奈尔派发日：沪深 300 在 50 日线上方、近 25 天派发日 ≤ 4 为主升浪；跌破 50 日线且派发日 ≥ 6 为下跌", "dist"),
]
# (主升浪, 震荡/埋伏, 下跌) 的仓位
LEVELS = [(1.0, 0.3, 0.0), (1.0, 0.3, 0.1), (1.0, 0.5, 0.0)]
STAGE2_TOP = 3  # 第一轮训练期最好的几个选股 × 卖出组合，进第二轮加市场阶段仓位


def _rank(f, mask):
    return model.xs_rank(f, mask) + 0.5  # 0..1


def signals(p, elig, live):
    """各选股方法的得分宽表（NaN = 当天不选）。live 是线上业绩预期得分。"""
    ret = (p.close / p.preclose - 1).where(p.traded)
    adj = (1 + ret.fillna(0)).cumprod().astype("float32")
    k = adj / p.close
    hi, lo = (p.high * k).astype("float32"), (p.low * k).astype("float32")
    turn = p.turn.where(p.turn > 0)

    def ma(w):
        return adj.rolling(w, min_periods=int(w * 0.8)).mean().astype("float32")

    ma5, ma10, ma20, ma50, ma60, ma150, ma200 = (ma(w) for w in (5, 10, 20, 50, 60, 150, 200))
    hi250 = hi.rolling(250, min_periods=200).max()
    lo250 = lo.rolling(250, min_periods=200).min()
    r = sum(wt * (adj / adj.shift(d) - 1) for wt, d in ((0.4, 63), (0.2, 126), (0.2, 189), (0.2, 252)))
    rs = _rank(r.where(adj.shift(252).notna()), elig)  # IBD 式相对强度，近一个季度权重 40%
    del r
    t50 = turn.rolling(50, min_periods=40).mean()
    vol_up = turn / t50
    box20 = hi.shift(1).rolling(20, min_periods=15).max()
    box60 = hi.shift(1).rolling(60, min_periods=50).max()
    tt = ((adj > ma150) & (adj > ma200) & (ma150 > ma200) & (ma200 > ma200.shift(20)) & (ma50 > ma150)
          & (adj > ma50) & (adj >= 1.3 * lo250) & (adj >= 0.75 * hi250) & (rs >= 0.7))
    rng10 = (hi.rolling(10, min_periods=8).max() / lo.rolling(10, min_periods=8).min() - 1)
    rng50 = (hi.rolling(50, min_periods=40).max() / lo.rolling(50, min_periods=40).min() - 1)
    contraction = (rng10 < 0.5 * rng50) & (turn.rolling(10, min_periods=8).mean() < 0.9 * t50)
    del rng10, rng50
    out = {}
    out["live"] = live
    out["live_tt"] = live.where(tt)
    out["tt"] = rs.where(tt & elig)
    out["vcp"] = rs.where(tt & contraction.shift(1, fill_value=False) & (adj > box20) & (vol_up >= 1.5) & elig)
    out["darvas"] = rs.where((adj > ma50) & (adj > box20) & (vol_up >= 1.5) & (rs >= 0.5) & elig)
    out["stage2"] = rs.where((adj > ma150) & (ma150 > ma150.shift(20)) & (adj > box60) & (vol_up >= 2.0) & elig)
    lr = _rank(live, elig)
    out["canslim"] = (lr + rs).where((lr >= 0.5) & (adj >= 0.85 * hi250) & (rs >= 0.8) & (vol_up >= 1.3)
                                    & (adj > ma50) & elig)
    out["zsl"] = rs.where((ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60) & (ma60 > ma60.shift(10)) & (adj > box60)
                          & (vol_up >= 1.5) & (adj / adj.shift(20) - 1 <= 0.4) & elig)
    return out


def market_regimes(p, elig):
    """各判断办法的市场阶段：2 = 主升浪，1 = 震荡/埋伏，0 = 下跌。按信号日 t（收盘后）对齐。"""
    ret = (p.close / p.preclose - 1).where(p.traded & elig)
    mkt = (1 + ret.mean(axis=1).fillna(0)).cumprod()
    m20, m60 = mkt.rolling(20).mean(), mkt.rolling(60).mean()
    up60, dn60 = m60 > m60.shift(5), m60 < m60.shift(5)
    out = {"ma": pd.Series(np.where((mkt > m20) & (m20 > m60) & up60, 2, np.where((mkt < m60) & dn60, 0, 1)), index=p.dates)}
    adj = (1 + (p.close / p.preclose - 1).where(p.traded).fillna(0)).cumprod()
    above = (adj > adj.rolling(50, min_periods=40).mean()).where(elig)
    share = above.sum(axis=1) / elig.sum(axis=1).replace(0, np.nan)
    out["breadth"] = pd.Series(np.where(share >= 0.6, 2, np.where(share <= 0.3, 0, 1)), index=p.dates)
    csi = pd.read_csv(config.ROOT / "data" / "raw" / "csi300.csv", parse_dates=["date"]).set_index("date")["value"]
    csi = csi[~csi.index.duplicated(keep="last")].sort_index().reindex(p.dates).ffill()
    amount = p.amount.where(elig).sum(axis=1)  # 全市场成交额代替指数成交量
    dist = ((csi.pct_change() <= -0.002) & (amount > amount.shift(1))).astype(int).rolling(25, min_periods=1).sum()
    c50 = csi.rolling(50).mean()
    out["dist"] = pd.Series(np.where((csi > c50) & (dist <= 4), 2, np.where((csi < c50) & (dist >= 6), 0, 1)), index=p.dates)
    return out, mkt


def simulate(p, ohl, sc, gate, n, ex, start, end, cap=None):
    """真实账户模拟（同 slots.simulate），多了移动止损、保本线和按市场阶段限制持仓数。

    cap：按信号日对齐的仓位比例（0..1）。T 开盘最多持有 round(n × cap[t]) 只；为 0 时开盘全部卖出。
    比例降低但不为 0 时不强卖，只是不再买新的，靠止损止盈和到期慢慢降下来。"""
    _, _, tp, sl, trail, be, hold = ex
    o, h, l, atr, locked = ohl
    up, _ = p.limit_prices()
    up_v, raw_open, traded = up.to_numpy(), p.open.to_numpy(), p.traded.to_numpy()
    sc_v = sc.to_numpy()
    gate_v = gate.reindex(p.dates).fillna(False).to_numpy(dtype=bool)
    cap_v = None if cap is None else cap.reindex(p.dates).fillna(1.0).to_numpy()
    dates = p.dates
    i0 = max(dates.searchsorted(pd.Timestamp(start)), 15)
    i1 = min(dates.searchsorted(pd.Timestamp(end), side="right"), len(dates) - 1)
    buy_cost = config.SLIPPAGE + config.COMMISSION
    stamp_change = pd.Timestamp(config.STAMP_DUTY_CHANGE)

    def level(x, entry, a, sign):
        if x is None:
            return np.inf if sign > 0 else 0.0
        return entry * (1 + sign * (float(x[3:]) * a if isinstance(x, str) else x))

    held = {}  # 列号 -> [买入价, 止盈价, 止损价, 买入日序号, 买入时 ATR]
    daily, trades, expo = [], [], []
    for T in range(i0, i1):
        stamp = config.STAMP_DUTY_NEW if dates[T] >= stamp_change else config.STAMP_DUTY_OLD
        sell_cost = config.SLIPPAGE + config.COMMISSION + stamp
        limit = n if cap_v is None else int(round(n * cap_v[T - 1]))
        r = 0.0
        for j in list(held):
            entry, target, stop, d0, a = held[j]
            px = None
            if not locked[T, j] and np.isfinite(o[T, j]):
                if T - d0 >= hold or limit == 0:
                    px = o[T, j]
                elif l[T, j] <= stop:
                    px = min(o[T, j], stop)
                elif h[T, j] >= target:
                    px = max(o[T, j], target)
            if px is None:
                if np.isfinite(h[T, j]) and T > d0:  # 收盘后按当天最高价上调止损，下一天起生效
                    if trail:
                        held[j][2] = max(stop, h[T, j] - trail * a * entry)
                    if be and h[T, j] >= entry * (1 + be):
                        held[j][2] = max(held[j][2], entry)
                continue
            r += (px / o[T, j] - 1 - sell_cost) / n
            trades.append(dict(entry=dates[d0], exit=dates[T], code=p.codes[j],
                               net=px * (1 - sell_cost) / (entry * (1 + buy_cost)) - 1))
            del held[j]
        if gate_v[T - 1] and len(held) < limit:
            row = sc_v[T - 1]
            ok = ~np.isnan(row)
            for j in np.argsort(-np.where(ok, row, -np.inf))[:n * 3]:
                if len(held) >= limit or not ok[j]:
                    break
                if j in held or not traded[T, j] or raw_open[T, j] >= up_v[T, j] - 0.005:
                    continue
                a = atr[T - 1, j]
                if not np.isfinite(a) or not np.isfinite(o[T, j]):
                    continue
                e = o[T, j]
                held[j] = [e, level(tp, e, a, 1), level(sl, e, a, -1), T, a]
                r -= buy_cost / n
        for j in held:
            if np.isfinite(o[T + 1, j]) and o[T, j] > 0:
                r += (o[T + 1, j] / o[T, j] - 1) / n
        daily.append((dates[T], r))
        expo.append(len(held) / n)
    d = pd.Series(dict(daily), dtype="float64")
    return d, trades, pd.Series(expo, index=d.index)


def run(kline, basic, valuation=None, forecast=None):
    kline = kline() if callable(kline) else kline
    valuation = valuation() if callable(valuation) else valuation
    forecast = forecast() if callable(forecast) else forecast
    if forecast is None or forecast.empty:
        raise RuntimeError("no forecast data: the live ranking needs 业绩预告")
    sm = json.loads(MODEL_FILE.read_text())
    weights = {k: v for k, v in sm["weights"].items() if v}
    share = sm.get("capital_share", 1.0)
    p = Panel(kline, valuation)
    del kline, valuation
    elig = eligible(p, basic)
    ranks = {}
    for k, v in iter_alt_factors(p, forecast):
        if k in weights:
            ranks[k] = model.xs_rank(v, elig)
        del v
    live = model.score(ranks, weights, elig)
    del ranks
    emotion = emotion_regime(p, elig)
    always = pd.Series(True, index=p.dates)
    ohl = adjusted_ohlc(p)
    sigs = signals(p, elig, live)
    regimes, mkt = market_regimes(p, elig)
    end = str(p.dates[-1].date())
    periods = dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, end))
    curves = {}

    def evaluate(name, sel, ex, gate, cap=None, trend20=False, **extra):
        row = dict(variant=name, selection=sel, exit=ex[1], trend20=trend20, **extra)
        eq = []
        for k, (s, e) in periods.items():
            d, t, expo = simulate(p, ohl, sigs[sel], gate, N, ex, s, e, cap)
            if trend20:
                d = overlay(d, "trend20")
            row[k] = stats(d, t)
            half = stats(d * share, t)
            row[k]["half_ann"], row[k]["half_dd"] = half.get("ann_return"), half.get("max_drawdown")
            row[k]["exposure"] = float(expo.mean()) if len(expo) else None
            row[k]["signals_per_day"] = float(sigs[sel].loc[s:e].notna().sum(axis=1).mean())
            eq.append(d)
        curves[name] = pd.concat(eq)
        log.info("%s: train %.3f/%.2f test %.3f/%.2f", name, row["train"].get("ann_return", 0), row["train"].get("sharpe", 0),
                 row["test"].get("ann_return", 0), row["test"].get("sharpe", 0))
        return row

    live_exit = EXITS[0]
    base = evaluate("线上规则（对照）", "live", live_exit, emotion, trend20=True, group="base")
    # 第一轮：选股 × 卖出，不加市场阶段（线上规则保留情绪开关）
    grid = []
    for sname, sel in SELECTIONS:
        for ename, *ex in EXITS:
            ex = (ename, *ex)
            gate = emotion if sel.startswith("live") else always
            grid.append(evaluate(f"{sname}｜{ename}", sel, ex, gate, group="grid", sel_name=sname, exit_name=ename))

    def score(r):
        return ((r["train"].get("trade_win") or 0) >= GOAL_WIN, r["train"].get("sharpe", -9))

    top = sorted(grid, key=score, reverse=True)[:STAGE2_TOP]
    # 第二轮：训练期最好的几个组合（加上线上规则）× 市场阶段仓位，再看加不加净值 20 日线开关
    stage2 = []
    cand = top + ([] if any(r["selection"] == "live" and r["exit"] == "live" for r in top)
                  else [next(r for r in grid if r["selection"] == "live" and r["exit"] == "live")])
    for r0 in cand:
        ex = next((e[0], *e[1:]) for e in EXITS if e[1] == r0["exit"])
        gate = emotion if r0["selection"].startswith("live") else always
        for rname, rk in REGIMES:
            for lv in LEVELS:
                cap = regimes[rk].map({2: lv[0], 1: lv[1], 0: lv[2]}).astype(float)
                for tr in (False, True):
                    nm = (f"{r0['sel_name']}｜{r0['exit_name']}｜{rname.split('：')[0]}，"
                          f"仓位 {lv[0]:.0%}/{lv[1]:.0%}/{lv[2]:.0%}" + ("｜净值跌破 20 日线空仓" if tr else ""))
                    stage2.append(evaluate(nm, r0["selection"], ex, gate, cap, tr, group="regime", regime=rk,
                                           levels=lv, base_variant=r0["variant"]))
    rows = [base] + grid + stage2
    pool = [r for r in rows if (r["train"].get("trade_win") or 0) >= GOAL_WIN] or rows
    chosen = max(pool, key=lambda r: r["train"].get("sharpe", -9))
    best_test = max(rows, key=lambda r: r["test"].get("sharpe", -9))

    # 市场阶段判断准不准：各阶段之后 20 天等权大盘的平均涨幅
    fwd = mkt.shift(-20) / mkt - 1
    regime_check = []
    for rname, rk in REGIMES:
        for k, (s, e) in periods.items():
            g = regimes[rk].loc[s:e]
            f = fwd.loc[s:e]
            for lvl, lab in ((2, "主升浪"), (1, "震荡/埋伏"), (0, "下跌")):
                m = g == lvl
                regime_check.append(dict(regime=rk, period=k, state=lab, share=float(m.mean()),
                                         fwd20=float(f[m].mean()) if m.any() else None))
    keep = {base["variant"], chosen["variant"], best_test["variant"]} | {r["variant"] for r in top}
    keep |= {r["variant"] for r in grid if r["exit"] == "live"}
    eq = {k: ((1 + v).cumprod()).resample("W-FRI").last().round(4) for k, v in curves.items() if k in keep}
    m = mkt.resample("W-FRI").last()
    curve = dict(dates=[str(d.date()) for d in m.index],
                 market=[round(float(x), 4) for x in (m / m.iloc[0])],
                 series={k: [None if pd.isna(x) else float(x) for x in v.reindex(m.index)] for k, v in eq.items()},
                 regimes={rk: [int(x) for x in regimes[rk].resample("W-FRI").last().reindex(m.index).fillna(1)] for _, rk in REGIMES})
    return dict(updated=end, n=N, capital_share=share, rows=rows, chosen=chosen["variant"], best_test=best_test["variant"],
                top=[r["variant"] for r in top], regime_check=regime_check, curve=curve,
                chosen_meets_win=pool is not rows)


def _p(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def _line(r):
    a, t = r["train"], r["test"]
    return (f"| {r['variant']} | {_p(a.get('ann_return'))} | {a.get('sharpe', 0):.2f} | {_p(a.get('max_drawdown'))} | "
            f"{_p(a.get('trade_win'), 0)} | {_p(t.get('ann_return'))} | {t.get('sharpe', 0):.2f} | {_p(t.get('max_drawdown'))} | "
            f"{_p(t.get('trade_win'), 0)} | {t.get('payoff') or 0:.2f} | {_p(t.get('exposure'), 0)} | "
            f"{_p(t.get('half_ann'))} | {_p(t.get('half_dd'))} |")


HEAD = ["| 做法 | 训练期年化 | 训练期夏普 | 训练期回撤 | 训练期胜率 | 样本外年化 | 样本外夏普 | 样本外回撤 | 样本外胜率 | 盈亏比 | 平均仓位 | 一半资金：样本外年化 | 一半资金：样本外回撤 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]


def to_markdown(res):
    rows = {r["variant"]: r for r in res["rows"]}
    lines = ["# 主升浪选股 + 按市场阶段控仓位：回测", "",
             f"数据截至 {res['updated']}。最多 {res['n']} 只，每只一份资金；收益扣佣金、双边滑点和印花税；开盘涨停买不进，一字跌停卖不掉。"
             "规则只用 2015–2020 挑（每笔胜率 ≥ 50% 里夏普最高），2021 至今只检验。", "",
             f"训练期挑出：**{res['chosen']}**" + ("" if res["chosen_meets_win"] else "（没有做法训练期胜率到 50%，只按夏普挑）"), "",
             f"样本外夏普最高（仅供参考，不能拿来挑）：{res['best_test']}", "",
             f"“一半资金”两列按线上建议只用 {res['capital_share'] * 100:.0f}% 资金折算到总仓位。", "",
             "## 对照和训练期挑出的", ""] + HEAD
    for v in dict.fromkeys(["线上规则（对照）", res["chosen"], *res["top"], res["best_test"]]):
        lines.append(_line(rows[v]))
    lines += ["", "## 第一轮：书里的选股方法 × 卖出规则（不按市场阶段控仓位）", ""] + HEAD
    lines += [_line(r) for r in res["rows"] if r.get("group") == "grid"]
    lines += ["", "## 第二轮：加市场阶段仓位（主升浪/震荡/下跌）", ""] + HEAD
    lines += [_line(r) for r in res["rows"] if r.get("group") == "regime"]
    lines += ["", "## 市场阶段判断：各阶段占比和之后 20 天等权大盘平均涨幅", "",
              "| 办法 | 时期 | 阶段 | 天数占比 | 之后 20 天平均涨幅 |", "| --- | --- | --- | --- | --- |"]
    for c in res["regime_check"]:
        lines.append(f"| {c['regime']} | {'训练期' if c['period'] == 'train' else '样本外'} | {c['state']} | {_p(c['share'], 0)} | {_p(c['fwd20'], 2)} |")
    lines += ["", "仅供个人研究参考，不构成投资建议。"]
    return "\n".join(lines) + "\n"


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "mainwave.md").write_text(to_markdown(res))
    (config.PAGE_DIR / "mainwave.json").write_text(json.dumps(res, ensure_ascii=False, default=str))
