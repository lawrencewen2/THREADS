"""动态仓位研究（2026-11-10 用户：仓位可以适时调整，根据宏观因素、微观经济、公司基本面、技术分析等等）。

线上规则不变（业绩预期选股 + 情绪开关，最多 20 只，止盈 2 倍 ATR、止损 3 倍 ATR、最长 10 天，
组合净值跌破 20 日均线空仓，只用 40% 资金）。这里只改“每只买多少”和“整体用几成仓”：

一、每只买多少（买入那天定好，之后不改；单只 0.5–1.5 份，总仓位不超过 100%）：
- 技术面·等风险：波动大的少买、波动小的多买（ATR 越大份额越小），每笔碰到止损亏的钱差不多；
- 技术面·趋势：在 20 日和 60 日均线上方的多买，都在下方的少买；
- 基本面·估值：盈利且市盈率低（盈利收益率高）的多买，亏损股少买；
- 信号强弱：当天得分排名越靠前买得越多。
二、整体几成仓（每天开盘前定，乘在“净值跌破 20 日线空仓”之上）：
- 宏观·恐慌指数（A股 / 全球）：越恐慌越多（逆向），或者反过来越贪婪越多（顺势），两个方向都测；
- 宏观·全球避险：全球恐慌贪婪指数 ≤ 40 且比 20 天前低，连续 2 天时半仓；
- 宏观·美债利率：10 年期美债 60 天内上行越多仓位越低；
- 微观·资金面：沪市融资余额 20 天增减、两市成交额热度；
- 策略自身状态：最近 30 笔胜率高时多买、低时少买；波动率目标（年化 15%）。
- 组合：上面几类取平均。

公平比较：仓位放大缩小本身就能换收益和回撤，所以每个方案都把“总资金占比”调到训练期最大回撤和线上一样
（线上 40% 资金时的训练期回撤），再看样本外年化、回撤、夏普。规则只在 2015–2020 挑，2021 至今只检验。
另外跑 30 组“随机仓位”（每笔 0.5–1.5 份随机）当噪声参照：样本外好过它们才算有用。
所有数据都只用信号日 t 收盘（或更早）已知的，T 开盘执行；收益扣佣金、滑点、印花税，仓位变化另扣进出成本。
"""

import json
import logging

import numpy as np
import pandas as pd

from . import cn_macro, config, model
from .alt_study import emotion_regime, iter_alt_factors
from .factors import Panel, eligible
from .live_slots import MODEL_FILE
from .slots import levels, stats
from .stability import SWITCH_COST
from .winrate import adjusted_ohlc

log = logging.getLogger(__name__)

RAW = config.ROOT / "data" / "raw"
LO, HI = 0.5, 1.5
N_RANDOM = 30

PER_TRADE = [
    ("技术面·等风险（波动大的少买）", "atr"),
    ("技术面·趋势（站上 20/60 日线的多买）", "trend"),
    ("基本面·估值（盈利且市盈率低的多买，亏损股少买）", "value"),
    ("信号强弱（得分越靠前买得越多）", "score"),
]
PORTFOLIO = [
    ("宏观·A股恐慌贪婪指数，越恐慌仓位越高（逆向）", "fear_cn"),
    ("宏观·A股恐慌贪婪指数，越贪婪仓位越高（顺势）", "fear_cn_pro"),
    ("宏观·全球恐慌贪婪指数，越恐慌仓位越高（逆向）", "fear_gl"),
    ("宏观·全球恐慌贪婪指数，越贪婪仓位越高（顺势）", "fear_gl_pro"),
    ("宏观·全球避险信号时半仓", "riskoff"),
    ("宏观·美债 10 年利率 60 天上行越多仓位越低", "rates"),
    ("微观·融资余额 20 天增加多买、减少少买", "margin"),
    ("微观·两市成交额热（5 日 / 60 日）多买、冷少买", "turnover"),
    ("策略自身·最近 30 笔胜率高多买、低少买", "health"),
    ("策略自身·波动率目标年化 15%", "vol15"),
    # 2026-11-10 用户：中国宏观数据也要测（PMI、货币、社融、价格；公开免密钥接口，按公布日错开）
    ("中国宏观·制造业 PMI 越高仓位越高", "cn_pmi"),
    ("中国宏观·PMI 近 3 个月回升多买、回落少买", "cn_pmi_chg"),
    ("中国宏观·M1−M2 剪刀差越高（资金活化）仓位越高", "cn_m1m2"),
    ("中国宏观·M1−M2 剪刀差近 3 个月走阔多买", "cn_m1m2_chg"),
    ("中国宏观·M2 增速近半年加快多买", "cn_m2_chg"),
    ("中国宏观·社融 12 个月累计同比（信贷脉冲）越高仓位越高", "cn_credit"),
    ("中国宏观·PPI 同比近 3 个月回升多买", "cn_ppi_chg"),
]
COMBOS = [
    ("组合·宏观（A股逆向恐慌 + 避险 + 利率）", ("fear_cn", "riskoff", "rates")),
    ("组合·微观资金面（融资 + 成交额）", ("margin", "turnover")),
    ("组合·宏观 + 微观 + 策略状态", ("fear_cn", "riskoff", "rates", "margin", "turnover", "health")),
    ("组合·中国经济（PMI + 剪刀差 + 社融 + PPI）", ("cn_pmi", "cn_m1m2", "cn_credit", "cn_ppi_chg")),
    ("组合·中国经济变化（PMI 回升 + 剪刀差走阔 + M2 加快）", ("cn_pmi_chg", "cn_m1m2_chg", "cn_m2_chg")),
]
COMBO_TRADE = [
    ("组合·每只：等风险 + 趋势 + 估值", ("atr", "trend", "value")),
]


def _clip(x):
    return np.clip(x, LO, HI)


# ---------- 每只买多少：得到 (日期 × 股票) 的系数，第 t 行用于 T = t+1 开盘的买入 ----------

def trade_multipliers(p, elig, atr):
    ret = (p.close / p.preclose - 1).where(p.traded)
    adj = (1 + ret.fillna(0)).cumprod().where(p.close.notna()).ffill()
    out = {}
    a = pd.DataFrame(atr, index=p.dates, columns=p.codes).where(elig)
    med = a.median(axis=1)
    out["atr"] = _clip(a.rdiv(med, axis=0)).astype("float32")
    above = (adj > adj.rolling(20, min_periods=15).mean()).astype("float32") + \
        (adj > adj.rolling(60, min_periods=40).mean()).astype("float32")
    out["trend"] = (0.7 + 0.3 * above).astype("float32")  # 0.7 / 1.0 / 1.3
    pe = p.valuation.get("peTTM")
    if pe is not None:
        ep = (1 / pe).where(pe > 0)
        pct = ep.where(elig).rank(axis=1, pct=True)
        out["value"] = (LO + pct).where(pct.notna(), LO).astype("float32")  # 亏损/无数据 0.5，盈利 0.5–1.5
    else:
        out["value"] = pd.DataFrame(1.0, index=p.dates, columns=p.codes, dtype="float32")
    return {k: v.to_numpy(dtype="float32") for k, v in out.items()}


def simulate(p, ohl, sc, gate, n, tp, sl, hold, start, end, mult=None, by_rank=False, rng=None):
    """slots.simulate 加上每只的份额：份额 = 系数 / n，买入时定好；总仓位不超过 1。"""
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
    held = {}  # 列号 -> [买入价, 止盈价, 止损价, 买入日序号, 份额]
    daily, trades = [], []
    for T in range(i0, i1):
        stamp = config.STAMP_DUTY_NEW if dates[T] >= stamp_change else config.STAMP_DUTY_OLD
        sell_cost = config.SLIPPAGE + config.COMMISSION + stamp
        r = 0.0
        for j in list(held):
            entry, target, stop, d0, w = held[j]
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
            r += (px / o[T, j] - 1 - sell_cost) * w
            trades.append(dict(entry=dates[d0], exit=dates[T], code=p.codes[j], w=w * n,
                               net=px * (1 - sell_cost) / (entry * (1 + buy_cost)) - 1))
            del held[j]
        if gate_v[T - 1] and len(held) < n:
            row = sc_v[T - 1]
            ok = ~np.isnan(row)
            used = sum(v[4] for v in held.values())
            for k, j in enumerate(np.argsort(-np.where(ok, row, -np.inf))[:n * 3]):
                if len(held) >= n or not ok[j]:
                    break
                if j in held or not traded[T, j] or raw_open[T, j] >= up_v[T, j] - 0.005:
                    continue
                a = atr[T - 1, j]
                if not np.isfinite(a) or not np.isfinite(o[T, j]):
                    continue
                if rng is not None:
                    m = rng.uniform(LO, HI)
                elif by_rank:
                    m = HI - (HI - LO) * k / (n * 3 - 1)
                elif mult is not None:
                    m = float(mult[T - 1, j]) if np.isfinite(mult[T - 1, j]) else 1.0
                else:
                    m = 1.0
                w = min(m / n, 1.0 - used)
                if w <= 1e-6:
                    break
                target, stop = levels(o[T, j], a, tp, sl)
                held[j] = [o[T, j], target, stop, T, w]
                used += w
                r -= buy_cost * w
        for j, v in held.items():
            if np.isfinite(o[T + 1, j]) and o[T, j] > 0:
                r += (o[T + 1, j] / o[T, j] - 1) * v[4]
        daily.append((dates[T], r))
    return pd.Series(dict(daily), dtype="float64"), trades


# ---------- 整体几成仓：日期序列，第 t 行是 t 收盘后可知、用于下一个交易日开盘 ----------

def _raw(name):
    try:
        s = pd.read_csv(RAW / f"{name}.csv", parse_dates=["date"]).set_index("date")["value"]
        return s[~s.index.duplicated(keep="last")].sort_index()
    except FileNotFoundError:
        return None


def _align(s, dates, lag=0):
    """按日期对齐到交易日（只往前填，最多 7 天），lag 额外错开几个交易日。"""
    if s is None:
        return pd.Series(np.nan, index=dates)
    out = s.reindex(s.index.union(dates)).ffill(limit=7).reindex(dates)
    return out.shift(lag)


def market_multipliers(dates):
    out = {}
    try:
        idx = pd.read_csv(config.FEAR_INDEX_FILE, parse_dates=["date"]).set_index("date")
    except FileNotFoundError:
        idx = pd.DataFrame(index=dates, columns=["cn", "global"], dtype=float)
    cn, gl = _align(idx["cn"], dates), _align(idx["global"], dates)
    out["fear_cn"] = _clip(1.5 - cn / 100)
    out["fear_cn_pro"] = _clip(0.5 + cn / 100)
    out["fear_gl"] = _clip(1.5 - gl / 100)
    out["fear_gl_pro"] = _clip(0.5 + gl / 100)
    g_all = idx["global"].dropna()
    off = (g_all <= 40) & (g_all < g_all.shift(20))
    off = off & off.shift(1, fill_value=False)
    out["riskoff"] = _align(off.astype(float), dates).map({1.0: 0.5, 0.0: 1.0})
    y = _raw("us10y")
    chg = (y - y.shift(60)) if y is not None else None
    out["rates"] = 1 - 0.5 * np.clip(_align(chg, dates), 0, 1)
    mg = _raw("sh_margin")
    g = (mg / mg.shift(20) - 1) if mg is not None else None
    out["margin"] = _clip(1 + 10 * _align(g, dates, lag=1))  # +5% → 1.5，−5% → 0.5；融资数据晚一天公布，再错开一天
    tv = _raw("a_turnover")
    heat = (tv.rolling(5).mean() / tv.rolling(60).mean()) if tv is not None else None
    out["turnover"] = _clip(1 + (_align(heat, dates) - 1))
    out = {k: v.fillna(1.0).astype(float) for k, v in out.items()}
    out.update(cn_macro.multipliers(dates))
    return out


def health_multiplier(trades, dates, k=30):
    """最近 k 笔已卖出交易的胜率：T 开盘前已知的只算卖出日早于 T 的。0.5 → 1.0，0.6 → 1.5，0.4 → 0.5。"""
    if not trades:
        return pd.Series(1.0, index=dates)
    t = pd.DataFrame(trades).sort_values("exit")
    wr = (t["net"] > 0).astype(float).rolling(k, min_periods=10).mean()
    s = pd.Series(wr.to_numpy(), index=t["exit"]).groupby(level=0).last()
    s = s.reindex(s.index.union(dates)).ffill().reindex(dates)
    return _clip(1 + 5 * (s - 0.5)).fillna(1.0)  # 第 t 行含 t 当天开盘卖出的，T = t+1 用


def apply(daily, m=None):
    """线上的“净值跌破 20 日线空仓”（错开两行，同 stability.overlay）再乘整体系数 m（第 t 行用于 t+1，错开一行）。"""
    eq = (1 + daily).cumprod()
    pos = (eq >= eq.rolling(20, min_periods=10).mean()).astype(float).shift(2).fillna(1.0)
    if m is not None:
        pos = pos * m.reindex(daily.index).shift(1).fillna(1.0)
    return daily * pos - pos.diff().abs().fillna(0) * SWITCH_COST / 2


def vol_target(daily, target=0.15):
    vol = daily.rolling(20, min_periods=10).std() * np.sqrt(244)
    return _clip(target / vol).fillna(1.0)


def matched_share(train_daily, target_dd):
    """把总资金占比调到训练期最大回撤 = target_dd（二分）。"""
    def dd(c):
        eq = (1 + train_daily * c).cumprod()
        return float((eq / eq.cummax() - 1).min())
    lo, hi = 0.01, 3.0
    if dd(hi) > target_dd:
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2
        if dd(mid) < target_dd:
            hi = mid
        else:
            lo = mid
    return lo


def summarize(name, kind, group, d_train, d_test, t_train, t_test, share, target_dd):
    c = matched_share(d_train, target_dd)
    row = dict(name=name, kind=kind, group=group, share=float(c))
    for k, d, t in (("train", d_train, t_train), ("test", d_test, t_test)):
        st = stats(d * c, t)
        st0 = stats(d * share, t)
        row[k] = {kk: st.get(kk) for kk in ("ann_return", "sharpe", "max_drawdown", "trade_win", "trades", "payoff", "yearly")}
        row[k]["live_share_ann"] = st0.get("ann_return")
        row[k]["live_share_dd"] = st0.get("max_drawdown")
        st1 = stats(d, t)
        row[k]["full_ann"], row[k]["full_dd"], row[k]["full_sharpe"] = (st1.get("ann_return"), st1.get("max_drawdown"),
                                                                        st1.get("sharpe"))
        avg_w = np.mean([x.get("w", 1.0) for x in t]) if t else None
        row[k]["avg_w"] = float(avg_w) if avg_w is not None else None
    return row


def run(kline, basic, valuation=None, forecast=None):
    kline = kline() if callable(kline) else kline
    valuation = valuation() if callable(valuation) else valuation
    forecast = forecast() if callable(forecast) else forecast
    sm = json.loads(MODEL_FILE.read_text())
    weights = {k: v for k, v in sm["weights"].items() if v}
    tp, sl, hold, n, share = sm["take_profit"], sm["stop_loss"], sm["max_hold"], sm["n"], sm.get("capital_share", 0.4)
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
    tm = trade_multipliers(p, elig, ohl[3])
    mm = market_multipliers(p.dates)

    def sim(**kw):
        return {k: simulate(p, ohl, sc, gate, n, tp, sl, hold, s, e, **kw) for k, (s, e) in periods.items()}

    base = sim()
    bt = {k: apply(d) for k, (d, _) in base.items()}
    target_dd = float(stats(bt["train"] * share, []).get("max_drawdown"))
    rows = [summarize("线上规则（对照，每只一样多、整体不调）", "live", "live", bt["train"], bt["test"],
                      base["train"][1], base["test"][1], share, target_dd)]
    curves = {"live": bt["test"] * rows[0]["share"]}
    log.info("live: share %.2f train dd %.3f", rows[0]["share"], target_dd)

    def add(name, kind, group, res, m=None, msrc=None):
        d = {}
        for k, (dd, tt) in res.items():
            mk = m if msrc is None else msrc(dd, tt)
            d[k] = apply(dd, mk)
        row = summarize(name, kind, group, d["train"], d["test"], res["train"][1], res["test"][1], share, target_dd)
        rows.append(row)
        curves[kind] = d["test"] * row["share"]
        log.info("%s: share %.2f train ann %.3f test ann %.3f dd %.3f", kind, row["share"], row["train"]["ann_return"] or 0,
                 row["test"]["ann_return"] or 0, row["test"]["max_drawdown"] or 0)

    for name, kind in PER_TRADE:
        add(name, kind, "trade", sim(by_rank=True) if kind == "score" else sim(mult=tm[kind]))
    for name, keys in COMBO_TRADE:
        m = np.mean([tm[k] for k in keys], axis=0)
        add(name, "+".join(keys), "trade", sim(mult=m))
    for name, kind in PORTFOLIO:
        if kind == "health":
            add(name, kind, "market", base, msrc=lambda dd, tt: health_multiplier(tt, dd.index))
        elif kind == "vol15":
            add(name, kind, "market", base, msrc=lambda dd, tt: vol_target(dd).shift(1).fillna(1.0))
        elif kind in mm:
            add(name, kind, "market", base, m=mm[kind])
    for name, keys in COMBOS:
        if any(k not in mm and k != "health" for k in keys):
            continue

        def src(dd, tt, keys=keys):
            parts = [health_multiplier(tt, dd.index) if k == "health" else mm[k].reindex(dd.index) for k in keys]
            return sum(parts) / len(parts)
        add(name, "+".join(keys), "combo", base, msrc=src)
    # 训练期最好的“每只”和“整体”各一个，叠在一起
    best_t = max((r for r in rows if r["group"] == "trade"), key=lambda r: r["train"]["ann_return"] or -9)
    best_m = max((r for r in rows if r["group"] in ("market", "combo")), key=lambda r: r["train"]["ann_return"] or -9)
    bt_kind = best_t["kind"]
    res_t = sim(by_rank=True) if bt_kind == "score" else sim(mult=np.mean([tm[k] for k in bt_kind.split("+")], axis=0))
    mk = best_m["kind"]

    def src2(dd, tt):
        parts = [health_multiplier(tt, dd.index) if k == "health" else
                 (vol_target(dd).shift(1).fillna(1.0) if k == "vol15" else mm[k].reindex(dd.index)) for k in mk.split("+")]
        return sum(parts) / len(parts)
    add(f"训练期最好的两个叠加：{best_t['name']} ＋ {best_m['name']}", "best_both", "both", res_t, msrc=src2)
    # 噪声参照：随机份额
    rand = []
    for seed in range(N_RANDOM):
        rng = np.random.default_rng(seed)
        res = {k: simulate(p, ohl, sc, gate, n, tp, sl, hold, s, e, rng=rng) for k, (s, e) in periods.items()}
        d = {k: apply(dd) for k, (dd, _) in res.items()}
        r = summarize(f"随机 {seed}", "random", "random", d["train"], d["test"], res["train"][1], res["test"][1], share, target_dd)
        rand.append(dict(train=r["train"]["ann_return"], test=r["test"]["ann_return"], test_dd=r["test"]["max_drawdown"],
                         test_sharpe=r["test"]["sharpe"]))
    tests = np.array([x["test"] for x in rand])
    noise = dict(n=N_RANDOM, test_p10=float(np.quantile(tests, 0.1)), test_p50=float(np.median(tests)),
                 test_p90=float(np.quantile(tests, 0.9)),
                 train_p90=float(np.quantile([x["train"] for x in rand], 0.9)))
    cands = [r for r in rows if r["group"] != "live"]
    chosen = max(cands, key=lambda r: r["train"]["ann_return"] or -9)
    live = rows[0]
    verdict = bool(chosen["train"]["ann_return"] > noise["train_p90"]
                   and chosen["test"]["ann_return"] > max(live["test"]["ann_return"], noise["test_p90"])
                   and chosen["test"]["max_drawdown"] >= live["test"]["max_drawdown"] - 0.01)
    eq = pd.DataFrame({k: (1 + v).cumprod() for k, v in curves.items()})
    wk = eq.resample("W-FRI").last()
    curve = dict(dates=[str(d.date()) for d in wk.index], series={k: [round(float(x), 4) for x in wk[k].ffill()] for k in wk})
    m_now = {k: float(v.iloc[-1]) for k, v in mm.items()}
    return dict(updated=end, n=n, share=share, target_dd=target_dd, rows=rows, noise=noise, random=rand,
                chosen=chosen["kind"], beats_live=verdict, curve=curve, market_now=m_now)


def _p(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def to_markdown(res):
    live = res["rows"][0]
    nz = res["noise"]
    ch = next(r for r in res["rows"] if r["kind"] == res["chosen"])
    lines = ["# 动态仓位回测：宏观 / 微观资金面 / 基本面 / 技术面", "",
             f"数据截至 {res['updated']}。选股和卖出规则与线上相同，只改每只买多少、整体用几成仓。"
             f"为了公平，每个方案都把总资金占比调到训练期（2015–2020）最大回撤 = 线上 {res['share'] * 100:.0f}% 资金时的 "
             f"{_p(res['target_dd'])}，再看 2021 年至今（样本外）。收益扣佣金、滑点、印花税和调仓成本。", "",
             f"训练期最好的方案：**{ch['name']}**，样本外年化 {_p(ch['test']['ann_return'])}、回撤 {_p(ch['test']['max_drawdown'])}；"
             f"线上 {_p(live['test']['ann_return'])}、回撤 {_p(live['test']['max_drawdown'])}。"
             f"随机仓位 {nz['n']} 组的样本外年化 10%–90% 区间 {_p(nz['test_p10'])} ~ {_p(nz['test_p90'])}。", "",
             f"线上规则用 100% 资金：训练期年化 {_p(live['train']['full_ann'])}、回撤 {_p(live['train']['full_dd'])}；"
             f"样本外年化 {_p(live['test']['full_ann'])}、回撤 {_p(live['test']['full_dd'])}、夏普 {live['test']['full_sharpe'] or 0:.2f}。", "",
             "**结论：" + ("这个方案样本外跑赢线上规则和随机仓位，可以考虑上线。" if res["beats_live"]
                         else "没有方案在样本外稳定跑赢线上规则（或没跑出随机仓位的区间），建议线上规则不变。") + "**", "",
             "| 方案 | 资金占比（同回撤） | 训练期年化 | 训练期夏普 | 样本外年化 | 样本外夏普 | 样本外回撤 | 样本外胜率 | 盈亏比 | 40% 资金：样本外年化 | 40% 资金：样本外回撤 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in res["rows"]:
        a, t = r["train"], r["test"]
        lines.append(f"| {r['name']} | {_p(r['share'], 0)} | {_p(a['ann_return'])} | {a['sharpe'] or 0:.2f} | {_p(t['ann_return'])} | "
                     f"{t['sharpe'] or 0:.2f} | {_p(t['max_drawdown'])} | {_p(t['trade_win'], 0)} | {t['payoff'] or 0:.2f} | "
                     f"{_p(t['live_share_ann'])} | {_p(t['live_share_dd'])} |")
    years = sorted(live["test"].get("yearly") or {})
    lines += ["", "## 样本外分年收益（同回撤资金占比）", "", "| 方案 | " + " | ".join(years) + " |", "| --- |" + " --- |" * len(years)]
    for r in res["rows"]:
        y = r["test"].get("yearly") or {}
        lines.append(f"| {r['name']} | " + " | ".join(_p(y.get(k)) for k in years) + " |")
    lines += ["", "## 今天各个整体仓位系数（1 = 不调）", "", "| 代号 | 系数 |", "| --- | --- |"]
    lines += [f"| {k} | {v:.2f} |" for k, v in res["market_now"].items()]
    lines += ["", "仅供个人研究参考，不构成投资建议。"]
    return "\n".join(lines) + "\n"


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "sizing.md").write_text(to_markdown(res))
    (config.PAGE_DIR / "sizing.json").write_text(json.dumps(res, ensure_ascii=False, default=str))
