"""米内尔维尼 VCP 加进线上选股（2026-11-12 用户：把 Minervini 的 VCP 策略加入规则中）。

2026-11-01 的主升浪研究里，VCP 单独选股样本外亏得很厉害（−19%/年）。这次不替换线上规则，而是把 VCP 叠在
线上规则（业绩预期 + 业绩加速 0.3 + 情绪开关，20 只，止盈 2 倍 ATR、止损 3 倍 ATR、最长 10 天，
净值跌破 20 日均线空仓）上面，三种用法分别测：
- 过滤：只买满足趋势模板 / 处在收缩里的线上候选；
- 加分：得分 = 线上得分 + w × VCP 相关排名（相对强度、收缩紧、缩量、靠近枢轴、VCP 综合分），w = 0.15、0.30；
- 入场触发：线上候选里，只在 VCP 形态放量突破那天买（次日开盘），或者按米内尔维尼的做法在枢轴价挂突破买单
  （当天最高价碰到枢轴价就按枢轴价/开盘价买，高开超过枢轴 5% 不追）。
VCP 的定义事先写死（米内尔维尼《股票魔法师》的趋势模板 8 条 + 波动逐级收缩 + 缩量 + 枢轴突破），不在样本外调。
训练期 2015–2020 挑，2021 至今检验；100% 资金口径，扣成本，叠加净值 20 日均线空仓。
随机参照：8 个随机因子加分，和随机保留 30%/60% 的过滤，看 VCP 是否好过“随便筛一下”。
"""

import json
import logging

import numpy as np
import pandas as pd

from . import config, model
from .alt_study import emotion_regime, iter_alt_factors
from .factor_hunt import _summ
from .factors import Panel, eligible
from .live_slots import MODEL_FILE, apply_blend
from .slots import levels, simulate
from .winrate import adjusted_ohlc

log = logging.getLogger(__name__)

WEIGHTS = (0.15, 0.30)
N_RANDOM = 8
CHASE = 0.05  # 高开超过枢轴 5% 不追（米内尔维尼）


def _rank(f, mask):
    return model.xs_rank(f, mask)  # −0.5..0.5


def features(p, elig):
    """VCP 相关的宽表，都只用当天收盘及以前的数据（信号日 t 收盘后可知）。"""
    ret = (p.close / p.preclose - 1).where(p.traded)
    adj = (1 + ret.fillna(0)).cumprod().astype("float32")
    k = adj / p.close
    hi, lo = (p.high * k).astype("float32"), (p.low * k).astype("float32")
    turn = p.turn.where(p.turn > 0)

    def ma(w):
        return adj.rolling(w, min_periods=int(w * 0.8)).mean().astype("float32")

    ma50, ma150, ma200 = ma(50), ma(150), ma(200)
    hi250 = hi.rolling(250, min_periods=200).max()
    lo250 = lo.rolling(250, min_periods=200).min()
    r = sum(wt * (adj / adj.shift(d) - 1) for wt, d in ((0.4, 63), (0.2, 126), (0.2, 189), (0.2, 252)))
    rs = _rank(r.where(adj.shift(252).notna()), elig) + 0.5  # IBD 式相对强度 0..1
    del r
    f = {}
    f["tt"] = ((adj > ma150) & (adj > ma200) & (ma150 > ma200) & (ma200 > ma200.shift(20)) & (ma50 > ma150)
               & (adj > ma50) & (adj >= 1.3 * lo250) & (adj >= 0.75 * hi250) & (rs >= 0.7))
    f["tt_lite"] = (adj > ma50) & (ma50 > ma150) & (ma150 > ma200) & (ma200 > ma200.shift(20))
    f["stage4"] = (adj < ma200) & (ma200 < ma200.shift(20))
    f["above50"] = adj > ma50
    del ma50, ma150, lo250

    def rng(a, b):  # 第 a..b 个交易日前（含 a，不含 b）的高低点振幅
        h = hi.shift(a).rolling(b - a, min_periods=int((b - a) * 0.8)).max()
        l_ = lo.shift(a).rolling(b - a, min_periods=int((b - a) * 0.8)).min()
        return h / l_ - 1

    r1, r2, r3 = rng(30, 45), rng(15, 30), rng(0, 15)
    f["seq"] = (r3 < r2) & (r2 < r1) & (r3 < 0.15)  # 三段振幅逐级收缩，最后一段 < 15%
    rng10, rng50 = rng(0, 10), rng(0, 50)
    t50 = turn.rolling(50, min_periods=40).mean()
    dry = turn.rolling(10, min_periods=8).mean() / t50
    f["contract"] = (rng10 < 0.5 * rng50) & (dry < 0.9)
    pivot = hi.rolling(20, min_periods=15).max()  # 枢轴：近 20 日最高价（含当天）
    near = adj / pivot
    f["breakout"] = (adj > hi.shift(1).rolling(20, min_periods=15).max()) & (turn / t50 >= 1.5)
    f["vcp_break"] = f["tt"] & (f["contract"] | f["seq"]).shift(1, fill_value=False) & f["breakout"]
    f["vcp_setup"] = f["tt"] & (f["contract"] | f["seq"]) & (near >= 0.9)
    f["seq_setup"] = f["tt"] & f["seq"] & (near >= 0.9)
    f["r_rs"] = rs - 0.5
    f["r_tight"] = -_rank(rng10 / rng50, elig)
    f["r_dry"] = -_rank(dry, elig)
    f["r_near"] = _rank(near, elig)
    comp = (f["r_rs"] + f["r_tight"] + f["r_dry"] + f["r_near"]) / 4 + 0.5 * f["tt"].astype("float32")
    f["r_vcp"] = _rank(comp, elig)
    f["r_tt"] = f["tt"].astype("float32") - 0.5
    del r1, r2, r3, rng10, rng50, dry, comp, near
    f["pivot"] = (pivot / k).astype("float32")  # 换回实际价格（近 20 日内有除权时略有偏差）
    return f


def simulate_stop(p, ohl, sc, setup, pivot_px, gate, n, tp, sl, hold, start, end):
    """枢轴突破买单：信号日 t 收盘后，在得分前（空位数）只、且处在 VCP 收缩里的线上候选上挂枢轴价买单；
    T 日最高价 ≥ 枢轴价就成交，价格 = max(开盘, 枢轴)；开盘已高于枢轴 5% 不追，涨停价 ≤ 枢轴不买。
    卖出规则和 slots.simulate 完全一样。pivot_px 是实际价格（不复权）。"""
    o, h, l, atr, locked = ohl
    up, _ = p.limit_prices()
    up_v, raw_open, traded = up.to_numpy(), p.open.to_numpy(), p.traded.to_numpy()
    raw_high = p.high.to_numpy()
    kk = o / raw_open  # 复权系数（当天）
    sc_v = sc.where(setup).to_numpy()
    piv = pivot_px.to_numpy()
    gate_v = gate.reindex(p.dates).fillna(False).to_numpy(dtype=bool)
    dates = p.dates
    i0 = max(dates.searchsorted(pd.Timestamp(start)), 15)
    i1 = min(dates.searchsorted(pd.Timestamp(end), side="right"), len(dates) - 1)
    buy_cost = config.SLIPPAGE + config.COMMISSION
    stamp_change = pd.Timestamp(config.STAMP_DUTY_CHANGE)
    held = {}
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
            r += (px / o[T, j] - 1 - sell_cost) / n
            trades.append(dict(entry=dates[d0], exit=dates[T], code=p.codes[j],
                               net=px * (1 - sell_cost) / (entry * (1 + buy_cost)) - 1))
            del held[j]
        new = {}
        free = n - len(held)
        if gate_v[T - 1] and free > 0:
            row = sc_v[T - 1]
            ok = ~np.isnan(row)
            orders = [j for j in np.argsort(-np.where(ok, row, -np.inf))[:free * 3]
                      if ok[j] and j not in held][:free]  # 开盘前挂好 free 张买单
            for j in orders:
                pv = piv[T - 1, j]
                if not traded[T, j] or not np.isfinite(pv) or not np.isfinite(o[T, j]):
                    continue
                if raw_open[T, j] > pv * (1 + CHASE) or up_v[T, j] <= pv + 0.005 or raw_high[T, j] < pv:
                    continue
                a = atr[T - 1, j]
                if not np.isfinite(a):
                    continue
                e = max(raw_open[T, j], pv) * kk[T, j]  # 复权口径成交价
                target, stop = levels(e, a, tp, sl)
                held[j] = [e, target, stop, T]
                new[j] = e
                r -= buy_cost / n
        for j in held:
            base = new.get(j, o[T, j])
            if np.isfinite(o[T + 1, j]) and base > 0:
                r += (o[T + 1, j] / base - 1) / n
        daily.append((dates[T], r))
    return pd.Series(dict(daily), dtype="float64"), trades


def run(kline, basic, valuation=None, forecast=None, dump=None):
    kline = kline() if callable(kline) else kline
    valuation = valuation() if callable(valuation) else valuation
    forecast = forecast() if callable(forecast) else forecast
    sm = json.loads(MODEL_FILE.read_text())
    weights = {k: v for k, v in sm["weights"].items() if v}
    tp, sl, hold, n = sm["take_profit"], sm["stop_loss"], sm["max_hold"], sm["n"]
    p = Panel(kline, valuation)
    del kline, valuation
    elig = eligible(p, basic)
    ranks = {}
    for k, v in iter_alt_factors(p, forecast):
        if k in weights:
            ranks[k] = model.xs_rank(v, elig)
        del v
    base = apply_blend(model.score(ranks, weights, elig), p, forecast, elig, sm.get("blend"))
    del ranks
    f = features(p, elig)
    gate = emotion_regime(p, elig)
    ohl = adjusted_ohlc(p)
    end = str(p.dates[-1].date())
    periods = dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, end))
    cols = {}

    def record(name, d, t, period):
        held = pd.Series(0, index=d.index)
        for x in t:
            held[(held.index >= x["entry"]) & (held.index < x["exit"])] += 1
        cols.setdefault(name, []).append(pd.DataFrame({name + "|raw": d, name + "|held": held}))

    def sim(name, sc, stop_setup=None):
        res = {}
        for k, (s, e) in periods.items():
            if stop_setup is None:
                d, t = simulate(p, ohl, sc, gate, n, tp, sl, hold, s, e)
            else:
                d, t = simulate_stop(p, ohl, sc, stop_setup, f["pivot"], gate, n, tp, sl, hold, s, e)
            res[k] = _summ(d, t)
            cand = sc if stop_setup is None else sc.where(stop_setup)
            res[k]["coverage"] = float(cand.loc[s:e].notna().sum(axis=1).mean())
            if dump is not None:
                record(name, d, t, k)
        return res

    rows = []

    def add(name, group, sc, key, stop_setup=None, **extra):
        rows.append(dict(name=name, group=group, key=key, **extra, **sim(key, sc, stop_setup)))
        r = rows[-1]
        log.info("%s: train %.2f/%s test %.2f/%s cov %.0f", key, r["train"]["sharpe"] or 0, r["train"]["ann_return"],
                 r["test"]["sharpe"] or 0, r["test"]["ann_return"], r["test"]["coverage"])

    add("线上规则（对照）", "live", base, "live")
    # 过滤
    add("只买趋势模板 8 条都满足的", "filter", base.where(f["tt"]), "f_tt")
    add("只买均线多头排列（50>150>200 且 200 日线向上）", "filter", base.where(f["tt_lite"]), "f_ttlite")
    add("只买股价在 50 日线上方", "filter", base.where(f["above50"]), "f_above50")
    add("不买第四阶段（股价在下行的 200 日线下）", "filter", base.where(~f["stage4"]), "f_not_stage4")
    add("只买趋势模板 + 正在收缩", "filter", base.where(f["tt"] & (f["contract"] | f["seq"])), "f_tt_contract")
    # 加分
    for key, desc in (("r_rs", "相对强度"), ("r_tight", "收缩紧（10 日振幅/50 日振幅小）"), ("r_dry", "缩量"),
                      ("r_near", "靠近枢轴（离 20 日高点近）"), ("r_vcp", "VCP 综合分"), ("r_tt", "满足趋势模板")):
        for w in WEIGHTS:
            add(f"{desc} 加分 {w}", "blend", (base + w * f[key].fillna(0)).where(base.notna()), f"{key}_{w}", w=w)
    # 入场触发（次日开盘买）
    add("只在放量突破 20 日高点那天买", "entry", base.where(f["breakout"]), "e_break")
    add("只在 VCP 放量突破那天买（趋势模板 + 收缩后突破）", "entry", base.where(f["vcp_break"]), "e_vcp_break")
    add("只在 VCP 收缩、离枢轴 10% 以内时买（突破前埋伏）", "entry", base.where(f["vcp_setup"]), "e_vcp_setup")
    # 入场触发（枢轴价挂突破买单，米内尔维尼原版）
    add("VCP 收缩中挂枢轴突破买单", "entry_stop", base, "s_vcp", stop_setup=f["vcp_setup"])
    add("三段逐级收缩中挂枢轴突破买单", "entry_stop", base, "s_seq", stop_setup=f["seq_setup"])
    add("趋势模板内挂枢轴突破买单（不要求收缩）", "entry_stop", base, "s_tt", stop_setup=f["tt"])
    add("所有线上候选挂 20 日高点突破买单", "entry_stop", base, "s_all", stop_setup=base.notna())
    # 随机参照
    rng = np.random.default_rng(11)
    for i in range(N_RANDOM):
        noise = pd.DataFrame(rng.random(base.shape, dtype=np.float32) - 0.5, index=base.index, columns=base.columns)
        for w in WEIGHTS:
            add(f"随机因子 {i + 1} 加分 {w}", "random", (base + w * noise).where(base.notna()), f"rnd{i + 1}_{w}", w=w)
        for keep in (0.3, 0.6):
            add(f"随机过滤 {i + 1} 保留 {int(keep * 100)}%", "random_filter", base.where(noise + 0.5 < keep),
                f"rndf{i + 1}_{keep}", keep=keep)
        del noise
    share = {k: float(v.where(elig).sum().sum() / max(elig.sum().sum(), 1)) for k, v in f.items()
             if v.dtypes.iloc[0] == bool}
    if dump is not None:
        out = pd.concat([pd.concat(v) for v in cols.values()], axis=1)
        out.index.name = "date"
        out.to_csv(dump, float_format="%.6g")
    return dict(updated=end, rows=rows, share=share)


def _p(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


GROUPS = {"live": "对照", "filter": "过滤", "blend": "加分", "entry": "入场触发（次日开盘买）",
          "entry_stop": "入场触发（枢轴突破买单）"}


def to_markdown(res):
    rows = res["rows"]
    live = rows[0]
    rnd = np.array([r["train"]["sharpe"] or 0 for r in rows if r["group"] == "random"])
    rndf = np.array([r["train"]["sharpe"] or 0 for r in rows if r["group"] == "random_filter"])
    L = [f"# VCP 叠加线上选股（数据到 {res['updated']}）", "",
         "线上规则（业绩预期 + 业绩加速 0.3 + 情绪开关，20 只，止盈止损，净值均线空仓）上叠加米内尔维尼 VCP。"
         "训练期 2015–2020，样本外 2021 至今，100% 资金口径，已扣成本。", "",
         f"随机加分训练夏普 90 分位 {np.quantile(rnd, 0.9):.2f}；随机过滤 90 分位 {np.quantile(rndf, 0.9):.2f}；"
         f"线上 {live['train']['sharpe']:.2f}。", "",
         "| 用法 | 方案 | 训练年化 | 训练回撤 | 训练夏普 | 训练胜率 | 样本外年化 | 样本外回撤 | 样本外夏普 | 样本外胜率 | 样本外笔数 | 候选数/天 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["group"].startswith("random"):
            continue
        a, b = r["train"], r["test"]
        L.append(f"| {GROUPS[r['group']]} | {r['name']} | {_p(a['ann_return'])} | {_p(a['max_drawdown'])} | "
                 f"{a['sharpe'] or 0:.2f} | {_p(a['trade_win'])} | {_p(b['ann_return'])} | {_p(b['max_drawdown'])} | "
                 f"{b['sharpe'] or 0:.2f} | {_p(b['trade_win'])} | {b['trades']} | {b['coverage']:.0f} |")
    L += ["", "可买股票里满足各条件的比例：" + "，".join(f"{k} {v * 100:.1f}%" for k, v in res["share"].items()), "",
          "随机参照：", ""]
    for r in rows:
        if r["group"].startswith("random"):
            L.append(f"- {r['name']}：训练夏普 {r['train']['sharpe'] or 0:.2f}，样本外夏普 {r['test']['sharpe'] or 0:.2f}，"
                     f"样本外年化 {_p(r['test']['ann_return'])}")
    return "\n".join(L) + "\n"


def write(res, stem="vcp_study"):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / f"{stem}.md").write_text(to_markdown(res))
    (config.PAGE_DIR / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
