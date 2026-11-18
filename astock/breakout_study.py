"""平台突破叠加线上选股（2026-11-18 用户发来两张 K 线截图：把这种平台突破的模式加入挑股的规则）。

截图里的形态：股价先涨一段，然后横着整理一两个月（平台高点 130.51，中间回踩到 99.50，回撤约 24%），
20 日线、60 日线一直向上托着，最后放量收到平台高点上方。写成事先定死的规则（不在样本外调）：
- 平台：前 W 个交易日（不含当天）的最高价 H、最低价 L，振幅 H/L−1 不超过上限（W=20 → 25%，W=40/60 → 35%）；
  H 不是最近 K 天才创出来的（W=20 → 5 天，W=40/60 → 10 天），也就是确实横了一段；
- 趋势：20 日线在 60 日线上方，60 日线比 10 天前高，收盘在 60 日线上方；
- 突破：当天收盘价高于 H（放量版：当天换手 ≥ 50 日平均的 1.5 倍）；
- 严格版另加：平台之前确实涨过（H 比平台前 60 天的最低价高 25% 以上）。
主口径 W=40（约两个月，和截图接近），20/60 做敏感性。

和 2026-11-13 的 VCP 研究一样，不替换线上规则（业绩预期 + 情绪开关，20 只，止盈 2 倍 ATR、止损 3 倍 ATR、
最长 10 天，净值跌破 20 日均线空仓），而是叠在上面，四种用法：
- 过滤：只买处在平台里 / 贴近平台上沿 / 刚突破 5 天内的线上候选；
- 加分：线上得分 + w × 形态排名（贴近平台上沿、刚突破、均线向上），w = 0.15、0.30；
- 入场触发：线上候选只在突破当天收盘后入选、次日开盘买；或在平台高点挂突破买单（盘中碰到就买）；
- 纯形态：不看业绩预期，全市场放量突破的股票按放量倍数排。
回测用 2026-11-16 修正后的口径（盘中卖出腾的仓位第二天才能买）。训练 2015–2020 挑，2021 至今检验。
同时给出线上总组合口径（选股 35% + 纳指/黄金/国债 ETF 分仓、闲钱停泊）。
随机参照：8 个随机因子加分、随机保留 30%/60% 的过滤。
"""

import json
import logging

import numpy as np
import pandas as pd

from . import config, diversify as dv, model, redline
from .alt_study import emotion_regime, iter_alt_factors
from .factor_hunt import _summ
from .factors import Panel, eligible
from .live_slots import MODEL_FILE, apply_blend
from .slots import levels, simulate
from .winrate import adjusted_ohlc

log = logging.getLogger(__name__)

WEIGHTS = (0.15, 0.30)
N_RANDOM = 8
CHASE = 0.05  # 开盘已高于平台高点 5% 不追
WINDOWS = {20: (0.25, 5), 40: (0.35, 10), 60: (0.35, 10)}  # W: (振幅上限, 高点至少是几天前的)
VOL_MULT = 1.5
CONS_SLIPPAGE = 0.002


def features(p, elig):
    """平台突破相关的宽表，都只用当天收盘及以前的数据（信号日 t 收盘后可知）。"""
    ret = (p.close / p.preclose - 1).where(p.traded)
    adj = (1 + ret.fillna(0)).cumprod().astype("float32")
    k = adj / p.close
    hi, lo = (p.high * k).astype("float32"), (p.low * k).astype("float32")
    turn = p.turn.where(p.turn > 0)
    ma20 = adj.rolling(20, min_periods=16).mean()
    ma60 = adj.rolling(60, min_periods=48).mean()
    trend = (ma20 > ma60) & (ma60 > ma60.shift(10)) & (adj > ma60)
    volx = turn / turn.rolling(50, min_periods=40).mean()
    f = {"trend": trend, "volx": volx.astype("float32")}
    for w, (depth, age) in WINDOWS.items():
        mp = int(w * 0.8)
        # 以昨天为止的平台（今天收盘去突破它）
        h_prev = hi.shift(1).rolling(w, min_periods=mp).max()
        l_prev = lo.shift(1).rolling(w, min_periods=mp).min()
        plat_prev = ((h_prev / l_prev - 1 <= depth) & (hi.shift(1).rolling(age, min_periods=age).max() < h_prev)
                     & trend.shift(1, fill_value=False))
        brk = plat_prev & (adj > h_prev)
        f[f"brk{w}"] = brk
        f[f"brk{w}_vol"] = brk & (volx >= VOL_MULT)
        # 含今天的平台（明天去突破它）：今天还没突破
        h_now = hi.rolling(w, min_periods=mp).max()
        l_now = lo.rolling(w, min_periods=mp).min()
        plat = (h_now / l_now - 1 <= depth) & (hi.rolling(age, min_periods=age).max() < h_now) & trend
        near = adj / h_now
        f[f"plat{w}"] = plat
        f[f"setup{w}"] = plat & (near >= 0.9)
        f[f"pivot{w}"] = (h_now / k).astype("float32")  # 换回实际价格（窗口内有除权时略有偏差）
        if w == 40:
            prior_low = lo.shift(w + 1).rolling(60, min_periods=48).min()
            f["brk40_strict"] = f["brk40_vol"] & (h_prev >= 1.25 * prior_low)
            f["post5"] = brk.astype("float32").rolling(5, min_periods=1).max().astype(bool) & (adj > ma20)
            f["r_near"] = model.xs_rank(near.where(plat), elig).fillna(-0.5)  # 不在平台里的排最后
            f["r_post"] = f["post5"].astype("float32") - 0.5
        del h_prev, l_prev, h_now, l_now, near
    f["r_trend"] = trend.astype("float32") - 0.5
    f["r_volbrk"] = model.xs_rank(volx.where(f["brk40_vol"]), elig)  # 纯形态：放量倍数
    return f


def simulate_stop(p, ohl, sc, setup, pivot_px, gate, n, tp, sl, hold, start, end):
    """平台高点突破买单：信号日 t 收盘后，在得分靠前、且处在平台里的线上候选上挂突破买单；
    T 日最高价 ≥ 平台高点就成交，价格 = max(开盘, 平台高点)；开盘已高于平台高点 5% 不追，涨停价 ≤ 平台高点不买。
    卖出规则和 slots.simulate（2026-11-16 修正后）一样：盘中止盈止损腾出的仓位当天不能再用，当天卖掉的不买回。
    pivot_px 是实际价格（不复权）。"""
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
        intraday_freed, sold_today = 0, set()
        for j in list(held):
            entry, target, stop, d0 = held[j]
            px, at_open = None, False
            if not locked[T, j] and np.isfinite(o[T, j]):
                if T - d0 >= hold:
                    px, at_open = o[T, j], True
                elif l[T, j] <= stop:
                    px, at_open = min(o[T, j], stop), o[T, j] <= stop
                elif h[T, j] >= target:
                    px, at_open = max(o[T, j], target), o[T, j] >= target
            if px is None:
                continue
            if not at_open:
                intraday_freed += 1
            sold_today.add(j)
            r += (px / o[T, j] - 1 - sell_cost) / n
            trades.append(dict(entry=dates[d0], exit=dates[T], code=p.codes[j],
                               net=px * (1 - sell_cost) / (entry * (1 + buy_cost)) - 1))
            del held[j]
        new = {}
        free = n - intraday_freed - len(held)
        if gate_v[T - 1] and free > 0:
            row = sc_v[T - 1]
            ok = ~np.isnan(row)
            orders = [j for j in np.argsort(-np.where(ok, row, -np.inf))[:n * 3]
                      if ok[j] and j not in held and j not in sold_today][:free]  # 开盘前挂好 free 张买单
            for j in orders:
                pv = piv[T - 1, j]
                if not traded[T, j] or not np.isfinite(pv) or not np.isfinite(o[T, j]):
                    continue
                if raw_open[T, j] > pv * (1 + CHASE) or up_v[T, j] <= pv + 0.005 or raw_high[T, j] < pv:
                    continue
                if raw_open[T, j] >= up_v[T, j] - 0.005:  # 开盘就涨停，买不到
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


def _total(frame, name, sleeve, share):
    """线上总组合口径：选股 share + ETF 分仓闲钱停泊（redline.simulate，不加红线）。"""
    out = {}
    old = redline.SHARE
    redline.SHARE = share
    try:
        parts = redline.load_part(name, frame=frame)
        for per in ("train", "test"):
            r, _, _ = redline.simulate(parts[per], sleeve, None)
            st = redline.stats(r)
            out[per] = {k: st.get(k) for k in ("ann_return", "max_drawdown", "sharpe")}
    finally:
        redline.SHARE = old
    return out


def run(kline, basic, valuation=None, forecast=None, dump=None, etf=True):
    kline = kline() if callable(kline) else kline
    valuation = valuation() if callable(valuation) else valuation
    forecast = forecast() if callable(forecast) else forecast
    sm = json.loads(MODEL_FILE.read_text())
    weights = {k: v for k, v in sm["weights"].items() if v}
    tp, sl, hold, n = sm["take_profit"], sm["stop_loss"], sm["max_hold"], sm["n"]
    share = float(sm.get("capital_share", 0.35))
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
    sleeve = None
    if etf:
        sleeve = dv.sleeve_returns(dv.etf_returns(), sm["sleeve"]["codes"], sm["sleeve"].get("rebalance", 20))
    cols = {}
    rows = []

    def sim(name, sc, stop=None):
        res, parts = {}, []
        for k, (s, e) in periods.items():
            if stop is None:
                d, t = simulate(p, ohl, sc, gate, n, tp, sl, hold, s, e)
            else:
                d, t = simulate_stop(p, ohl, sc, stop[0], stop[1], gate, n, tp, sl, hold, s, e)
            res[k] = _summ(d, t)
            cand = sc if stop is None else sc.where(stop[0])
            res[k]["coverage"] = float(cand.loc[s:e].notna().sum(axis=1).mean())
            held = pd.Series(0, index=d.index)
            for x in t:
                held[(held.index >= x["entry"]) & (held.index < x["exit"])] += 1
            res[k]["avg_held"] = float(held.mean()) if len(held) else 0.0
            parts.append(pd.DataFrame({name + "|raw": d, name + "|held": held}))
        frame = pd.concat(parts)
        if sleeve is not None:
            res["total"] = _total(frame, name, sleeve, share)
        return res, frame

    def add(name, group, sc, key, stop=None, **extra):
        res, frame = sim(key, sc, stop)
        rows.append(dict(name=name, group=group, key=key, **extra, **res))
        if dump is not None and not group.startswith("random"):
            for c in frame:
                cols[c] = frame[c]
        r = rows[-1]
        log.info("%s: train %.2f/%s test %.2f/%s cov %.0f", key, r["train"]["sharpe"] or 0, r["train"]["ann_return"],
                 r["test"]["sharpe"] or 0, r["test"]["ann_return"], r["test"]["coverage"])
        return sc, stop

    specs = {}

    def reg(name, group, sc, key, stop=None, **extra):
        specs[key] = (name, group, sc, stop)
        add(name, group, sc, key, stop, **extra)

    reg("线上规则（对照）", "live", base, "live")
    # 过滤
    reg("只买均线向上的（20 日线在 60 日线上、60 日线上行、收盘在 60 日线上）", "filter", base.where(f["trend"]), "f_trend")
    reg("只买处在 40 日平台里的", "filter", base.where(f["plat40"]), "f_plat40")
    reg("只买处在 40 日平台里、离平台高点 10% 以内的（突破前埋伏）", "filter", base.where(f["setup40"]), "f_setup40")
    reg("只买 5 天内刚突破 40 日平台、还站在 20 日线上的", "filter", base.where(f["post5"]), "f_post5")
    # 加分
    for key, desc in (("r_near", "贴近 40 日平台高点"), ("r_post", "5 天内刚突破平台"), ("r_trend", "均线向上")):
        for w in WEIGHTS:
            reg(f"{desc} 加分 {w}", "blend", (base + w * f[key].fillna(0)).where(base.notna()), f"{key}_{w}", w=w)
    # 入场触发（突破当天收盘后入选，次日开盘买）
    for w in WINDOWS:
        reg(f"只在突破 {w} 日平台那天买", "entry", base.where(f[f"brk{w}"]), f"e_brk{w}")
    reg("只在放量突破 40 日平台那天买", "entry", base.where(f["brk40_vol"]), "e_brk40_vol")
    reg("只在放量突破 40 日平台、且平台前涨过 25% 那天买（严格版）", "entry", base.where(f["brk40_strict"]),
        "e_brk40_strict")
    # 入场触发（平台高点挂突破买单）
    for w in WINDOWS:
        reg(f"{w} 日平台里的候选，在平台高点挂突破买单", "entry_stop", base, f"s_plat{w}",
            stop=(f[f"setup{w}"], f[f"pivot{w}"]))
    # 纯形态（不看业绩预期）
    reg("纯形态：全市场放量突破 40 日平台，按放量倍数排", "pure", f["r_volbrk"].where(elig), "p_volbrk")
    # 随机参照
    rng = np.random.default_rng(18)
    for i in range(N_RANDOM):
        noise = pd.DataFrame(rng.random(base.shape, dtype=np.float32) - 0.5, index=base.index, columns=base.columns)
        for w in WEIGHTS:
            add(f"随机因子 {i + 1} 加分 {w}", "random", (base + w * noise).where(base.notna()), f"rnd{i + 1}_{w}", w=w)
        for keep in (0.3, 0.6):
            add(f"随机过滤 {i + 1} 保留 {int(keep * 100)}%", "random_filter", base.where(noise + 0.5 < keep),
                f"rndf{i + 1}_{keep}", keep=keep)
        del noise
    # 保守成本（滑点 0.2%/边）：线上 + 训练期总组合夏普最好的 3 个方案（只按训练期挑）
    def train_key(r):
        return (r.get("total", r)["train"] if "total" in r else r["train"])["sharpe"] or -9

    cand = sorted((r for r in rows[1:] if not r["group"].startswith("random")), key=train_key, reverse=True)[:3]
    cons = []
    slip0 = config.SLIPPAGE
    config.SLIPPAGE = CONS_SLIPPAGE
    try:
        for key in ["live"] + [r["key"] for r in cand]:
            name, group, sc, stop = specs[key]
            res, _ = sim(key + "_cons", sc, stop)
            cons.append(dict(name=name, group=group, key=key, **res))
    finally:
        config.SLIPPAGE = slip0
    share_of = {k: float(v.where(elig).sum().sum() / max(elig.sum().sum(), 1)) for k, v in f.items()
                if v.dtypes.iloc[0] == bool}
    if dump is not None:
        out = pd.DataFrame(cols)
        out.index.name = "date"
        out.to_csv(dump, float_format="%.6g")
    return dict(updated=end, capital_share=share, rows=rows, cons=cons, share=share_of)


def _p(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


GROUPS = {"live": "对照", "filter": "过滤", "blend": "加分", "entry": "入场触发（次日开盘买）",
          "entry_stop": "入场触发（平台高点突破买单）", "pure": "纯形态"}


def _tot_cells(r):
    t = r.get("total")
    if not t:
        return "— | — | — | — | — | —"
    a, b = t["train"], t["test"]
    return (f"{_p(a['ann_return'])} | {_p(a['max_drawdown'])} | {a['sharpe'] or 0:.2f} | "
            f"{_p(b['ann_return'])} | {_p(b['max_drawdown'])} | {b['sharpe'] or 0:.2f}")


def to_markdown(res):
    rows = res["rows"]
    live = rows[0]
    rnd = np.array([r["train"]["sharpe"] or 0 for r in rows if r["group"] == "random"])
    rndf = np.array([r["train"]["sharpe"] or 0 for r in rows if r["group"] == "random_filter"])
    rnd_t = np.array([r["test"]["sharpe"] or 0 for r in rows if r["group"] == "random"])
    L = [f"# 平台突破叠加线上选股（数据到 {res['updated']}）", "",
         "线上规则（业绩预期 + 情绪开关，20 只，止盈止损，净值均线空仓）上叠加“平台突破”形态。"
         "训练期 2015–2020，样本外 2021 至今，已扣成本，回测用 2026-11-16 修正后的口径。", "",
         f"随机加分训练夏普 10–90 分位 {np.quantile(rnd, 0.1):.2f}–{np.quantile(rnd, 0.9):.2f}"
         f"（样本外 {np.quantile(rnd_t, 0.1):.2f}–{np.quantile(rnd_t, 0.9):.2f}）；"
         f"随机过滤训练夏普 90 分位 {np.quantile(rndf, 0.9):.2f}；线上 {live['train']['sharpe'] or 0:.2f}。", "",
         f"## 总组合（选股 {res['capital_share'] * 100:.0f}% + ETF 分仓闲钱停泊，线上口径）", "",
         "| 用法 | 方案 | 训练年化 | 训练回撤 | 训练夏普 | 样本外年化 | 样本外回撤 | 样本外夏普 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["group"].startswith("random"):
            L.append(f"| {GROUPS[r['group']]} | {r['name']} | {_tot_cells(r)} |")
    L += ["", "## 选股层（100% 资金口径，含 20 日均线空仓）", "",
          "| 用法 | 方案 | 训练年化 | 训练回撤 | 训练夏普 | 训练胜率 | 样本外年化 | 样本外回撤 | 样本外夏普 | 样本外胜率 | 样本外笔数 | 候选数/天 | 平均持仓只数 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["group"].startswith("random"):
            continue
        a, b = r["train"], r["test"]
        L.append(f"| {GROUPS[r['group']]} | {r['name']} | {_p(a['ann_return'])} | {_p(a['max_drawdown'])} | "
                 f"{a['sharpe'] or 0:.2f} | {_p(a['trade_win'])} | {_p(b['ann_return'])} | {_p(b['max_drawdown'])} | "
                 f"{b['sharpe'] or 0:.2f} | {_p(b['trade_win'])} | {b['trades']} | {b['coverage']:.0f} | "
                 f"{b['avg_held']:.1f} |")
    L += ["", "## 保守成本（滑点 0.2%/边）：线上 + 训练期最好的 3 个", "",
          "| 方案 | 选股层训练年化 | 选股层样本外年化 | 样本外回撤 | 总组合训练年化 | 训练回撤 | 训练夏普 | 总组合样本外年化 | 样本外回撤 | 样本外夏普 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.get("cons", []):
        L.append(f"| {r['name']} | {_p(r['train']['ann_return'])} | {_p(r['test']['ann_return'])} | "
                 f"{_p(r['test']['max_drawdown'])} | {_tot_cells(r)} |")
    L += ["", "可买股票里满足各条件的比例：" + "，".join(f"{k} {v * 100:.2f}%" for k, v in res["share"].items()), "",
          "随机参照（选股层）：", ""]
    for r in rows:
        if r["group"].startswith("random"):
            L.append(f"- {r['name']}：训练夏普 {r['train']['sharpe'] or 0:.2f}，样本外夏普 {r['test']['sharpe'] or 0:.2f}，"
                     f"样本外年化 {_p(r['test']['ann_return'])}")
    return "\n".join(L) + "\n"


def write(res, stem="breakout_study"):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / f"{stem}.md").write_text(to_markdown(res))
    (config.PAGE_DIR / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
