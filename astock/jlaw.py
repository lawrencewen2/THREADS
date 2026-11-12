"""J Law（香港交易员，2024、2025 年美国投资冠军赛百万组冠军，JLawStock 课程）的方法转成规则回测
（2026-11-12 用户：学习下香港教师 JLaw 的教育模式策略并转化）。

公开资料里他讲的要点（访谈、课程摘要、TraderLion 专访），能用日线写死的部分：
- 选股池“Stage 2”：股价在 200 日均线上方、市值和成交额够大（A股用近 20 日日均成交额 ≥ 2 亿元代替）、beta ≥ 1；
- 只做领头羊：相对强度（RS）排前 20%，RS 线在 21 日均线上方，所在行业本身也是最强的一批（行业 RS 中位数排前 30%）；
- 买点是 M.E.T.A.（多重优势进场区）：强势股缩量回调，到几种支撑叠在一起的位置才买，要 3–5 个独立优势同时出现：
  均线支撑（10/20/50 日线一带）、横向支撑（前高突破后回踩）、缩量回调、均线多头 + 低点抬高（上升趋势线）、
  之前有放量大阳线、当天止跌收阳；
- 止损放在支撑下方（近 5 日最低价下 1%），每笔最多亏总资金 0.3%（最多 0.5%）来定仓位，一次建满；
- 卖出：趁弱卖（收盘跌破 10 日或 20 日线第二天开盘卖），或趁强卖（到 3 倍风险的目标价），他自己两种都用；
- 大盘：大盘在 200 日线上方 / Stage 2 股票够多才积极做多；连续亏损时把每笔风险减半（0.15%）。

两部分：
1. 照他的方法单独做一套（上面的选股 + 买点 + 止损 + 仓位 + 卖出 + 大盘过滤），参数网格只在训练期（2015–2020）挑；
2. 把他方法里的单个零件加到线上规则上（线上 = 业绩预期 + 业绩加速 + 情绪开关，20 只，止盈 2 倍 ATR、
   止损 3 倍 ATR、最长 10 天，净值跌破 20 日线空仓），看哪个零件在训练期和样本外都有帮助。
样本外 2021 至今只检验。扣佣金、滑点、印花税；涨停开盘买不到，一字跌停卖不掉。
"""

import json
import logging

import numpy as np
import pandas as pd

from . import config, model
from .alt_study import emotion_regime, iter_alt_factors
from .factors import Panel, eligible
from .live_slots import MODEL_FILE, apply_blend
from .slots import stats
from .stability import SWITCH_COST
from .winrate import adjusted_ohlc

log = logging.getLogger(__name__)

N = 20
MIN_AMOUNT = 2e8  # Stage 2：近 20 日日均成交额（元）
MAX_STOP = 0.10   # 止损离买入价超过 10% 的不做（支撑太远，风险报酬不好）
MIN_STOP = 0.02
W_CAP = 0.10      # 按风险算仓位时单只最多 10%

# 单独一套的参数网格（训练期挑）
META_K = (3, 4)
EXITS = (("ma10", None), ("ma20", None), ("ma20", 3.0))  # (趁弱卖的均线, 趁强卖的风险倍数)
RISKS = (0.003, 0.005)
MARKETS = ("none", "idx200", "breadth50")
MAX_HOLD = 60


def _rank01(f, mask):
    return model.xs_rank(f, mask) + 0.5


def features(p, elig, industry=None):
    """J Law 的各个条件（t 收盘后可知，T 开盘用）。返回 dict of 宽表 / 序列。"""
    ret = (p.close / p.preclose - 1).where(p.traded)
    adj = (1 + ret.fillna(0)).cumprod().astype("float32")
    k = adj / p.close
    hi, lo = (p.high * k).astype("float32"), (p.low * k).astype("float32")
    vol = p.volume.where(p.volume > 0)

    def ma(w):
        return adj.rolling(w, min_periods=int(w * 0.8)).mean().astype("float32")

    ma10, ma20, ma50, ma200 = ma(10), ma(20), ma(50), ma(200)
    f = {"adj": adj, "ma10": ma10, "ma20": ma20, "ma200": ma200}
    mret = ret.where(elig).mean(axis=1).fillna(0)
    mkt = (1 + mret).cumprod()
    # beta60
    r0 = ret.fillna(0)
    exy = r0.mul(mret, axis=0).rolling(60, min_periods=40).mean()
    ex = r0.rolling(60, min_periods=40).mean()
    em = mret.rolling(60, min_periods=40).mean()
    vm = mret.rolling(60, min_periods=40).var(ddof=0)
    beta = exy.sub(ex.mul(em, axis=0)).div(vm, axis=0)
    del exy, ex, r0
    amount20 = p.amount.rolling(20, min_periods=15).mean()
    stage2 = (adj > ma200) & (amount20 >= MIN_AMOUNT) & elig
    f["stage2"] = stage2 & (beta >= 1.0)
    f["above200"] = (adj > ma200) & elig
    del beta, amount20
    # 相对强度（IBD 式，近一季权重 40%）、RS 线
    r = sum(wt * (adj / adj.shift(d) - 1) for wt, d in ((0.4, 63), (0.2, 126), (0.2, 189), (0.2, 252)))
    rs = _rank01(r.where(adj.shift(252).notna()), elig)
    del r
    rsline = adj.div(mkt, axis=0)
    rs_up = rsline > rsline.ewm(span=21, adjust=False).mean()
    del rsline
    f["rs"] = rs
    ind_ok = pd.DataFrame(True, index=p.dates, columns=p.codes)
    if industry is not None:
        ind = pd.Series(industry).reindex(p.codes)
        med = {}
        for name, cols in ind.groupby(ind).groups.items():
            if len(cols) >= 5:
                med[name] = rs[list(cols)].median(axis=1)
        med = pd.DataFrame(med)
        top = med.rank(axis=1, pct=True) >= 0.7
        ind_ok = top.reindex(columns=ind.values).set_axis(p.codes, axis=1).fillna(False).astype(bool)
        ind_ok.index = p.dates
    f["leader"] = (rs >= 0.8) & rs_up & ind_ok
    f["leader_noind"] = (rs >= 0.8) & rs_up
    f["ind_ok"] = ind_ok
    # 回调：离近 10 日高点 3%–15%，仍在 50 日线上方
    hi10 = hi.rolling(10, min_periods=8).max()
    dd = adj / hi10 - 1
    pull = (dd <= -0.03) & (dd >= -0.15) & (adj > ma50 * 0.98)
    # 各项优势
    e_ma = pd.DataFrame(False, index=p.dates, columns=p.codes)
    for m in (ma10, ma20, ma50):
        e_ma |= (lo <= m * 1.02) & (adj >= m * 0.98)
    old = hi.shift(20).rolling(40, min_periods=30).max()  # t-60..t-21 的前高
    e_hz = (lo <= old * 1.03) & (adj >= old * 0.97) & (hi10 > old * 1.03)
    vol50 = vol.rolling(50, min_periods=40).mean()
    e_lowvol = vol.rolling(3, min_periods=3).mean() < 0.7 * vol50
    e_trend = ((ma10 > ma20) & (ma20 > ma50) & (ma50 > ma50.shift(10))
               & (lo.rolling(20, min_periods=15).min() > lo.shift(20).rolling(20, min_periods=15).min()))
    e_thrust = (((ret >= 0.05) & (vol >= 2 * vol50)).astype("float32").rolling(20, min_periods=1).max() > 0)
    e_pa = (p.close > p.open) & (p.close > p.preclose)
    meta = sum(e.astype("int8") for e in (e_ma, e_hz, e_lowvol, e_trend, e_thrust, e_pa))
    f["pull"] = pull
    f["meta"] = meta.where(pull, 0)
    del e_ma, e_hz, e_lowvol, e_trend, e_thrust, e_pa, hi10, dd, old, vol50
    # 止损：近 5 日最低价下 1%（复权价）
    f["support"] = (lo.rolling(5, min_periods=3).min() * 0.99).astype("float32")
    # 大盘
    csi = pd.read_csv(config.ROOT / "data" / "raw" / "csi300.csv", parse_dates=["date"]).set_index("date")["value"]
    csi = csi[~csi.index.duplicated(keep="last")].sort_index().reindex(p.dates).ffill()
    share200 = f["above200"].sum(axis=1) / elig.sum(axis=1).replace(0, np.nan)
    f["markets"] = {
        "none": pd.Series(True, index=p.dates),
        "idx200": csi > csi.rolling(200, min_periods=150).mean(),
        "breadth40": share200 >= 0.4,
        "breadth50": share200 >= 0.5,
    }
    f["share200"] = share200
    return f


def simulate(p, ohl, sc, gate, start, end, n=N, sizing=None, stop=("atr", 3.0), target=("atr", 2.0),
             weak=None, hold=10, support=None):
    """真实账户模拟，仓位可以不等权（每只的资金比例在买入时定，之后按每天调回该比例计，和 slots.simulate 一样）。

    sizing=None：每只 1/n；sizing=r：每只 = r / 止损幅度（最多 W_CAP），合计不超过 100%。
    stop=("atr", k) 止损 k 倍 ATR；("support",) 止损在 support（复权价，t 收盘可知）。
    target=("atr", k) / ("R", k)（k 倍止损幅度）/ None。weak：t 收盘为 True 的持仓 T 开盘卖（跌破均线）。"""
    o, h, l, atr, locked = ohl
    up, _ = p.limit_prices()
    up_v, raw_open, traded = up.to_numpy(), p.open.to_numpy(), p.traded.to_numpy()
    raw_close = p.close.to_numpy()
    sc_v = sc.to_numpy()
    gate_v = gate.reindex(p.dates).fillna(False).to_numpy(dtype=bool)
    weak_v = None if weak is None else weak.to_numpy()
    sup_v = None if support is None else support.to_numpy()
    adj_close = None
    if sup_v is not None:
        adj_close = ((p.close / p.preclose).where(p.traded).fillna(1.0).cumprod()).to_numpy()
    dates = p.dates
    i0 = max(dates.searchsorted(pd.Timestamp(start)), 15)
    i1 = min(dates.searchsorted(pd.Timestamp(end), side="right"), len(dates) - 1)
    buy_cost = config.SLIPPAGE + config.COMMISSION
    stamp_change = pd.Timestamp(config.STAMP_DUTY_CHANGE)
    held = {}  # 列 -> [买入价, 止盈价, 止损价, 买入日, 权重]
    daily, trades, expo = [], [], []
    for T in range(i0, i1):
        stamp = config.STAMP_DUTY_NEW if dates[T] >= stamp_change else config.STAMP_DUTY_OLD
        sell_cost = config.SLIPPAGE + config.COMMISSION + stamp
        r = 0.0
        for j in list(held):
            entry, tgt, stp, d0, w = held[j]
            px = None
            if not locked[T, j] and np.isfinite(o[T, j]):
                if T - d0 >= hold or (weak_v is not None and T > d0 and weak_v[T - 1, j]):
                    px = o[T, j]
                elif l[T, j] <= stp:
                    px = min(o[T, j], stp)
                elif h[T, j] >= tgt:
                    px = max(o[T, j], tgt)
            if px is None:
                continue
            r += w * (px / o[T, j] - 1 - sell_cost)
            trades.append(dict(entry=dates[d0], exit=dates[T], code=p.codes[j],
                               net=px * (1 - sell_cost) / (entry * (1 + buy_cost)) - 1))
            del held[j]
        used = sum(v[4] for v in held.values())
        if gate_v[T - 1] and len(held) < n and used < 0.999:
            row = sc_v[T - 1]
            ok = ~np.isnan(row)
            for j in np.argsort(-np.where(ok, row, -np.inf))[:max(n * 3, 60)]:
                if len(held) >= n or used >= 0.999 or not ok[j]:
                    break
                if j in held or not traded[T, j] or raw_open[T, j] >= up_v[T, j] - 0.005:
                    continue
                a = atr[T - 1, j]
                e = o[T, j]
                if not np.isfinite(a) or not np.isfinite(e):
                    continue
                if stop[0] == "atr":
                    sd = stop[1] * a
                else:
                    s_adj = sup_v[T - 1, j]
                    if not np.isfinite(s_adj) or not np.isfinite(raw_close[T - 1, j]):
                        continue
                    # 支撑价换成实际价，再按 T 开盘实际价算止损幅度
                    s_raw = s_adj / adj_close[T - 1, j] * raw_close[T - 1, j]
                    sd = 1 - s_raw / raw_open[T, j]
                    if sd > MAX_STOP or sd <= 0:
                        continue
                    sd = max(sd, MIN_STOP)
                if target is None:
                    tg = np.inf
                elif target[0] == "atr":
                    tg = e * (1 + target[1] * a)
                else:
                    tg = e * (1 + target[1] * sd)
                w = 1.0 / n if sizing is None else min(sizing / sd, W_CAP)
                w = min(w, 1.0 - used)
                if w < 0.005:
                    continue
                held[j] = [e, tg, e * (1 - sd), T, w]
                used += w
                r -= w * buy_cost
        for j, v in held.items():
            if np.isfinite(o[T + 1, j]) and o[T, j] > 0:
                r += v[4] * (o[T + 1, j] / o[T, j] - 1)
        daily.append((dates[T], r))
        expo.append(sum(v[4] for v in held.values()))
    d = pd.Series(dict(daily), dtype="float64")
    return d, trades, pd.Series(expo, index=d.index, dtype="float64")


def eq_overlay(daily, low=0.0):
    """净值跌破 20 日均线时仓位降到 low（线上 = 0 空仓；J Law 的“连续亏损时风险减半”= 0.5）。错开两行，同 stability.overlay。"""
    eq = (1 + daily).cumprod()
    pos = pd.Series(np.where(eq >= eq.rolling(20, min_periods=10).mean(), 1.0, low), index=daily.index)
    pos = pos.shift(2).fillna(1.0)
    return daily * pos - pos.diff().abs().fillna(0) * SWITCH_COST / 2, pos


def _summ(d, t, expo):
    st = stats(d, t)
    st.pop("yearly", None)
    st["exposure"] = float(expo.mean()) if len(expo) else None
    return st


def run(kline, basic, valuation=None, forecast=None, industry=None, dump=None):
    kline = kline() if callable(kline) else kline
    valuation = valuation() if callable(valuation) else valuation
    forecast = forecast() if callable(forecast) else forecast
    sm = json.loads(MODEL_FILE.read_text())
    weights = {k: v for k, v in sm["weights"].items() if v}
    p = Panel(kline, valuation)
    del kline, valuation
    elig = eligible(p, basic)
    ranks = {}
    for k, v in iter_alt_factors(p, forecast):
        if k in weights:
            ranks[k] = model.xs_rank(v, elig)
        del v
    live = apply_blend(model.score(ranks, weights, elig), p, forecast, elig, sm.get("blend"))
    del ranks
    emotion = emotion_regime(p, elig)
    ohl = adjusted_ohlc(p)
    f = features(p, elig, industry)
    end = str(p.dates[-1].date())
    periods = dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, end))
    cols = {}
    rows = []

    def evaluate(name, group, sc, gate, low=0.0, **kw):
        row = dict(name=name, group=group)
        parts = []
        for k, (s, e) in periods.items():
            d, t, ex = simulate(p, ohl, sc, gate, s, e, **kw)
            dd, pos = eq_overlay(d, low)
            row[k] = _summ(dd, t, ex * pos)
            row[k]["signals_per_day"] = float(sc.loc[s:e].notna().sum(axis=1).mean())
            parts.append(pd.DataFrame({name + "|raw": d, name + "|held": ex, name + "|low": low}))
        if dump is not None:
            df = pd.concat(parts)
            for c in df:
                cols[c] = df[c]
        rows.append(row)
        log.info("%s: train %.3f/%.3f/%.2f test %.3f/%.3f/%.2f", name,
                 row["train"].get("ann_return", 0), row["train"].get("max_drawdown", 0), row["train"].get("sharpe", 0),
                 row["test"].get("ann_return", 0), row["test"].get("max_drawdown", 0), row["test"].get("sharpe", 0))
        return row

    live_kw = dict(n=sm["n"], hold=sm["max_hold"], stop=("atr", float(sm["stop_loss"][3:])),
                   target=("atr", float(sm["take_profit"][3:])))
    evaluate("线上规则（对照）", "live", live, emotion, **live_kw)

    # ---- 1. 照 J Law 方法单独一套 ----
    base = f["stage2"] & f["leader"]
    for K in META_K:
        sig = f["rs"].where(base & (f["meta"] >= K))
        for wk, R in EXITS:
            weak = f["adj"] < f[wk]
            for risk in RISKS:
                for mk in MARKETS:
                    nm = (f"J Law：Stage2 领头羊 + M.E.T.A.≥{K}｜跌破 {wk[2:]} 日线卖" + (f"或到 {R:g}R 止盈" if R else "")
                          + f"｜每笔风险 {risk:.1%}｜大盘 {mk}")
                    evaluate(nm, "jlaw", sig, f["markets"][mk], low=0.5, n=30, sizing=risk, stop=("support",),
                             target=None if R is None else ("R", R), weak=weak, hold=MAX_HOLD, support=f["support"])
    # 第二轮（公平起见）：回调买在均线附近，收盘稍破均线就卖太紧；改成跌破均线 3% 才卖，止损也试 3 倍 ATR
    for K in META_K:
        sig = f["rs"].where(base & (f["meta"] >= K))
        for wk in ("ma10", "ma20"):
            weak = f["adj"] < f[wk] * 0.97
            for stop_name, stop in (("支撑下止损", ("support",)), ("3 倍 ATR 止损", ("atr", 3.0))):
                nm = f"J Law 放宽：M.E.T.A.≥{K}｜跌破 {wk[2:]} 日线 3% 才卖｜{stop_name}｜每笔风险 0.5%｜大盘 idx200"
                evaluate(nm, "jlaw2", sig, f["markets"]["idx200"], low=0.5, n=30, sizing=0.005, stop=stop,
                         target=None, weak=weak, hold=MAX_HOLD, support=f["support"])
    jl = [r for r in rows if r["group"] in ("jlaw", "jlaw2")]
    pick = max(jl, key=lambda r: r["train"].get("sharpe", -9))
    pick["chosen"] = True

    # ---- 2. 单个零件加到线上规则 ----
    parts = [
        ("线上 + 只买 Stage 2（200 日线上方、成交额 ≥ 2 亿、beta ≥ 1）", dict(sc=live.where(f["stage2"]))),
        ("线上 + 只买 200 日线上方", dict(sc=live.where(f["above200"]))),
        ("线上 + 只买领头羊（RS 前 20%、RS 线向上、强势行业）", dict(sc=live.where(f["leader"]))),
        ("线上 + 只买强势行业里的", dict(sc=live.where(f["ind_ok"]))),
        ("线上 + RS 排名加分 0.3", dict(sc=(live + 0.3 * (f["rs"] - 0.5).fillna(0)).where(live.notna()))),
        ("线上 + 只在回调里买（离 10 日高点 3–15%）", dict(sc=live.where(f["pull"]))),
        ("线上 + M.E.T.A. 优势数加分（每项 0.1）", dict(sc=(live + 0.1 * f["meta"]).where(live.notna()))),
        ("线上 + 大盘（沪深300）在 200 日线上方才买", dict(gate=emotion & f["markets"]["idx200"])),
        ("线上 + 200 日线上方股票 ≥ 40% 才买", dict(gate=emotion & f["markets"]["breadth40"])),
        ("线上 + 200 日线上方股票 ≥ 50% 才买", dict(gate=emotion & f["markets"]["breadth50"])),
        ("线上 + 跌破 10 日线卖（不设止盈，最长 20 天）",
         dict(kw=dict(target=None, weak=f["adj"] < f["ma10"], hold=20))),
        ("线上 + 跌破 10 日线也卖（保留止盈止损和 10 天）", dict(kw=dict(weak=f["adj"] < f["ma10"]))),
        ("线上 + 止损放在支撑下（近 5 日低点下 1%）", dict(kw=dict(stop=("support",), support=f["support"]))),
        ("线上 + 每笔风险 0.3% 定仓位（止损 3 倍 ATR，单只 ≤ 10%）", dict(kw=dict(sizing=0.003, n=40))),
        ("线上 + 每笔风险 0.5% 定仓位（止损 3 倍 ATR，单只 ≤ 10%）", dict(kw=dict(sizing=0.005, n=40))),
        ("线上 + 净值跌破 20 日线时风险减半（不空仓）", dict(low=0.5)),
    ]
    for nm, spec in parts:
        kw = dict(live_kw)
        kw.update(spec.get("kw", {}))
        evaluate(nm, "part", spec.get("sc", live), spec.get("gate", emotion), low=spec.get("low", 0.0), **kw)

    out = dict(updated=end, rows=rows, chosen=pick["name"],
               share200_now=float(f["share200"].iloc[-1]) if len(f["share200"]) else None)
    if dump is not None:
        df = pd.DataFrame(cols)
        df.index.name = "date"
        df.to_csv(dump, float_format="%.6g")
    return out


def _p(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def to_markdown(res):
    L = [f"# J Law 方法回测（数据到 {res['updated']}）", "",
         "100% 选股资金口径，净值跌破 20 日线空仓（J Law 单独一套是风险减半），已扣成本。训练 2015–2020，样本外 2021 至今。", "",
         f"J Law 单独一套里训练期夏普最高的：{res['chosen']}", "",
         "| 方案 | 训练年化 | 训练回撤 | 训练夏普 | 训练胜率 | 样本外年化 | 样本外回撤 | 样本外夏普 | 样本外胜率 | 样本外笔数 | 平均仓位 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        a, b = r["train"], r["test"]
        L.append(f"| {r['name']}{' ★' if r.get('chosen') else ''} | {_p(a.get('ann_return'))} | {_p(a.get('max_drawdown'))} | "
                 f"{a.get('sharpe', 0):.2f} | {_p(a.get('trade_win'))} | {_p(b.get('ann_return'))} | {_p(b.get('max_drawdown'))} | "
                 f"{b.get('sharpe', 0):.2f} | {_p(b.get('trade_win'))} | {b.get('trades', 0)} | {_p(b.get('exposure'), 0)} |")
    return "\n".join(L) + "\n"


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "jlaw.md").write_text(to_markdown(res))
    (config.PAGE_DIR / "jlaw.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
