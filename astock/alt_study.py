"""另类因子对照实验（2026-10-28 用户要求“结合短线技术分析、基本面分析、预期等筛选进行另类因子回测”）。

三组新因子，全部只用信号日 t 收盘后已经公开的数据：
- 短线技术：RSI、MACD 柱、布林带位置、KDJ 的 J 值、20 日突破、量价相关、均线多头排列、OBV 变化、上影线。
- 基本面：由每日估值（市盈率 TTM、市净率、市销率）推出 ROE、近一年盈利增长、近一年营收增长。
  估值里的盈利和营收只在财报公布后才更新，所以天然是“当时可知”的数据。
- 业绩预期：近 60 日盈利变化（新财报带来的盈利意外）、业绩预告的预告变动幅度和方向（公告后下一个交易日起生效）。

每组和“全部”分别只在训练期（2015–2020）按持有 5 天的扣费收益定权重，样本外（2021 以后）只检验：
得分十分位的收益、固定规则的分散组合（持有 20 只，排名跌出前 80 才卖），以及叠加“恐慌时买、贪婪时卖”。
组合规则对每组都一样，不在样本外挑参数。
"""

import json
import logging

import numpy as np
import pandas as pd

from . import config, model, portfolio
from .factors import FACTORS, Panel, eligible, iter_factors, trade_outcomes

log = logging.getLogger(__name__)

ALT_FACTORS = {
    "rsi14": "14 日 RSI",
    "macd_hist": "MACD 柱（相对股价）",
    "boll_b": "布林带位置 %B（20 日，2 倍标准差）",
    "kdj_j": "KDJ 的 J 值（9 日）",
    "break20": "收盘价 / 前 20 日最高价（突破）",
    "pv_corr10": "近 10 日涨跌与成交量变化的相关（量价配合）",
    "ma_align": "均线多头排列（5>10>20>60 的个数）",
    "obv20": "近 20 日 OBV 变化 / 平均成交量",
    "shadow20": "近 20 日平均上影线（相对振幅）",
    "roe": "ROE（市净率 / 市盈率）",
    "eps_g250": "近一年盈利增长（每股收益 TTM）",
    "sales_g250": "近一年营收增长（每股营收 TTM）",
    "eps_chg60": "近 60 日盈利变化（新财报的盈利意外）",
    "fc_chg": "业绩预告变动幅度（中值，公告后 120 个交易日内）",
    "fc_sign": "业绩预告方向（预增/扭亏 +1，预减/首亏 -1）",
    "fc_fresh20": "近 20 个交易日内新发的业绩预告方向",
    # 短线高手和量化经理公开的做法（见 MASTERS）
    "ind_mom5": "所属行业近 5 日涨幅（板块强度）",
    "ind_rel5": "个股近 5 日涨幅减去行业平均（板块内的龙头）",
    "seal_up1": "昨日涨停且收在最高价（封板）",
    "boards": "连续涨停天数（连板高度）",
    "first_board": "昨日首板（此前 20 日没有涨停）",
    "rs250": "近一年涨幅，剔除最近 20 日（相对强度 RS）",
    "vcp": "近 10 日振幅 / 近 60 日振幅（波动收缩）",
    "vol_dry": "近 5 日成交量 / 近 60 日成交量（缩量）",
    "dip_uptrend": "上升趋势中的回调（60 日线向上时的近 5 日跌幅）",
    "fscore_lite": "简化 F-score（盈利为正、盈利增长、营收增长、ROE 提升）",
}

# 公开案例 -> 可回测的规则。只用公开书籍、论文和广为流传的做法，规则写死，不在样本外调
MASTERS = [
    ("A股短线游资：龙头战法", "做最强板块里最强的那只", "ind_mom5、ind_rel5"),
    ("A股短线游资：打板、首板", "买封死涨停的强势股，首板溢价高于高位连板", "seal_up1、first_board、boards"),
    ("A股短线：情绪周期", "全市场涨停家数回升时做多，退潮时空仓", "组合叠加“情绪开关”：近 5 日平均涨停家数高于近 20 日平均才持仓"),
    ("欧奈尔 CAN SLIM", "只买相对强度前列、接近新高、盈利加速的股票", "rs250、high250、eps_g250、eps_chg60"),
    ("米奈尔维尼 VCP", "波动逐级收缩、成交量萎缩后突破", "vcp、vol_dry、break20"),
    ("强势股低吸", "上升趋势中回调到均线附近买入", "dip_uptrend、ma20gap"),
    ("AQR（阿斯内斯）：价值 + 动量 + 质量 + 低风险", "多因子组合；质量减垃圾（QMJ）、押注低贝塔（BAB）", "ep、bp、roe、beta60、idiovol60、mom60"),
    ("皮奥特洛斯基 F-score", "财务质量打分，买分高的便宜股", "fscore_lite、bp"),
    ("文艺复兴等统计套利", "剔除行业影响后的短期反转", "ind_rel5（反向即行业中性反转）、rev5"),
    ("国内头部量化（公开路演口径）", "小市值、低换手、低特异波动、反转、预期调整，做行业和市值中性", "size、turn20、idiovol60、rev20、fc_chg"),
]
NAMES = {**FACTORS, **ALT_FACTORS}

TECH_OLD = ("rev5", "rev20", "ret1", "clv", "ma20gap", "overnight20", "intraday20", "turn_ratio", "vol_spike",
            "gap1", "maxret20", "limitup20", "high250", "low250", "downstreak", "skew20", "range20", "vol20")
GROUPS = {
    "短线技术": ("rsi14", "macd_hist", "boll_b", "kdj_j", "break20", "pv_corr10", "ma_align", "obv20", "shadow20",
             *TECH_OLD),
    "基本面": ("ep", "bp", "sp", "roe", "eps_g250", "sales_g250", "size"),
    "业绩预期": ("eps_chg60", "fc_chg", "fc_sign", "fc_fresh20"),
    "高手与量化经理": ("ind_mom5", "ind_rel5", "seal_up1", "first_board", "boards", "rs250", "high250", "eps_g250",
                  "eps_chg60", "vcp", "vol_dry", "break20", "dip_uptrend", "ma20gap", "ep", "bp", "roe", "beta60",
                  "idiovol60", "mom60", "fscore_lite", "rev5", "size", "turn20", "rev20", "fc_chg"),
}
GROUPS["全部"] = tuple(dict.fromkeys(k for g in GROUPS.values() for k in g))
GOOD_TYPES = ("预增", "略增", "扭亏", "续盈", "预盈", "增长")
BAD_TYPES = ("预减", "略减", "首亏", "续亏", "预亏", "下降")
HOLD = 5  # 定权重和十分位都用持有 5 天的扣费收益（T 开盘买，T+5 开盘卖）
PORT_N, PORT_BUFFER = 20, 4
CONTRA_BUY, CONTRA_SELL = 20, 70  # 上一轮在训练期挑出的恐慌买、贪婪卖线
FORECAST_DAYS = 120
LIVE_GROUP = "业绩预期"  # 2026-10-28 用户要求上线：每日选股用这一组 + 情绪开关
LIVE_FILE = config.ROOT / "astock" / "forecast_model.json"


def _ema(x, span):
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def forecast_frames(p, forecast):
    """业绩预告对齐到交易日：公告日之后的第一个交易日起生效，最多保留 FORECAST_DAYS 个交易日。"""
    empty = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype="float32")
    if forecast is None or forecast.empty:
        return empty, empty.copy(), empty.copy()
    f = forecast[forecast["code"].isin(p.codes)].sort_values("pub")
    pos = p.dates.searchsorted(f["pub"].values, side="right")
    keep = pos < len(p.dates)
    f, pos = f[keep], pos[keep]
    mid = f[["up", "dn"]].mean(axis=1).clip(-500, 500)
    sign = np.where(f["type"].isin(GOOD_TYPES), 1.0, np.where(f["type"].isin(BAD_TYPES), -1.0, 0.0))
    col = p.codes.get_indexer(f["code"])

    def place(vals):
        a = np.full((len(p.dates), len(p.codes)), np.nan, dtype="float32")
        a[pos, col] = vals  # 同一天多条时后写的覆盖，已按公告日排好
        return pd.DataFrame(a, index=p.dates, columns=p.codes)

    chg = place(mid.to_numpy(dtype="float32")).ffill(limit=FORECAST_DAYS)
    sgn_event = place(sign.astype("float32"))
    return chg, sgn_event.ffill(limit=FORECAST_DAYS), sgn_event.ffill(limit=20)


def emotion_regime(p, elig):
    """情绪开关：全市场近 5 日平均涨停家数高于近 20 日平均。信号日 t 收盘后可知。"""
    up, _ = p.limit_prices()
    cnt = ((p.close >= up - 0.005) & p.traded & elig).sum(axis=1).astype(float)
    return cnt.rolling(5, min_periods=5).mean() > cnt.rolling(20, min_periods=15).mean()


def iter_alt_factors(p, forecast=None, industry=None):
    def fin(v):
        return v.where(p.traded).astype("float32")

    ret = (p.close / p.preclose - 1).where(p.traded)
    adj = (1 + ret.fillna(0)).cumprod()
    # 复权的最高、最低价：按收盘价的复权比例换算
    k = adj / p.close
    hi, lo = p.high * k, p.low * k

    up = ret.clip(lower=0).rolling(14, min_periods=10).mean()
    dn = (-ret.clip(upper=0)).rolling(14, min_periods=10).mean()
    yield "rsi14", fin(up / (up + dn))
    del up, dn
    dif = _ema(adj, 12) - _ema(adj, 26)
    yield "macd_hist", fin((dif - _ema(dif, 9)) / adj)
    del dif
    ma20, sd20 = adj.rolling(20, min_periods=15).mean(), adj.rolling(20, min_periods=15).std()
    yield "boll_b", fin((adj - (ma20 - 2 * sd20)) / (4 * sd20))
    del sd20
    h9, l9 = hi.rolling(9, min_periods=7).max(), lo.rolling(9, min_periods=7).min()
    rsv = ((adj - l9) / (h9 - l9)).where(h9 > l9) * 100
    kk = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    dd = kk.ewm(alpha=1 / 3, adjust=False).mean()
    yield "kdj_j", fin(3 * kk - 2 * dd)
    del h9, l9, rsv, kk, dd
    yield "break20", fin(adj / adj.shift(1).rolling(20, min_periods=15).max())
    dv = np.log(p.volume / p.volume.shift(1))
    yield "pv_corr10", fin(ret.rolling(10, min_periods=8).corr(dv))
    del dv
    ma5, ma10, ma60 = (adj.rolling(w, min_periods=int(w * 0.75)).mean() for w in (5, 10, 60))
    yield "ma_align", fin((ma5 > ma10).astype("float32") + (ma10 > ma20) + (ma20 > ma60))
    del ma5, ma10, ma60, ma20
    obv = (np.sign(ret.fillna(0)) * p.volume.fillna(0)).cumsum()
    yield "obv20", fin((obv - obv.shift(20)) / p.volume.rolling(20, min_periods=15).mean() / 20)
    del obv
    rng = (p.high - p.low).where(p.high > p.low)
    shadow = (p.high - np.maximum(p.open, p.close)) / rng
    yield "shadow20", fin(shadow.rolling(20, min_periods=15).mean())
    del shadow, rng, hi, lo, k

    pe, pb, ps = (p.valuation.get(c) for c in ("peTTM", "pbMRQ", "psTTM"))
    if pe is not None and pb is not None:
        pos = (pe > 0) & (pb > 0)
        yield "roe", fin((pb / pe).where(pos).ffill(limit=5))
        # 每股收益 TTM（复权）= 复权价 × 盈利收益率；用差值除以绝对值，亏损股也有意义
        eps = (adj * (1 / pe.where(pe.abs() > 1e-6))).ffill(limit=5)
        yield "eps_g250", fin(((eps - eps.shift(250)) / eps.shift(250).abs()).clip(-10, 10))
        yield "eps_chg60", fin(((eps - eps.shift(60)) / eps.shift(60).abs()).clip(-10, 10))
        del eps
    if ps is not None:
        sps = (adj * (1 / ps.where(ps > 1e-6))).ffill(limit=5)
        yield "sales_g250", fin((sps / sps.shift(250) - 1).clip(-10, 10))
        del sps

    chg, sgn, fresh = forecast_frames(p, forecast)
    yield "fc_chg", fin(chg)
    yield "fc_sign", fin(sgn)
    yield "fc_fresh20", fin(fresh)
    del chg, sgn, fresh

    # 短线高手和量化经理的做法
    rev5 = adj / adj.shift(5) - 1
    if industry is not None:
        ind = industry.reindex(p.codes).fillna("未知")
        ind_mean = rev5.where(p.traded).T.groupby(ind.values).mean().T  # 行业等权近 5 日涨幅
        ind_mom = ind_mean.reindex(columns=ind.values)
        ind_mom.columns = p.codes
        yield "ind_mom5", fin(ind_mom)
        yield "ind_rel5", fin(rev5 - ind_mom)
        del ind_mean, ind_mom
    up_px, _ = p.limit_prices()
    lim = (p.close >= up_px - 0.005) & p.traded
    yield "seal_up1", fin((lim & (p.high <= p.close + 0.005)).astype("float32"))
    lim_np = lim.to_numpy()
    streak = np.zeros(lim_np.shape, dtype="float32")
    for i in range(len(lim_np)):
        streak[i] = np.where(lim_np[i], (streak[i - 1] if i else 0) + 1, 0)
    yield "boards", fin(pd.DataFrame(streak, index=p.dates, columns=p.codes))
    del streak, lim_np
    prior = lim.astype("float32").shift(1).rolling(20, min_periods=15).sum()
    yield "first_board", fin((lim & (prior == 0)).astype("float32"))
    del prior, lim, up_px
    yield "rs250", fin(adj.shift(20) / adj.shift(250) - 1)
    rng_pct = (p.high - p.low) / p.preclose
    yield "vcp", fin(rng_pct.rolling(10, min_periods=8).mean() / rng_pct.rolling(60, min_periods=40).mean())
    del rng_pct
    yield "vol_dry", fin(p.volume.rolling(5, min_periods=4).mean() / p.volume.rolling(60, min_periods=40).mean())
    ma60 = adj.rolling(60, min_periods=45).mean()
    trend_up = (ma60 > ma60.shift(10)) & (adj > ma60)
    yield "dip_uptrend", fin((-rev5).where(trend_up, -1.0))
    del ma60, trend_up
    if pe is not None and pb is not None and ps is not None:
        eps = adj / pe.where(pe.abs() > 1e-6)
        sps = adj / ps.where(ps > 1e-6)
        roe = (pb / pe).where((pe > 0) & (pb > 0))
        score = ((eps > 0).astype("float32") + (eps > eps.shift(250)) + (sps > sps.shift(250))
                 + (roe > roe.shift(250)))
        yield "fscore_lite", fin(score.where(pe.notna()).ffill(limit=5))


def fit_weights(ranks, target, mask, start, end, step=HOLD):
    """训练期 IC 定权重。标签是持有 5 天的收益，相邻日子重叠，t 值只用每 5 天取一个的样本算，免得高估显著性。"""
    stats = {}
    for name, r in ranks.items():
        ic = model.daily_ic(r, target, mask).loc[start:end].dropna()
        sub = ic.iloc[::step]
        mean, std = float(ic.mean()), float(ic.std())
        t = float(sub.mean() / sub.std() * np.sqrt(len(sub))) if len(sub) > 2 and sub.std() > 0 else 0.0
        stats[name] = dict(ic_mean=mean, ic_ir=mean / std if std > 0 else 0.0, t=t)
    w = {k: (v["ic_ir"] if abs(v["t"]) >= 2 else 0.0) for k, v in stats.items()}
    total = sum(abs(x) for x in w.values()) or 1.0
    return {k: v / total for k, v in w.items()}, stats


def deciles(sc, target, mask, start, end):
    """样本外按得分分十组，每组持有 5 天扣费收益的平均（每笔）。"""
    q = sc.where(mask).rank(axis=1, pct=True).loc[start:end]
    tg = target.where(mask).loc[start:end]
    return [float(np.nanmean(tg.where((q > (d - 1) / 10) & (q <= d / 10)).to_numpy())) for d in range(1, 11)]


def run(kline, basic, valuation=None, forecast=None, industry=None):
    kline = kline() if callable(kline) else kline
    valuation = valuation() if callable(valuation) else valuation
    forecast = forecast() if callable(forecast) else forecast
    p = Panel(kline, valuation)
    del kline, valuation
    elig = eligible(p, basic)
    need = set(GROUPS["全部"])
    ranks = {}
    for k, v in iter_factors(p):
        if k in need:
            ranks[k] = model.xs_rank(v, elig)
        del v
    for k, v in iter_alt_factors(p, forecast, industry):
        ranks[k] = model.xs_rank(v, elig)
        del v
    log.info("panel %d days x %d codes, %d factors", len(p.dates), len(p.codes), len(ranks))
    out = trade_outcomes(p)
    target = out[f"hold{HOLD}_net"]
    mask = elig & out["buyable"]
    for key in [k for k in out if k not in ("buyable", f"hold{HOLD}_net")]:
        del out[key]
    end = p.dates[-1]
    train_end = p.dates[p.dates <= pd.Timestamp(config.TRAIN_END)][-(HOLD + 1)]
    periods = dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, end))

    # 每个因子单独：训练期 / 样本外 IC
    _, fstats = fit_weights(ranks, target, mask, config.TRAIN_START, train_end)
    for name, r in ranks.items():
        ic = model.daily_ic(r, target, mask).loc[config.TEST_START:].dropna()
        fstats[name]["test_ic"] = float(ic.mean()) if len(ic) else None

    ctx = portfolio.context(p)
    fear = model.load_fear(p.dates)
    regime = model.contrarian_regime(fear, CONTRA_BUY, CONTRA_SELL)
    emotion = emotion_regime(p, elig)
    bench = {k: portfolio.stats(portfolio.market(ctx, elig, None, s, e), []) for k, (s, e) in periods.items()}
    bench_regime = {k: portfolio.stats(portfolio.market(ctx, elig, regime, s, e), []) for k, (s, e) in periods.items()}
    bench_emotion = {k: portfolio.stats(portfolio.market(ctx, elig, emotion, s, e), []) for k, (s, e) in periods.items()}
    for b in (bench, bench_regime, bench_emotion):
        for v in b.values():
            v.pop("_equity", None)

    groups = []
    for gname, members in GROUPS.items():
        sub = {k: ranks[k] for k in members if k in ranks}
        w, _ = fit_weights(sub, target, mask, config.TRAIN_START, train_end)
        row = dict(group=gname, weights={k: round(v, 3) for k, v in w.items() if v})
        if not row["weights"]:
            groups.append(row)
            continue
        sc = model.score(sub, w, elig)
        row["deciles"] = deciles(sc, target, mask, config.TEST_START, end)
        for label, reg in (("always", None), ("contra", regime), ("emotion", emotion)):
            for k, (s, e) in periods.items():
                d, t = portfolio.simulate(ctx, sc, fear, PORT_N, PORT_BUFFER, None, s, e, regime=reg)
                st = portfolio.stats(d, t)
                if label == "always" and k == "test":
                    row["equity"] = st.get("_equity")
                st.pop("_equity", None)
                row[f"{label}_{k}"] = st
        log.info("%s: weights %s test %s", gname, row["weights"], row["always_test"].get("ann_return"))
        groups.append(row)
    return dict(updated=str(end.date()), factors=fstats, groups=groups, bench=bench, bench_regime=bench_regime,
                bench_emotion=bench_emotion)


def _p(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def to_markdown(res):
    lines = ["# 另类因子回测：短线技术 + 基本面 + 业绩预期 + 高手与量化经理的做法", "",
             f"数据截至 {res['updated']}。训练期 2015–2020 定权重，样本外 2021 至今只检验。"
             "收益口径：T 日开盘买入，持有 5 个交易日后开盘卖出，扣佣金、双边滑点和印花税。", "",
             "## 公开案例整理成的因子", "",
             "来源是公开书籍、论文和广为流传的做法；具体到个人的“战绩”无法核实，这里只取可以写成规则的部分。", "",
             "| 来源 | 核心做法 | 对应因子或规则 |", "| --- | --- | --- |"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in MASTERS]
    lines += ["", "行业用的是当前的证监会行业分类（baostock），对已退市或改过行业的股票略有前视。", "",
             "## 各组因子：样本外得分分组（每笔持有 5 天的扣费收益）", "",
             "| 因子组 | 用到的因子数 | 最低 10% | 中间（第 5 组） | 最高 10% | 最高减最低 |", "| --- | --- | --- | --- | --- | --- |"]
    for g in res["groups"]:
        if "deciles" not in g:
            lines.append(f"| {g['group']} | 0（训练期没有显著因子） | — | — | — | — |")
            continue
        d = g["deciles"]
        lines.append(f"| {g['group']} | {len(g['weights'])} | {_p(d[0], 2)} | {_p(d[4], 2)} | {_p(d[9], 2)} | {_p(d[9] - d[0], 2)} |")
    lines += ["", f"## 分散组合（持有 {PORT_N} 只，排名跌出前 {PORT_N * PORT_BUFFER} 才卖，扣费）", "",
              "| 因子组 | 训练期年化 | 样本外年化 | 样本外夏普 | 样本外最大回撤 | 每笔胜率 | "
              f"叠加恐慌买贪婪卖（≤{CONTRA_BUY} 买 ≥{CONTRA_SELL} 卖）样本外年化 | 叠加后最大回撤 | 叠加情绪开关样本外年化 | 叠加后最大回撤 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g in res["groups"]:
        if "always_test" not in g:
            continue
        a, t, c, em = g["always_train"], g["always_test"], g["contra_test"], g["emotion_test"]
        lines.append(f"| {g['group']} | {_p(a.get('ann_return'))} | {_p(t.get('ann_return'))} | {t.get('sharpe', 0):.2f} | "
                     f"{_p(t.get('max_drawdown'))} | {_p(t.get('trade_win'))} | {_p(c.get('ann_return'))} | {_p(c.get('max_drawdown'))} | "
                     f"{_p(em.get('ann_return'))} | {_p(em.get('max_drawdown'))} |")
    b, br, be = res["bench"], res["bench_regime"], res["bench_emotion"]
    lines.append(f"| 对照：等权持有全市场（不扣费） | {_p(b['train'].get('ann_return'))} | {_p(b['test'].get('ann_return'))} | "
                 f"{b['test'].get('sharpe', 0):.2f} | {_p(b['test'].get('max_drawdown'))} | — | "
                 f"{_p(br['test'].get('ann_return'))} | {_p(br['test'].get('max_drawdown'))} | "
                 f"{_p(be['test'].get('ann_return'))} | {_p(be['test'].get('max_drawdown'))} |")
    lines += ["", "## 每个因子（对持有 5 天扣费收益的 IC）", "",
              "t 值按每 5 天取一个样本计算（标签重叠，不这样会高估显著性）；|t| < 2 的因子不进组合。", "",
              "| 因子 | 含义 | 训练期 IC | t 值 | 样本外 IC | 方向是否一致 |", "| --- | --- | --- | --- | --- | --- |"]
    order = sorted(res["factors"].items(), key=lambda kv: -abs(kv[1]["t"]))
    for k, s in order:
        tic = s.get("test_ic")
        same = "—" if tic is None or abs(s["t"]) < 2 else ("是" if np.sign(tic) == np.sign(s["ic_mean"]) else "否")
        lines.append(f"| {k} | {NAMES.get(k, k)} | {s['ic_mean']:.4f} | {s['t']:.1f} | {'—' if tic is None else f'{tic:.4f}'} | {same} |")
    lines += ["", "仅供个人研究参考，不构成投资建议。"]
    return "\n".join(lines) + "\n"


def page_data(res):
    groups = []
    for g in res["groups"]:
        row = {k: v for k, v in g.items() if k != "equity"}
        eq = g.get("equity")
        if eq is not None and len(eq):
            row["equity"] = {str(d.date()): round(float(v), 4) for d, v in eq.iloc[::5].items()}
        groups.append(row)
    return dict(updated=res["updated"], groups=groups, bench=res["bench"], bench_regime=res["bench_regime"],
                bench_emotion=res["bench_emotion"], masters=[dict(source=a, idea=b, factors=c) for a, b, c in MASTERS],
                factors=[dict(factor=k, name=NAMES.get(k, k), **v) for k, v in res["factors"].items()])


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "alt_study.md").write_text(to_markdown(res))
    (config.PAGE_DIR / "alt_study.json").write_text(json.dumps(page_data(res), ensure_ascii=False, default=float))
    write_live_model(res)


def write_live_model(res):
    """把上线那一组在训练期定的权重和回测结果写进 astock/forecast_model.json，每日选股读它。"""
    g = next((g for g in res["groups"] if g["group"] == LIVE_GROUP and g.get("weights")), None)
    if g is None:
        return
    bt = {k: {kk: vv for kk, vv in g[f"emotion_{k}"].items() if kk != "_equity"} for k in ("train", "test")}
    m = dict(strategy=f"{LIVE_GROUP} + 涨停情绪开关", group=LIVE_GROUP, weights=g["weights"], n=PORT_N, buffer=PORT_BUFFER,
             gate="emotion", hold_note=f"持有 {PORT_N} 只，排名跌出前 {PORT_N * PORT_BUFFER} 才卖；情绪开关关时清仓",
             train=[config.TRAIN_START, config.TRAIN_END], updated=res["updated"], backtest=bt)
    LIVE_FILE.write_text(json.dumps(m, ensure_ascii=False, indent=2, default=float) + "\n")
