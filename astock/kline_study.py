"""纯 K 线（量价）选股研究（2026-11-18 用户：选股按业绩预期不大行，还是得以纯 K 线图说话，价格是骗不了人的；
先学习研究 K 线走势，结合以前的和现在最新的、得到验证的知识）。

做法：
1. 把学术文献和 A股卖方金工里反复被验证过的量价规律，写成只用开高低收、成交量、换手率的因子（LIB），
   每个因子的方向按文献事先定死（不是从数据里挑方向），先看它们在 A股训练期（2015–2020）和样本外（2021 至今）各自灵不灵；
2. 把因子按家族（反转、动量趋势、低风险、换手流动性、量价配合、K 线形态）等权合成选股得分，
   另加几种组合（A股文献组合、追涨型技术分析、训练期 IC 加权、训练期最显著 8 个、趋势中的回调、强势股里挑低风险）；
3. 每个得分都套进和线上完全一样的框架：情绪开关、20 只、止盈 2 倍 ATR、止损 3 倍 ATR、最长 10 天、
   净值跌破 20 日均线空仓；总组合 = 选股 35% + 纳指/黄金/国债 ETF 分仓闲钱停泊。和线上规则（业绩预期）同口径比；
4. 随机选股参照：全市场随机挑、在有业绩预告的股票里随机挑，各 20 次，看“选股”这一步到底值多少；
5. 折中方案：业绩预期只当过滤 / 只当排雷，K 线说了算；或线上得分上用价格过滤（已经在跌的不买、高位的不买、公告后价格确认才买）；
6. 检验用户的说法“好消息出在某些点位是忽悠人入套的”：业绩预喜公告按公告时的价格位置分组，看之后 10 天相对全市场的表现。

规则只在训练期挑，样本外只检验。只做研究，不改线上规则。
"""

import json
import logging

import numpy as np
import pandas as pd

from . import config, diversify as dv, model, redline
from .alt_study import FORECAST_DAYS, GOOD_TYPES, emotion_regime, forecast_frames, iter_alt_factors
from .factor_hunt import _summ
from .factors import Panel, eligible, iter_factors, trade_outcomes
from .live_slots import MODEL_FILE, apply_blend
from .slots import simulate
from .winrate import adjusted_ohlc

log = logging.getLogger(__name__)

N_RANDOM = 20  # 随机选股次数
N_RANDOM_MIX = 8  # 随机过滤 / 随机加分次数
BLEND_W = (0.15, 0.30, 1.0)
CONS_SLIPPAGE = 0.002
IC_HOLD = 10  # 单因子 IC 的标签：T 开盘买、T+10 开盘卖的扣费收益（和线上最长持有 10 天一致）
LONG_EXIT = (None, "atr3", 20)  # 敏感性：不设止盈、止损 3 倍 ATR、最长 20 天（趋势类因子需要更长的持有期）

# 因子：家族、事先定的方向（+1 = 数值越大越好）、说明、出处
LIB = {
    # —— 反转：A股里最稳的一类。Liu-Stambaugh-Yuan (2019, JFE)：A股有显著的一个月反转，没有美股那种中期动量 ——
    "rev5": ("反转", -1, "近 5 日涨幅", "短期反转（Jegadeesh 1990；Lehmann 1990）"),
    "rev20": ("反转", -1, "近 20 日涨幅", "一个月反转（Liu-Stambaugh-Yuan 2019）"),
    "ret1": ("反转", -1, "昨日涨幅", "日度反转"),
    "ma20gap": ("反转", -1, "收盘价偏离 20 日均线", "乖离率"),
    "rsi14": ("反转", -1, "14 日 RSI", "超买超卖"),
    "boll_b": ("反转", -1, "布林带位置 %B", "超买超卖"),
    "kdj_j": ("反转", -1, "KDJ 的 J 值", "超买超卖"),
    "ind_rel5": ("反转", -1, "近 5 日涨幅减去行业平均", "行业中性反转（统计套利常用）"),
    "downstreak": ("反转", 1, "连续下跌天数", "过度反应（De Bondt-Thaler 1985）"),
    # —— 动量趋势：美股经典，A股文献普遍认为弱或不存在；趋势因子在 A股要靠成交量才有效（Liu-Zhou-Zhu 2024, RAPS）——
    "rs250": ("动量趋势", 1, "近一年涨幅（剔除最近 20 日）", "中期动量（Jegadeesh-Titman 1993）"),
    "mom120": ("动量趋势", 1, "近半年涨幅（剔除最近 20 日）", "中期动量"),
    "mom60": ("动量趋势", 1, "近 60 日涨幅（剔除最近 5 日）", "短中期动量"),
    "high250": ("动量趋势", 1, "收盘价 / 近一年最高价", "52 周新高（George-Hwang 2004）"),
    "ma200gap": ("动量趋势", 1, "收盘价偏离 200 日均线", "长期均线趋势（Han-Zhou-Zhu 2016 趋势因子的长端）"),
    "ma_align": ("动量趋势", 1, "均线多头排列（5>10>20>60 的个数）", "均线系统"),
    "macd_hist": ("动量趋势", 1, "MACD 柱", "趋势指标"),
    "break20": ("动量趋势", 1, "收盘价 / 前 20 日最高价", "突破"),
    "fip": ("动量趋势", -1, "信息离散度：一年涨幅方向 ×（下跌天数占比 − 上涨天数占比）", "温水煮青蛙，小步慢涨的动量更持久（Da-Gurun-Warachka 2014）"),
    "cgo": ("动量趋势", 1, "浮盈：收盘价相对换手加权持仓成本", "处置效应（Grinblatt-Han 2005）"),
    # —— 低风险：彩票型股票被散户高估（Bali-Cakici-Whitelaw 2011；Ang et al. 2006），A股散户多、更明显 ——
    "vol20": ("低风险", -1, "近 20 日波动率", "低波动异象"),
    "idiovol60": ("低风险", -1, "近 60 日特质波动率", "特质波动率之谜（Ang-Hodrick-Xing-Zhang 2006）"),
    "beta60": ("低风险", -1, "近 60 日 beta", "押注低贝塔（Frazzini-Pedersen 2014）"),
    "maxret20": ("低风险", -1, "近 20 日最大单日涨幅", "彩票效应 MAX（Bali-Cakici-Whitelaw 2011）"),
    "skew20": ("低风险", -1, "近 20 日收益偏度", "彩票偏好"),
    "range20": ("低风险", -1, "近 20 日平均振幅", "低波动异象"),
    "limitup20": ("低风险", -1, "近 20 日涨停次数", "A股涨停板的彩票效应"),
    "salience20": ("低风险", -1, "显著性：近 20 日里“特别扎眼”的涨跌日的加权收益", "显著性理论（Cosemans-Frehen 2021；近似算法）"),
    # —— 换手流动性：A股机器学习研究里最重要的一类预测变量（Leippold-Wang-Zhou 2022, JFE）；异常换手 = 散户情绪（LSY 2019）——
    "turn20": ("换手流动性", -1, "近 20 日平均换手率", "低换手溢价"),
    "abturn": ("换手流动性", -1, "异常换手：近 20 日换手 / 近 250 日换手", "情绪因子 PMO（Liu-Stambaugh-Yuan 2019）"),
    "turn_ratio": ("换手流动性", -1, "近 5 日换手 / 近 60 日换手", "短期异常换手"),
    "turn_std20": ("换手流动性", -1, "近 20 日换手率的波动", "换手波动"),
    "vol_spike": ("换手流动性", -1, "昨日成交量 / 近 20 日均量", "放量 = 散户关注（Barber-Odean 2008）"),
    "amihud20": ("换手流动性", 1, "非流动性：近 20 日 |涨跌幅| / 成交额", "非流动性溢价（Amihud 2002）"),
    "amount20": ("换手流动性", -1, "近 20 日成交额", "流动性"),
    # —— 量价配合：技术分析说“量价齐升”好；A股金工研究反而发现量价同步是负向信号 ——
    "pv_corr10": ("量价配合", -1, "近 10 日涨跌与成交量变化的相关", "量价相关性（A股金工：负向）"),
    "pv_corr20": ("量价配合", -1, "近 20 日收盘价与换手率的相关", "量价相关性（A股金工：负向）"),
    "obv20": ("量价配合", 1, "近 20 日 OBV 变化", "能量潮（技术分析教科书：正向）"),
    "upvol20": ("量价配合", 1, "近 20 日上涨日成交量占比", "上涨放量、下跌缩量（技术分析教科书：正向）"),
    # —— K 线形态：把蜡烛图量化。CNN 读 K 线图（Jiang-Kelly-Xiu 2023, JF）学到的主要也是收盘位置、短期反转这类信息 ——
    "clv": ("K线形态", -1, "昨日收盘在当日高低区间的位置", "收盘位置（Jiang-Kelly-Xiu 2023）"),
    "shadow20": ("K线形态", -1, "近 20 日平均上影线", "上影线 = 抛压（A股金工）"),
    "lshadow20": ("K线形态", 1, "近 20 日平均下影线", "下影线 = 支撑（技术分析教科书：正向）"),
    "overnight20": ("K线形态", -1, "近 20 日平均隔夜涨幅", "隔夜与日内的拉锯（Lou-Polk-Skouras 2019）；A股隔夜高开多为散户追涨"),
    "intraday20": ("K线形态", -1, "近 20 日平均日内涨幅", "日内收益反转"),
    "gap1": ("K线形态", -1, "昨日开盘跳空", "跳空高开回补"),
    "ideal_amp": ("K线形态", -1, "理想振幅：近 20 日里价格最高的 5 天平均振幅 − 最低的 5 天平均振幅", "高位振幅大 = 分歧出货（A股金工）"),
}
FAMILIES = ("反转", "动量趋势", "低风险", "换手流动性", "量价配合", "K线形态")
LIT = ("反转", "低风险", "换手流动性")  # A股文献里最站得住的三类
CHASE = ("rs250", "mom60", "high250", "ma_align", "macd_hist", "break20", "obv20", "upvol20")  # 追涨型技术分析


def _adj(p):
    ret = (p.close / p.preclose - 1).where(p.traded)
    return ret, (1 + ret.fillna(0)).cumprod().astype("float32")


def _ideal_amp(adj, amp, w=20, q=5, chunk=400):
    """近 w 日里复权收盘价最高的 q 天的平均振幅 − 最低的 q 天的平均振幅。窗口里有停牌日则为空。"""
    a, m = adj.to_numpy(dtype="float32"), amp.to_numpy(dtype="float32")
    out = np.full(a.shape, np.nan, dtype="float32")
    if len(a) < w:
        return pd.DataFrame(out, index=adj.index, columns=adj.columns)
    for c0 in range(0, a.shape[1], chunk):
        wa = np.lib.stride_tricks.sliding_window_view(a[:, c0:c0 + chunk], w, axis=0)
        wm = np.lib.stride_tricks.sliding_window_view(m[:, c0:c0 + chunk], w, axis=0)
        order = np.argsort(wa, axis=2)
        srt = np.take_along_axis(wm, order, axis=2)
        out[w - 1:, c0:c0 + chunk] = srt[:, :, -q:].mean(axis=2) - srt[:, :, :q].mean(axis=2)
    return pd.DataFrame(out, index=adj.index, columns=adj.columns)


def extra_factors(p):
    """LIB 里现有因子库（factors.iter_factors、alt_study.iter_alt_factors）没有的那几个。都只用 t 日收盘及以前的数据。"""
    ret, adj = _adj(p)
    turn = p.turn.where(p.turn > 0)
    f = {}
    f["mom120"] = adj.shift(20) / adj.shift(120) - 1
    f["ma200gap"] = adj / adj.rolling(200, min_periods=150).mean() - 1
    up_d = (ret > 0).astype("float32").where(ret.notna()).shift(20).rolling(230, min_periods=150).mean()
    dn_d = (ret < 0).astype("float32").where(ret.notna()).shift(20).rolling(230, min_periods=150).mean()
    f["fip"] = np.sign(adj.shift(20) / adj.shift(250) - 1) * (dn_d - up_d)
    del up_d, dn_d
    # 显著性（近似）：某天个股涨跌和全市场差得越远越“扎眼”，σ = |r − rm| / (|r| + |rm| + 0.1)；
    # 因子 = σ 加权的平均收益 − 简单平均收益。数值大 = 扎眼的日子多是大涨 → 被高估
    mkt = ret.mean(axis=1)
    sig = ret.sub(mkt, axis=0).abs().div(ret.abs().add(mkt.abs(), axis=0) + 0.1)
    f["salience20"] = ((sig * ret).rolling(20, min_periods=15).sum() / sig.rolling(20, min_periods=15).sum()
                       - ret.rolling(20, min_periods=15).mean())
    del sig
    f["abturn"] = turn.rolling(20, min_periods=15).mean() / turn.rolling(250, min_periods=120).mean()
    f["turn_std20"] = turn.rolling(20, min_periods=15).std()
    f["pv_corr20"] = adj.rolling(20, min_periods=15).corr(turn)
    vol = p.volume.where(p.traded)
    f["upvol20"] = (vol.where(ret > 0, 0.0).where(ret.notna()).rolling(20, min_periods=15).sum()
                    / vol.rolling(20, min_periods=15).sum())
    rng = (p.high - p.low).where(p.high > p.low)
    f["lshadow20"] = ((np.minimum(p.open, p.close) - p.low) / rng).rolling(20, min_periods=15).mean()
    f["ideal_amp"] = _ideal_amp(adj.where(p.traded), ((p.high - p.low) / p.preclose).where(p.traded))
    return {k: v.where(p.traded).astype("float32") for k, v in f.items()}


def all_ranks(p, elig, industry=None):
    """LIB 里每个因子的横截面排名（已乘事先定的方向，越大越好）。"""
    ranks = {}

    def take(name, v):
        if name in LIB and name not in ranks:
            ranks[name] = model.xs_rank(v, elig) * LIB[name][1]

    for k, v in iter_factors(p):
        take(k, v)
        del v
    for k, v in iter_alt_factors(p, None, industry):
        take(k, v)
        del v
    for k, v in extra_factors(p).items():
        take(k, v)
    return ranks


def composite(ranks, names, elig):
    names = [n for n in names if n in ranks]
    return model.score(ranks, {n: 1.0 / len(names) for n in names}, elig)


def family_scores(ranks, elig):
    return {fam: composite(ranks, [k for k, v in LIB.items() if v[0] == fam], elig) for fam in FAMILIES}


def mix(parts, elig):
    """几个得分等权合成：各自先排名再平均。"""
    rk = {str(i): model.xs_rank(s, elig) for i, s in enumerate(parts)}
    return model.score(rk, {k: 1.0 / len(rk) for k in rk}, elig)


def ic_table(ranks, target, elig, step=IC_HOLD):
    """每个因子（已乘事先方向）的逐日排名 IC：训练期、样本外的均值和 t 值（t 值每 step 天取一个样本，避免重叠高估）。"""
    t_rank = model.xs_rank(target, elig & target.notna())
    out = {}
    for name, r in ranks.items():
        x = r.where(t_rank.notna())
        t = t_rank.where(x.notna())
        x = x.sub(x.mean(axis=1), axis=0)
        t = t.sub(t.mean(axis=1), axis=0)
        ic = ((x * t).sum(axis=1) / np.sqrt((x ** 2).sum(axis=1) * (t ** 2).sum(axis=1))).where(x.notna().sum(axis=1) >= 30)
        row = {}
        for per, (s, e) in dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, None)).items():
            v = ic.loc[s:e].dropna()
            sub = v.iloc[::step]
            row[per] = dict(ic=float(v.mean()) if len(v) else None,
                            t=float(sub.mean() / sub.std() * np.sqrt(len(sub))) if len(sub) > 2 and sub.std() > 0 else 0.0)
        out[name] = row
    return out


def forecast_price(p, forecast, adj):
    """业绩预告和价格的关系。返回：
    since_pub  公告后到今天的涨幅（公告生效日前一天收盘为基准），公告后 FORECAST_DAYS 个交易日内有值；
    react      公告生效日当天的涨幅（市场对公告的第一反应），同样往后保留；
    events     (行号, 列号, 是否预喜) —— 生效日 = 公告日之后的第一个交易日，当天收盘后才用。"""
    nan = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype="float32")
    if forecast is None or forecast.empty:
        return nan, nan.copy(), (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=bool))
    f = forecast[forecast["code"].isin(p.codes)].sort_values("pub")
    pos = p.dates.searchsorted(f["pub"].values, side="right")
    keep = (pos < len(p.dates)) & (pos >= 1)
    f, pos = f[keep], pos[keep]
    col = p.codes.get_indexer(f["code"])
    a = adj.to_numpy()
    ref = np.full(a.shape, np.nan, dtype="float32")
    ref[pos, col] = a[pos - 1, col]
    ref = pd.DataFrame(ref, index=p.dates, columns=p.codes).ffill(limit=FORECAST_DAYS)
    rc = np.full(a.shape, np.nan, dtype="float32")
    rc[pos, col] = a[pos, col] / a[pos - 1, col] - 1
    react = pd.DataFrame(rc, index=p.dates, columns=p.codes).ffill(limit=FORECAST_DAYS)
    return (adj / ref - 1).astype("float32"), react, (pos, col, f["type"].isin(GOOD_TYPES).to_numpy())


def event_study(p, elig, ohl, events, high250, rev20, react):
    """业绩预喜公告（生效日收盘后可知）之后：次日开盘买、持有 10 天的收益减去同期全市场可买股票的平均，
    按公告时的价格位置、公告前 20 日涨幅、公告当天的市场反应分组。描述统计，不是交易规则。"""
    o = pd.DataFrame(ohl[0], index=p.dates, columns=p.codes)
    fwd = (o.shift(-11) / o.shift(-1) - 1).where(elig)
    ex = fwd.sub(fwd.mean(axis=1), axis=0).to_numpy()
    pos, col, good = events
    ok = good & elig.to_numpy()[pos, col] if len(pos) else good
    pos, col = pos[ok], col[ok]
    e = ex[pos, col]
    hi, rv, rc = high250.to_numpy()[pos, col], rev20.to_numpy()[pos, col], react.to_numpy()[pos, col]
    groups = {
        "全部预喜公告": np.ones(len(e), dtype=bool),
        "公告时离一年高点 5% 以内（高位）": hi >= 0.95,
        "离一年高点 5%–20%": (hi < 0.95) & (hi >= 0.8),
        "离一年高点 20% 以上（低位）": hi < 0.8,
        "公告前 20 日已涨 15% 以上": rv > 0.15,
        "公告前 20 日涨 0–15%": (rv >= 0) & (rv <= 0.15),
        "公告前 20 日是跌的": rv < 0,
        "公告当天股价涨（价格确认）": rc > 0,
        "公告当天股价跌（见光死）": rc <= 0,
        "高位 + 公告当天跌": (hi >= 0.95) & (rc <= 0),
        "低位 + 公告当天涨": (hi < 0.8) & (rc > 0),
    }
    d = p.dates[pos]
    out = []
    for per, (s, t) in dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, "2100-01-01")).items():
        in_per = (d >= pd.Timestamp(s)) & (d <= pd.Timestamp(t)) & np.isfinite(e)
        for name, m in groups.items():
            v = e[in_per & m]
            out.append(dict(period=per, group=name, n=int(len(v)), excess10=float(v.mean()) if len(v) else None,
                            win=float((v > 0).mean()) if len(v) else None,
                            t=float(v.mean() / v.std() * np.sqrt(len(v))) if len(v) > 2 and v.std() > 0 else None))
    return out


def total_returns(frame, name, sleeve, share):
    """线上总组合口径的逐日收益：选股 share + ETF 分仓闲钱停泊（redline.simulate，不加红线）。"""
    old = redline.SHARE
    redline.SHARE = share
    try:
        parts = redline.load_part(name, frame=frame)
        return {per: redline.simulate(parts[per], sleeve, None)[0] for per in ("train", "test")}
    finally:
        redline.SHARE = old


def _diff_t(a, b):
    d = (a - b).dropna()
    return float(d.mean() / d.std() * np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else None


def run(kline, basic, valuation=None, forecast=None, industry=None, dump=None, etf=True):
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
    # 线上得分（业绩预期）
    lr = {}
    for k, v in iter_alt_factors(p, forecast):
        if k in weights:
            lr[k] = model.xs_rank(v, elig)
        del v
    base = apply_blend(model.score(lr, weights, elig), p, forecast, elig, sm.get("blend"))
    del lr
    _, sgn, _ = forecast_frames(p, forecast)
    # 纯 K 线因子
    ranks = all_ranks(p, elig, industry)
    log.info("factors: %d", len(ranks))
    size_rank = None
    for k, v in iter_factors(p):
        if k == "size":
            size_rank = model.xs_rank(v, elig)
        del v
    outc = trade_outcomes(p)
    target = outc[f"hold{IC_HOLD}_net"]
    del outc
    ics = ic_table(ranks, target, elig)
    del target
    fam = family_scores(ranks, elig)
    ret, adj = _adj(p)
    ma20 = adj.rolling(20, min_periods=16).mean()
    ma60 = adj.rolling(60, min_periods=48).mean()
    uptrend = (ma20 > ma60) & (ma60 > ma60.shift(10)) & (adj > ma60)
    falling = (adj < ma20) & (ma20 < ma20.shift(5))
    high250 = adj / adj.rolling(250, min_periods=120).max()
    rev20 = adj / adj.shift(20) - 1
    hot = model.xs_rank(rev20, elig) > 0.3  # 近 20 日涨幅排全市场前 20%
    since_pub, react, events = forecast_price(p, forecast, adj)

    pure = {f"fam_{f}": (f"{f}类因子等权", s) for f, s in fam.items()}
    pure["lit"] = ("A股文献组合：反转 + 低风险 + 低换手（三类等权）", mix([fam[f] for f in LIT], elig))
    pure["all6"] = ("六类因子等权", mix([fam[f] for f in FAMILIES], elig))
    pure["chase"] = ("追涨型技术分析：动量 + 新高 + 均线多头 + 突破 + 量价齐升", composite(ranks, CHASE, elig))
    # 训练期拟合：只用训练期的 IC
    sig = {k: v["train"]["t"] for k, v in ics.items()}
    w_ic = {k: (v["train"]["ic"] if abs(v["train"]["t"]) >= 2 else 0.0) for k, v in ics.items()}
    tot = sum(abs(x) for x in w_ic.values()) or 1.0
    w_ic = {k: v / tot for k, v in w_ic.items()}
    if any(w_ic.values()):
        pure["ic_all"] = ("训练期 IC 加权（全部量价因子，方向和权重都由训练期数据定）", model.score(ranks, w_ic, elig))
    top8 = sorted(sig, key=lambda k: abs(sig[k]), reverse=True)[:8]
    pure["top8"] = ("训练期最显著的 8 个因子等权（方向按训练期）",
                    model.score(ranks, {k: float(np.sign(sig[k]) or 1.0) / 8 for k in top8}, elig))
    pure["dip"] = ("趋势中的回调：均线向上的股票里买跌得多的", fam["反转"].where(uptrend))
    strong = ranks["rs250"] > 0.2  # 一年涨幅前 30%
    pure["strong_lowrisk"] = ("强势股里挑低风险：一年涨幅前 30% 里按低风险排", fam["低风险"].where(strong))
    pure["lit_uptrend"] = ("A股文献组合，只在均线向上的股票里选", pure["lit"][1].where(uptrend))

    gate = emotion_regime(p, elig)
    gate_on = pd.Series(True, index=p.dates)
    ohl = adjusted_ohlc(p)
    end = str(p.dates[-1].date())
    periods = dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, end))
    sleeve = None
    if etf:
        sleeve = dv.sleeve_returns(dv.etf_returns(), sm["sleeve"]["codes"], sm["sleeve"].get("rebalance", 20))
    cols, rows, specs, series = {}, [], {}, {}

    def sim(key, sc, exits=None, g=None, size=True):
        e_tp, e_sl, e_hold = exits or (tp, sl, hold)
        res, parts = {}, []
        for k, (s, e) in periods.items():
            d, t = simulate(p, ohl, sc, gate if g is None else g, n, e_tp, e_sl, e_hold, s, e)
            res[k] = _summ(d, t)
            res[k]["coverage"] = float(sc.loc[s:e].notna().sum(axis=1).mean())
            held = pd.Series(0, index=d.index)
            for x in t:
                held[(held.index >= x["entry"]) & (held.index < x["exit"])] += 1
            res[k]["avg_held"] = float(held.mean()) if len(held) else 0.0
            if size_rank is not None and size:
                top = sc.loc[s:e].rank(axis=1, ascending=False) <= n
                res[k]["size_pct"] = float(np.nanmean(size_rank.loc[s:e].where(top).to_numpy()) + 0.5) if top.to_numpy().any() else None
            parts.append(pd.DataFrame({key + "|raw": d, key + "|held": held}))
        frame = pd.concat(parts)
        tot_r = None
        if sleeve is not None:
            tot_r = total_returns(frame, key, sleeve, share)
            res["total"] = {}
            for per, r in tot_r.items():
                st = redline.stats(r)
                res["total"][per] = {k: st.get(k) for k in ("ann_return", "max_drawdown", "sharpe")}
                res["total"][per]["yearly"] = {str(y): float((1 + g_).prod() - 1) for y, g_ in r.groupby(r.index.year)}
        return res, frame, tot_r

    def add(name, group, sc, key, exits=None, g=None, keep=True, **extra):
        res, frame, tot_r = sim(key, sc, exits, g, size=keep)
        row = dict(name=name, group=group, key=key, **extra, **res)
        if tot_r is not None:
            series[key] = tot_r
            if "live" in series and key != "live":
                row["t_vs_live"] = {per: _diff_t(tot_r[per], series["live"][per]) for per in tot_r}
        rows.append(row)
        if keep:
            specs[key] = (name, group, sc, exits, g)
            if dump is not None:
                for c in frame:
                    cols[c] = frame[c]
                if tot_r is not None:
                    cols[key + "|total"] = pd.concat([tot_r["train"], tot_r["test"]])
        elif tot_r is not None:
            del series[key]
        log.info("%s: train %.2f/%s test %.2f/%s", key, row["train"]["sharpe"] or 0, row["train"]["ann_return"],
                 row["test"]["sharpe"] or 0, row["test"]["ann_return"])
        return row

    def tkey(r):  # 只按训练期排：有总组合用总组合夏普，否则用选股层夏普
        return (r["total"]["train"] if "total" in r else r["train"])["sharpe"] or -9

    add("线上规则：业绩预期（对照）", "live", base, "live")
    # —— 纯 K 线 ——
    for key, (name, sc) in pure.items():
        add(name, "pure", sc, "k_" + key)
    for key, (name, sc) in pure.items():
        add(name + "｜不止盈、最长 20 天", "pure_long", sc, "k_" + key + "_h20", exits=LONG_EXIT)
    pure_rows = [r for r in rows if r["group"] in ("pure", "pure_long")]
    best_pure = max(pure_rows, key=tkey)
    for r in sorted(pure_rows, key=tkey, reverse=True)[:3]:
        nm, _, sc, exits, _ = specs[r["key"]]
        add(nm + "｜不用情绪开关", "pure_nogate", sc, r["key"] + "_nogate", exits=exits, g=gate_on)
    best_key = best_pure["key"]
    kbest = specs[best_key][2]
    kbest_rank = model.xs_rank(kbest, elig)
    lit_rank = model.xs_rank(pure["lit"][1], elig)
    # —— 随机选股 ——
    rng = np.random.default_rng(1118)
    for i in range(N_RANDOM):
        noise = pd.DataFrame(rng.random(base.shape, dtype=np.float32), index=base.index, columns=base.columns)
        add(f"全市场随机选 {i + 1}", "random_all", noise.where(elig), f"rnd_all{i + 1}", keep=False)
        add(f"有业绩预告的股票里随机选 {i + 1}", "random_fc", noise.where(base.notna()), f"rnd_fc{i + 1}", keep=False)
        if i < N_RANDOM_MIX:
            add(f"线上得分随机保留 60% {i + 1}", "random_filter", base.where(noise < 0.6), f"rnd_f{i + 1}", keep=False)
            add(f"线上得分 + 0.3 × 随机因子 {i + 1}", "random_blend", (base + 0.3 * (noise - 0.5)).where(base.notna()),
                f"rnd_b{i + 1}", keep=False)
        del noise
    # —— 折中：业绩预期 + 价格 ——
    add("线上得分，已经在跌的不买（收盘在 20 日线下且 20 日线向下）", "hyb_filter", base.where(~falling), "h_notfall")
    add("线上得分，只买收盘在 60 日线上方的", "hyb_filter", base.where(adj > ma60), "h_ma60")
    add("线上得分，只买均线向上的（20 日线在 60 日线上、60 日线上行）", "hyb_filter", base.where(uptrend), "h_uptrend")
    add("线上得分，近 20 日涨幅排全市场前 20% 的不买（不追高）", "hyb_filter", base.where(~hot), "h_nothot")
    add("线上得分，离一年高点 5% 以内的不买（高位不接）", "hyb_filter", base.where(~(high250 >= 0.95)), "h_nothigh")
    add("线上得分，公告后已经涨了 15% 以上的不买", "hyb_filter", base.where(~(since_pub > 0.15)), "h_notrun")
    add("线上得分，公告后股价是涨的才买（价格确认）", "hyb_filter", base.where(since_pub > 0), "h_confirm")
    add("线上得分，公告当天股价涨的才买", "hyb_filter", base.where(react > 0), "h_react")
    add("线上得分，公告当天股价跌的才买（对照）", "hyb_filter", base.where(react <= 0), "h_react_neg")
    add("线上得分，只买 K 线得分（A股文献组合）在全市场前一半的", "hyb_filter", base.where(lit_rank > 0), "h_lit_half")
    add("线上得分，只买 K 线得分（训练期最好的纯 K 线方案）在全市场前一半的", "hyb_filter", base.where(kbest_rank > 0),
        "h_best_half")
    for w in BLEND_W:
        add(f"线上得分 + {w} × K 线得分（A股文献组合）", "hyb_blend", (base + w * lit_rank.fillna(0)).where(base.notna()),
            f"b_lit_{w}", w=w)
        add(f"线上得分 + {w} × K 线得分（训练期最好的纯 K 线方案）", "hyb_blend",
            (base + w * kbest_rank.fillna(0)).where(base.notna()), f"b_best_{w}", w=w)
    for tag, nm, sc in (("lit", "A股文献组合", pure["lit"][1]), ("best", "训练期最好的纯 K 线方案", kbest)):
        ex = specs[best_key][3] if tag == "best" else None
        add(f"K 线说了算（{nm}），业绩预告只排雷：近期预减/首亏/续亏的不买", "k_main", sc.where(~(sgn < 0)), f"m_{tag}_veto",
            exits=ex)
        add(f"K 线说了算（{nm}），只在近期业绩预喜的股票里选", "k_main", sc.where(sgn > 0), f"m_{tag}_good", exits=ex)
        add(f"K 线说了算（{nm}），只在线上候选（有业绩预告或盈利变化）里选", "k_main", sc.where(base.notna()),
            f"m_{tag}_cand", exits=ex)
    # —— 保守成本（滑点 0.2%/边）：线上 + 训练期最好的纯 K 线 + A股文献组合 + 训练期最好的折中方案 ——
    hyb_rows = [r for r in rows if r["group"] in ("hyb_filter", "hyb_blend", "k_main")]
    best_hyb = max(hyb_rows, key=tkey)
    cons = []
    slip0 = config.SLIPPAGE
    config.SLIPPAGE = CONS_SLIPPAGE
    try:
        for key in dict.fromkeys(["live", best_key, "k_lit", best_hyb["key"]]):
            nm, group, sc, exits, g = specs[key]
            res, _, _ = sim(key + "_cons", sc, exits, g)
            cons.append(dict(name=nm, group=group, key=key, **res))
    finally:
        config.SLIPPAGE = slip0
    ev = event_study(p, elig, ohl, events, high250, rev20, react)
    if dump is not None:
        out = pd.DataFrame(cols)
        out.index.name = "date"
        out.to_csv(dump, float_format="%.6g")
    factors = [dict(key=k, family=LIB[k][0], direction=LIB[k][1], desc=LIB[k][2], source=LIB[k][3], **ics[k])
               for k in LIB if k in ics]
    return dict(updated=end, capital_share=share, rows=rows, cons=cons, factors=factors, events=ev,
                best_pure=best_key, best_hybrid=best_hyb["key"], ic_weights={k: v for k, v in w_ic.items() if v},
                top8={k: float(np.sign(sig[k])) for k in top8})


def _p(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def _f(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


GROUPS = {"live": "对照", "pure": "纯 K 线", "pure_long": "纯 K 线（不止盈、最长 20 天）", "pure_nogate": "纯 K 线（不用情绪开关）",
          "hyb_filter": "业绩预期 + 价格过滤", "hyb_blend": "业绩预期 + K 线加分", "k_main": "K 线为主、业绩预告为辅"}


def _tot_cells(r):
    t = r.get("total")
    if not t:
        return "— | — | — | — | — | — | —"
    a, b = t["train"], t["test"]
    tv = (r.get("t_vs_live") or {}).get("test")
    return (f"{_p(a['ann_return'])} | {_p(a['max_drawdown'])} | {_f(a['sharpe'])} | "
            f"{_p(b['ann_return'])} | {_p(b['max_drawdown'])} | {_f(b['sharpe'])} | {_f(tv)}")


def _band(rows, group, per, key="ann_return", total=False):
    v = np.array([((r["total"][per] if total and "total" in r else r[per])[key] or 0) for r in rows if r["group"] == group])
    if not len(v):
        return "—"
    fmt = _p if key != "sharpe" else _f
    return f"{fmt(float(np.quantile(v, 0.1)))} ~ {fmt(float(np.quantile(v, 0.9)))}（中位 {fmt(float(np.median(v)))}）"


def to_markdown(res):
    rows = res["rows"]
    live = rows[0]
    has_total = "total" in live
    by_key = {r["key"]: r for r in rows}
    L = [f"# 纯 K 线（量价）选股研究（数据到 {res['updated']}）", "",
         "只用开高低收、成交量、换手率的因子选股，套进和线上完全一样的框架（情绪开关、20 只、止盈 2 倍 ATR、止损 3 倍 ATR、"
         f"最长 10 天、净值跌破 20 日均线空仓；总组合 = 选股 {res['capital_share'] * 100:.0f}% + ETF 分仓闲钱停泊），"
         "和线上规则（业绩预期）同口径比。训练期 2015–2020，样本外 2021 至今，已扣成本，回测用 2026-11-16 修正后的口径。", "",
         f"训练期最好的纯 K 线方案：**{by_key[res['best_pure']]['name']}**；训练期最好的折中方案：**{by_key[res['best_hybrid']]['name']}**。", ""]
    L += ["## 随机选股参照（“选股”这一步到底值多少）", "",
          "| 参照 | 训练年化（10–90 分位） | 样本外年化（10–90 分位） | 样本外夏普 |", "|---|---|---|---|"]
    for g, nm in (("random_all", "全市场随机选 20 只"), ("random_fc", "有业绩预告的股票里随机选"),
                  ("random_filter", "线上得分随机保留 60%"), ("random_blend", "线上得分 + 0.3 × 随机因子")):
        L.append(f"| {nm}（{'总组合' if has_total else '选股层'}） | {_band(rows, g, 'train', total=True)} | "
                 f"{_band(rows, g, 'test', total=True)} | {_band(rows, g, 'test', 'sharpe', total=True)} |")
        L.append(f"| {nm}（选股层） | {_band(rows, g, 'train')} | {_band(rows, g, 'test')} | {_band(rows, g, 'test', 'sharpe')} |")
    L += ["", f"## 总组合（选股 {res['capital_share'] * 100:.0f}% + ETF 分仓闲钱停泊，线上口径）", "",
          "| 类别 | 方案 | 训练年化 | 训练回撤 | 训练夏普 | 样本外年化 | 样本外回撤 | 样本外夏普 | 样本外相对线上的 t 值 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["group"] in GROUPS:
            L.append(f"| {GROUPS[r['group']]} | {r['name']} | {_tot_cells(r)} |")
    L += ["", "## 选股层（100% 资金口径，含 20 日均线空仓）", "",
          "| 类别 | 方案 | 训练年化 | 训练回撤 | 训练夏普 | 训练胜率 | 样本外年化 | 样本外回撤 | 样本外夏普 | 样本外胜率 | 样本外笔数 | 候选数/天 | 平均持仓 | 入选股市值分位 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["group"] not in GROUPS:
            continue
        a, b = r["train"], r["test"]
        L.append(f"| {GROUPS[r['group']]} | {r['name']} | {_p(a['ann_return'])} | {_p(a['max_drawdown'])} | {_f(a['sharpe'])} | "
                 f"{_p(a['trade_win'])} | {_p(b['ann_return'])} | {_p(b['max_drawdown'])} | {_f(b['sharpe'])} | "
                 f"{_p(b['trade_win'])} | {b['trades']} | {b['coverage']:.0f} | {b['avg_held']:.1f} | {_p(b.get('size_pct'), 0)} |")
    if has_total:
        years = sorted(live["total"]["test"]["yearly"])
        L += ["", "## 样本外逐年（总组合）", "", "| 方案 | " + " | ".join(years) + " |", "|---|" + "---|" * len(years)]
        for key in dict.fromkeys(["live", res["best_pure"], "k_lit", "k_all6", "k_chase", res["best_hybrid"]]):
            r = by_key.get(key)
            if r:
                L.append(f"| {r['name']} | " + " | ".join(_p(r["total"]["test"]["yearly"].get(y)) for y in years) + " |")
    L += ["", "## 保守成本（滑点 0.2%/边）", "",
          "| 方案 | 选股层训练年化 | 选股层样本外年化 | 样本外回撤 | 总组合训练年化 | 训练回撤 | 训练夏普 | 总组合样本外年化 | 样本外回撤 | 样本外夏普 | — |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["cons"]:
        L.append(f"| {r['name']} | {_p(r['train']['ann_return'])} | {_p(r['test']['ann_return'])} | "
                 f"{_p(r['test']['max_drawdown'])} | {_tot_cells(r)} |")
    L += ["", f"## 单因子检验（排名 IC，标签 = 次日开盘买、持有 {IC_HOLD} 天的扣费收益；方向按文献事先定，IC 为正 = 和文献方向一致）", "",
          "| 家族 | 因子 | 事先方向 | 训练 IC | 训练 t | 样本外 IC | 样本外 t | 出处 |", "|---|---|---|---|---|---|---|---|"]
    for f in res["factors"]:
        L.append(f"| {f['family']} | {f['desc']} | {'越大越好' if f['direction'] > 0 else '越小越好'} | {_p(f['train']['ic'], 2)} | "
                 f"{_f(f['train']['t'], 1)} | {_p(f['test']['ic'], 2)} | {_f(f['test']['t'], 1)} | {f['source']} |")
    L += ["", "## 业绩预喜公告之后的表现，按公告时的价格位置分组（描述统计）", "",
          "次日开盘买、持有 10 天的收益减去同期全市场平均。", "",
          "| 时期 | 分组 | 次数 | 10 天平均超额 | 跑赢市场的比例 | t 值 |", "|---|---|---|---|---|---|"]
    for x in res["events"]:
        L.append(f"| {'训练' if x['period'] == 'train' else '样本外'} | {x['group']} | {x['n']} | {_p(x['excess10'], 2)} | "
                 f"{_p(x['win'])} | {_f(x['t'], 1)} |")
    L += ["", "训练期 IC 加权用到的因子和权重（正 = 和文献方向一致）：" +
          "，".join(f"{LIB[k][2]} {v:+.2f}" for k, v in sorted(res["ic_weights"].items(), key=lambda kv: -abs(kv[1]))), "",
          "训练期最显著的 8 个：" + "，".join(f"{LIB[k][2]}（{'同' if v > 0 else '反'}文献方向）" for k, v in res["top8"].items())]
    return "\n".join(L) + "\n"


def write(res, stem="kline_study"):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / f"{stem}.md").write_text(to_markdown(res))
    (config.PAGE_DIR / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
