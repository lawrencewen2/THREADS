"""选股面板的决策辅助板块：市场温度、行业强弱、个股体检、策略可信度、决策清单。

思路借鉴自几个开源金融项目：
- 市场温度（涨跌家数、涨跌停、成交额、站上 20 日线比例）：daily_stock_analysis 的“大盘复盘”
- 行业强弱：TradingAgents-CN、a-stock-data 的板块轮动数据
- 个股体检（估值自身分位、行业内估值分位、风险提示）：ai-berkshire 的 quality-screen / investment-checklist，
  以及 Anthropic financial-services 的可比公司分析
- 策略可信度（样本外收益的自助法置信区间）：Vibe-Trading 的回测统计检验

只读行情缓存和选股结果，不改模型、回测和选股逻辑。所有数字都只用信号日 t 收盘及以前的数据。
"""

import json
import logging

import numpy as np
import pandas as pd

from . import config, model
from .factors import Panel

log = logging.getLogger(__name__)

INDUSTRY_FILE = config.DATA_DIR / "industry.csv"
OUT_FILE = config.PAGE_DIR / "insight.json"
TEMP_DAYS = 60  # 面板上画的市场温度天数
TEMP_RANK_WINDOW = 250  # 温度分：各指标在近 250 个交易日里的分位
MIN_INDUSTRY_SIZE = 10  # 行业内股票少于这个数不列
VAL_HISTORY_DAYS = 750  # 估值自身分位用近 3 年
BOOT_SAMPLES = 5000
BOOT_BLOCK = 4  # 周收益按 4 周一块重抽，保留短期自相关


# ---------- 市场温度 ----------

def market_temperature(p):
    """每天的涨跌家数、涨跌停家数、两市成交额、站上 20 日线比例，以及合成的温度分（0 冷 100 热）。"""
    traded = p.traded
    ret = (p.close / p.preclose - 1).where(traded)
    pct = p.limit_pct().where(~p.is_st, 0.05)
    up_px = (p.preclose * (1 + pct) + 1e-6).round(2)
    dn_px = (p.preclose * (1 - pct) + 1e-6).round(2)
    ma20 = p.close.rolling(20, min_periods=15).mean()
    df = pd.DataFrame({
        "up": (ret > 0).sum(axis=1),
        "down": (ret < 0).sum(axis=1),
        "limit_up": ((p.close >= up_px - 0.005) & traded).sum(axis=1),
        "limit_down": ((p.close <= dn_px + 0.005) & traded).sum(axis=1),
        "amount": p.amount.sum(axis=1) / 1e8,  # 亿元
        "above_ma20": (p.close > ma20).sum(axis=1) / ma20.notna().sum(axis=1).replace(0, np.nan),
        "median_ret": ret.median(axis=1),
    })
    df["up_share"] = df["up"] / (df["up"] + df["down"]).replace(0, np.nan)
    df["net_limit"] = df["limit_up"] - df["limit_down"]
    df["amount_ratio"] = df["amount"] / df["amount"].rolling(20, min_periods=10).mean()

    def pr(s):
        return s.rolling(TEMP_RANK_WINDOW, min_periods=60).rank(pct=True)

    parts = [pr(df["up_share"].rolling(5, min_periods=3).mean()), pr(df["above_ma20"]),
             pr(df["net_limit"].rolling(5, min_periods=3).mean()), pr(df["amount_ratio"])]
    df["temperature"] = (pd.concat(parts, axis=1).mean(axis=1) * 100)
    return df


def temperature_label(v):
    if v is None or np.isnan(v):
        return "未知"
    for upper, label in ((20, "冰点"), (40, "偏冷"), (60, "正常"), (80, "偏热"), (101, "过热")):
        if v < upper:
            return label
    return "过热"


# ---------- 行业 ----------

def clean_industry(df):
    """baostock 的证监会行业名带字母数字前缀（如 C39计算机、通信…），去掉前缀。"""
    df = df[["code", "industry"]].copy()
    df["industry"] = df["industry"].fillna("").str.replace(r"^[A-Z]\d*", "", regex=True).str.strip()
    return df[df["industry"] != ""]


def load_industry():
    if not INDUSTRY_FILE.exists():
        return None
    return pd.read_csv(INDUSTRY_FILE, dtype=str)


def sector_strength(p, industry, ranking):
    """各行业近 5 日、20 日涨幅中位数，站上 20 日线比例，成交额放大倍数，以及今日选股得分前 10% 的股票数。"""
    t = p.dates[-1]
    ret = (p.close / p.preclose - 1).where(p.traded).fillna(0)
    adj = (1 + ret).cumprod()
    last = p.traded.loc[t]
    r5 = (adj.loc[t] / adj.iloc[-6] - 1) if len(adj) > 5 else pd.Series(np.nan, index=p.codes)
    r20 = (adj.loc[t] / adj.iloc[-21] - 1) if len(adj) > 20 else pd.Series(np.nan, index=p.codes)
    ma20 = p.close.rolling(20, min_periods=15).mean().loc[t]
    above = (p.close.loc[t] > ma20).where(ma20.notna())
    amt5 = p.amount.iloc[-5:].mean()
    amt60 = p.amount.iloc[-60:].mean()
    df = pd.DataFrame({"r5": r5, "r20": r20, "above": above.astype(float), "amt5": amt5, "amt60": amt60})
    df = df[last.reindex(df.index).fillna(False).astype(bool)]
    ind = industry.set_index("code")["industry"]
    df["industry"] = ind.reindex(df.index)
    df = df.dropna(subset=["industry"])
    top = set()
    if ranking is not None and len(ranking):
        top = set(ranking.loc[ranking["pct"] >= 0.9, "code"])
    df["top"] = df.index.isin(top)
    g = df.groupby("industry")
    out = pd.DataFrame({
        "stocks": g.size(),
        "ret5": g["r5"].median(),
        "ret20": g["r20"].median(),
        "above_ma20": g["above"].mean(),
        "amount_ratio": g["amt5"].sum() / g["amt60"].sum(),
        "top_picks": g["top"].sum(),
    })
    out = out[out["stocks"] >= MIN_INDUSTRY_SIZE]
    # 强弱分：三项在行业间的分位平均，0 最弱 100 最强
    out["strength"] = out[["ret5", "ret20", "above_ma20"]].rank(pct=True).mean(axis=1) * 100
    out["top_share"] = out["top_picks"] / out["stocks"]
    return out.sort_values("strength", ascending=False)


# ---------- 个股体检 ----------

def checkups(p, industry, ranking, names):
    """每只可选股票：行业、估值（市盈率、市净率）、估值在自身近 3 年和行业内的分位、近期走势与风险提示。"""
    t = p.dates[-1]
    codes = list(ranking["code"]) if ranking is not None else list(p.codes[p.traded.loc[t]])
    codes = [c for c in codes if c in p.codes]
    ret = (p.close / p.preclose - 1).where(p.traded).fillna(0)
    adj = (1 + ret).cumprod()
    n = len(p.dates)
    r20 = adj.loc[t] / adj.iloc[max(0, n - 21)] - 1
    r60 = adj.loc[t] / adj.iloc[max(0, n - 61)] - 1
    hi250 = adj.iloc[-250:].max()
    lo250 = adj.iloc[-250:].min()
    amt20 = p.amount.iloc[-20:].mean() / 1e8
    up_px, dn_px = p.limit_prices()
    dn_hit = (p.close <= dn_px + 0.005) & p.traded
    dn3 = dn_hit.iloc[-3:].sum()
    st = p.is_st.loc[t]
    df = pd.DataFrame(index=pd.Index(codes, name="code"))
    df["ret20"], df["ret60"] = r20.reindex(codes), r60.reindex(codes)
    df["from_high"] = (adj.loc[t] / hi250 - 1).reindex(codes)
    df["from_low"] = (adj.loc[t] / lo250 - 1).reindex(codes)
    df["amount20"] = amt20.reindex(codes)
    df["limit_down3"] = dn3.reindex(codes)
    df["st"] = st.reindex(codes).fillna(False)
    ind = industry.set_index("code")["industry"] if industry is not None else pd.Series(dtype=str)
    df["industry"] = ind.reindex(codes).fillna("")
    for col, key in (("peTTM", "pe"), ("pbMRQ", "pb")):
        w = p.valuation.get(col)
        if w is None:
            df[key] = np.nan
            df[key + "_self_pct"] = np.nan
            df[key + "_ind_pct"] = np.nan
            continue
        hist = w.iloc[-VAL_HISTORY_DAYS:]
        cur = hist.ffill(limit=5).loc[t]
        df[key] = cur.reindex(codes)
        # 自身分位：当前值在近 3 年里的位置（只看正值，亏损时市盈率为负没有可比性）
        pos = hist.where(hist > 0)
        self_pct = (pos.le(cur, axis=1).sum() / pos.notna().sum().replace(0, np.nan)).where(cur > 0)
        df[key + "_self_pct"] = self_pct.reindex(codes)
        # 行业内分位：同行业正值里排第几（越低越便宜）
        tmp = pd.DataFrame({"v": cur.where(cur > 0), "ind": ind.reindex(cur.index)}).dropna()
        df[key + "_ind_pct"] = tmp.groupby("ind")["v"].rank(pct=True).reindex(codes)
    if ranking is not None:
        r = ranking.set_index("code")
        df["rank"], df["score_pct"] = r["rank"].reindex(codes), r["pct"].reindex(codes)
    df["name"] = [names.get(c, "") for c in codes]
    df["flags"] = [";".join(risk_flags(row)) for _, row in df.iterrows()]
    return df


def risk_flags(r):
    f = []
    if r.get("st"):
        f.append("ST")
    if pd.notna(r.get("pe")) and r["pe"] <= 0:
        f.append("亏损（市盈率为负）")
    if pd.notna(r.get("pe_ind_pct")) and r["pe_ind_pct"] >= 0.9:
        f.append("市盈率在行业里最贵的 10%")
    if pd.notna(r.get("pe_self_pct")) and r["pe_self_pct"] >= 0.9:
        f.append("市盈率处于自身近3年高位")
    if pd.notna(r.get("ret20")) and r["ret20"] >= 0.5:
        f.append("近20日涨超50%")
    if pd.notna(r.get("ret20")) and r["ret20"] <= -0.3:
        f.append("近20日跌超30%")
    if pd.notna(r.get("limit_down3")) and r["limit_down3"] >= 2:
        f.append("近3日两次以上跌停")
    if pd.notna(r.get("amount20")) and r["amount20"] < 1:
        f.append("日均成交额不足1亿")
    return f


# ---------- 策略可信度 ----------

def _weekly_returns(curve, start):
    s = pd.Series(curve, dtype=float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s[s.index >= pd.Timestamp(start)]
    return s.pct_change().dropna().to_numpy()


def bootstrap(rets, hold=None, samples=BOOT_SAMPLES, block=BOOT_BLOCK, seed=0):
    """按块重抽周收益，估计年化收益的 90% 区间、年化为正的概率，以及跑赢一直持有的概率（同一组周一起抽）。"""
    n = len(rets)
    if n < block * 4:
        return None
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(samples, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(samples, -1)[:, :n]

    def ann(r):
        return np.exp(np.log1p(r).sum(axis=1) * 52 / n) - 1

    a = ann(rets[idx])
    actual = float(np.exp(np.log1p(rets).sum() * 52 / n) - 1)
    out = dict(weeks=n, ann=actual, ci_low=float(np.quantile(a, 0.05)), ci_high=float(np.quantile(a, 0.95)),
               p_positive=float((a > 0).mean()))
    if hold is not None and len(hold) == n:
        out["p_beat_hold"] = float((a > ann(hold[idx])).mean())
    return out


def confidence(bt):
    """对回测里样本外的几条净值曲线做自助法检验。"""
    if not bt:
        return []
    start = bt.get("test_start", config.TEST_START)
    rows = []
    eq = bt.get("equity") or []
    if eq:
        e = pd.DataFrame(eq).set_index("date")
        rows.append(("隔夜选股（T 开盘买、T+1 开盘卖）", e["strategy"].to_dict(), e["baseline"].to_dict(), "全部可买股票"))
    ct = (bt.get("contrarian") or {}).get("equity") or {}
    if ct.get("always"):
        if ct.get("strategy"):
            rows.append(("恐慌买贪婪卖（选股）", ct["strategy"], ct["always"], "一直持有全市场"))
        if ct.get("timing"):
            rows.append(("恐慌买贪婪卖（全市场择时）", ct["timing"], ct["always"], "一直持有全市场"))
        rows.append(("一直持有全市场", ct["always"], None, None))
    out = []
    for name, curve, base, base_name in rows:
        r = _weekly_returns(curve, start)
        h = _weekly_returns(base, start) if base else None
        if h is not None and len(h) != len(r):
            h = None
        b = bootstrap(r, h)
        if b:
            out.append(dict(strategy=name, benchmark=base_name, **{k: round(v, 4) if isinstance(v, float) else v
                                                                   for k, v in b.items()}))
    return out


# ---------- 决策清单 ----------

def contrarian_state(bt, fear):
    ch = ((bt or {}).get("contrarian") or {}).get("chosen")
    if not ch or fear is None or fear.dropna().empty:
        return None
    reg = model.contrarian_regime(fear.dropna(), ch["buy"], ch["sell"])
    return dict(buy=ch["buy"], sell=ch["sell"], in_market=bool(np.asarray(reg)[-1]))


def checklist(picks, temp_row, sectors, conf, cstate, top_checks, bt):
    """把各板块的结论压成几条“可以做 / 小心 / 不要做”。只是规则汇总，不是投资建议。"""
    items = []
    fear = picks.get("fear_cn") if picks else None
    if fear is not None:
        if fear <= 20:
            s, d = "好", f"A股恐慌贪婪指数 {fear:.0f}，处于极度恐慌，按“恐慌时买”的原则是分批买入的区间"
        elif fear >= 70:
            s, d = "差", f"A股恐慌贪婪指数 {fear:.0f}，偏贪婪，按“贪婪时卖”的原则应减仓"
        elif fear <= 30:
            s, d = "中", f"A股恐慌贪婪指数 {fear:.0f}，恐慌但还没到买入线 20，可以准备、先不急"
        else:
            s, d = "中", f"A股恐慌贪婪指数 {fear:.0f}，不在买卖线附近"
        items.append(dict(item="市场情绪", status=s, detail=d))
    if cstate:
        s = "好" if cstate["in_market"] else "中"
        d = (f"按回测选出的规则（≤{cstate['buy']} 买入、≥{cstate['sell']} 卖出），当前应"
             + ("持有" if cstate["in_market"] else "空仓等待"))
        items.append(dict(item="恐慌买贪婪卖规则", status=s, detail=d))
    if temp_row is not None:
        tv = temp_row["temperature"]
        label = temperature_label(tv)
        s = "中" if label in ("正常", "未知") else ("好" if label in ("冰点", "偏冷") else "差")
        d = (f"市场温度 {tv:.0f}（{label}）：上涨 {int(temp_row['up'])} 家、下跌 {int(temp_row['down'])} 家，"
             f"涨停 {int(temp_row['limit_up'])}、跌停 {int(temp_row['limit_down'])}，成交额 {temp_row['amount']:.0f} 亿"
             + {"冰点": "；逆向看是机会", "偏冷": "；逆向看偏向机会", "偏热": "；追高要小心", "过热": "；过热，追高风险大"}.get(label, ""))
        items.append(dict(item="市场温度", status=s, detail=d))
    test = ((bt or {}).get("chosen") or {}).get("test") or {}
    if test.get("avg_next_open_net") is not None:
        v = test["avg_next_open_net"]
        s = "好" if v > 0 else "差"
        d = f"样本外每笔扣费后平均 {v * 100:+.2f}%，" + ("优势覆盖了成本" if v > 0 else "选股优势不够付交易成本，不宜照单隔夜买卖")
        items.append(dict(item="隔夜选股能否赚钱", status=s, detail=d))
    for c in conf:
        if c["strategy"].startswith("恐慌买贪婪卖（选股）"):
            p = c["p_positive"]
            s = "好" if p >= 0.9 else ("中" if p >= 0.6 else "差")
            d = (f"样本外年化 {c['ann'] * 100:.1f}%，90% 区间 {c['ci_low'] * 100:.1f}% 到 {c['ci_high'] * 100:.1f}%，"
                 f"重抽 {BOOT_SAMPLES} 次里赚钱的比例 {p * 100:.0f}%")
            items.append(dict(item="恐慌买贪婪卖可信度", status=s, detail=d))
    if sectors is not None and len(sectors):
        top3 = "、".join(sectors.index[:3])
        bot3 = "、".join(sectors.index[-3:])
        items.append(dict(item="行业强弱", status="中", detail=f"最强：{top3}；最弱：{bot3}"))
    if top_checks is not None and len(top_checks):
        flagged = top_checks[top_checks["flags"] != ""]
        s = "好" if flagged.empty else ("中" if len(flagged) <= 2 else "差")
        d = ("前几名候选没有明显风险提示" if flagged.empty else
             "有风险提示：" + "；".join(f"{r['name']}（{r['flags'].replace(';', '、')}）" for _, r in flagged.iterrows()))
        items.append(dict(item="候选股体检", status=s, detail=d))
    return items


# ---------- 汇总 ----------

def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def build(kline, basic, valuation, industry, ranking, picks, bt):
    recent = kline[kline["date"] >= kline["date"].max() - pd.Timedelta(days=int(VAL_HISTORY_DAYS * 1.5))]
    if valuation is not None:
        valuation = valuation[valuation["date"] >= recent["date"].min()]
    p = Panel(recent, valuation)
    names = basic.set_index("code")["code_name"]
    t = p.dates[-1]

    temp = market_temperature(p)
    temp_row = temp.iloc[-1]
    sectors = sector_strength(p, industry, ranking) if industry is not None else None
    checks = checkups(p, industry, ranking, names)
    conf = confidence(bt)
    fear = model.load_fear(p.dates)
    cstate = contrarian_state(bt, fear)
    top_codes = [pk["code"] for pk in (picks or {}).get("picks", [])][:(picks or {}).get("top_n", 5)]
    top_checks = checks.loc[[c for c in top_codes if c in checks.index]]
    items = checklist(picks, temp_row, sectors, conf, cstate, top_checks, bt)

    th = temp.iloc[-TEMP_DAYS:].reset_index().rename(columns={"index": "date"})
    th["date"] = th["date"].dt.strftime("%Y-%m-%d")
    temp_rows = [{k: _clean(v) for k, v in r.items()} for r in th.to_dict("records")]
    sector_rows = [] if sectors is None else [
        {"industry": i, **{k: _clean(v) for k, v in r.items()}} for i, r in sectors.iterrows()]
    check_rows = [{"code": c, **{k: _clean(v) for k, v in r.items()}} for c, r in checks.iterrows()]
    return dict(
        signal_date=str(t.date()),
        temperature=dict(value=_clean(temp_row["temperature"]), label=temperature_label(temp_row["temperature"])),
        contrarian=cstate,
        checklist=items,
        market=temp_rows,
        sectors=sector_rows,
        confidence=conf,
        checkups=check_rows,
    )


def write(out):
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, default=_clean))
