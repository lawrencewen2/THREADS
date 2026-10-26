"""主线行情：找出当下资金持续在做的概念题材，并在每条主线里挑龙头和潜力股。

做法（只用信号日 t 收盘及以前的数据）：
1. 每个概念用成分股等权日收益合成一条概念指数。
2. 打分看六件事：近 20 日跑赢全市场多少、持续性（近 20 天里有几天涨幅排在所有概念前 10%）、
   近 10 日每只成分股平均涨停次数、成交额放大（近 5 日对近 60 日）、上涨家数比例、快讯热度。
   各项在所有概念里排分位再平均。成分高度重叠的概念只留分高的一个。
3. 按近 5 日和近 20 日的走势判断阶段：启动、加速、分歧、退潮。
4. 每条主线挑两类股票：
   - 龙头：近 20 日涨停次数、涨幅、成交额都排在前面的；
   - 潜力：近 20 日涨得比主线里大多数少，但近 5 日开始走强、放量、站上 20 日线，还没连续涨停的（补涨候选）。
概念成分是当前的，用来回看历史会有幸存者偏差，历史检验只作参考。
"""

import json
import logging
import math
import re

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

CONCEPT_FILE = config.DATA_DIR / "concepts.csv"
OUT_FILE = config.PAGE_DIR / "themes.json"
MIN_MEMBERS, MAX_MEMBERS = 10, 400
TOP_THEMES = 6
PICKS = 3
CHART_DAYS = 60
# 不是题材的“概念”：资金通道、指数成分、财务状态、行情标签
NOT_THEME = re.compile(r"融资融券|转融|沪股通|深股通|港股通|MSCI|富时|标普|道琼斯|证金|汇金|社保|QFII|养老金|基金重仓|机构重仓|"
                       r"龙头股|昨日|连板|涨停|跌停|首板|破净|破发|ST|次新|新股|预增|预亏|扭亏|高送转|送转|股权转让|举牌|AH股|AB股|GDR|"
                       r"创业板综|深成500|上证180|上证380|中证500|沪深300|央视50|同花顺漂亮|百元股|低价股|高股息|红利|微盘|"
                       r"大盘|小盘|中盘|注册制|核准制|北交所|科创次新|专精特新|独角兽|参股|壳资源|并购重组|业绩")



# ---------- 概念成分（腾讯、新浪的公开行情接口） ----------

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
SINA_NODE = ("http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
             "?page={page}&num=100&sort=symbol&asc=1&node={node}&symbol=&_s_r_a=init")


def _bs_code(code):
    code = str(code)[-6:]
    if code.startswith("6"):
        return "sh." + code
    if code[:1] in ("0", "3"):
        return "sz." + code
    return None


def _get(url, enc="gbk", timeout=20):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "http://vip.stock.finance.sina.com.cn/"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(enc, errors="replace")


def sina_nodes():
    """新浪的概念（gn_）和新行业（new_）分类。"""
    s = _get("http://vip.stock.finance.sina.com.cn/q/view/newFLJK.php?param=class")
    s += _get("http://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php")
    nodes = dict(re.findall(r'"((?:gn|new)_\w+)":"(?:gn|new)_\w+,([^,]+),', s))
    return {k: v for k, v in nodes.items() if not NOT_THEME.search(v)}


def fetch_sina(pause=0.15):
    import time
    rows = []
    for node, name in sina_nodes().items():
        for page in range(1, 6):
            try:
                items = json.loads(_get(SINA_NODE.format(page=page, node=node)) or "[]")
            except Exception as exc:
                log.warning("sina %s p%d: %s", node, page, exc)
                break
            rows += [dict(concept=name, code=bc, source="sina") for bc in (_bs_code(it.get("code", "")) for it in items) if bc]
            time.sleep(pause)
            if len(items) < 100:
                break
    return pd.DataFrame(rows, columns=["concept", "code", "source"])


TENCENT_LIST = ("https://proxy.finance.qq.com/cgi/cgi-bin/rank/pt/getRank?board_type=gn&sort_type=price"
                "&direct=down&offset={off}&count=100")
TENCENT_MEMBERS = ("https://proxy.finance.qq.com/cgi/cgi-bin/rank/hs/getBoardRankList?_appver=11.17.0&board_code={code}"
                   "&sort_type=price&direct=down&offset={off}&count=100")


def tencent_boards():
    boards, off, total = {}, 0, 1
    while off < total:
        d = json.loads(_get(TENCENT_LIST.format(off=off), "utf-8"))["data"]
        total = int(d.get("total") or 0)
        boards.update({x["code"]: x["name"] for x in d["rank_list"]})
        off += 100
        if not d["rank_list"]:
            break
    return {k: v for k, v in boards.items() if not NOT_THEME.search(v)}


def fetch_tencent(pause=0.1):
    """腾讯自选股的概念板块（800 多个，含商业航天、半导体设备等新题材）。"""
    import time
    rows = []
    for code, name in tencent_boards().items():
        for off in range(0, MAX_MEMBERS + 100, 100):
            try:
                items = json.loads(_get(TENCENT_MEMBERS.format(code=code, off=off), "utf-8"))["data"]["rank_list"]
            except Exception as exc:
                log.warning("tencent %s %s: %s", name, off, exc)
                break
            rows += [dict(concept=name, code=bc, source="tencent")
                     for bc in (_bs_code(it.get("code", "")) for it in items) if bc]
            time.sleep(pause)
            if len(items) < 100:
                break
    return pd.DataFrame(rows, columns=["concept", "code", "source"])


def update_members(fetchers=(("tencent", lambda: fetch_tencent()), ("sina", lambda: fetch_sina()))):
    """抓概念成分写 concepts.csv：先用腾讯，不够再用新浪；都不行就保留旧文件。"""
    df = None
    for name, fn in fetchers:
        try:
            got = fn()
            log.info("%s: %d concepts, %d rows", name, got["concept"].nunique(), len(got))
            if got["concept"].nunique() >= 50:
                df = got.drop_duplicates(["concept", "code"])
                break
        except Exception as exc:
            log.warning("%s members failed: %s", name, exc)
    if df is None:
        return None
    CONCEPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CONCEPT_FILE, index=False)
    return df

# ---------- 合成概念指数 ----------

def concept_returns(p, members):
    """每个概念的等权日收益（单日涨跌截在 ±20%，停牌的不算）。返回 日期 × 概念。"""
    ret = (p.close / p.preclose - 1).where(p.traded).clip(-0.2, 0.2)
    out = {}
    for concept, codes in members.items():
        cols = [c for c in codes if c in ret.columns]
        if len(cols) >= MIN_MEMBERS:
            out[concept] = ret[cols].mean(axis=1)
    return pd.DataFrame(out)


def _limit_up(p):
    pct = p.limit_pct().where(~p.is_st, 0.05)
    up_px = (p.preclose * (1 + pct) + 1e-6).round(2)
    return (p.close >= up_px - 0.005) & p.traded


def _cum(r, n):
    return (1 + r.iloc[-n:]).prod() - 1


def score_concepts(p, members, cr, heat=None):
    """在信号日 t 给每个概念打分。cr 是 concept_returns 的结果（已截到 t）。"""
    mkt = (p.close / p.preclose - 1).where(p.traded).clip(-0.2, 0.2).mean(axis=1)
    lu = _limit_up(p)
    ret20 = p.close.iloc[-1] / p.close.iloc[-21] - 1
    ma20 = p.close.rolling(20, min_periods=15).mean().iloc[-1]
    amt = p.amount.fillna(0)
    # 每天涨幅排在所有概念前 10% 的记一次
    top = cr.iloc[-20:].rank(axis=1, pct=True) >= 0.9
    rows = {}
    for concept in cr.columns:
        cols = [c for c in members[concept] if c in p.close.columns]
        r = cr[concept]
        a5, a60 = amt[cols].iloc[-5:].sum(axis=1).mean(), amt[cols].iloc[-60:].sum(axis=1).mean()
        rows[concept] = dict(
            members=len(cols),
            r5=_cum(r, 5), r20=_cum(r, 20), r60=_cum(r, 60),
            excess20=_cum(r, 20) - _cum(mkt, 20),
            persist=int(top[concept].sum()),
            limit_ups10=float(lu[cols].iloc[-10:].sum().sum() / len(cols)),
            amount_ratio=float(a5 / a60) if a60 > 0 else np.nan,
            breadth=float((ret20[cols] > 0).mean()),
            above_ma20=float((p.close[cols].iloc[-1] > ma20[cols]).mean()),
            heat=float((heat or {}).get(concept, 0)),
        )
    df = pd.DataFrame(rows).T.apply(pd.to_numeric, errors="coerce")
    parts = ["excess20", "persist", "limit_ups10", "amount_ratio", "breadth"]
    if heat:
        parts.append("heat")
    df["score"] = df[parts].rank(pct=True).mean(axis=1) * 100
    df["stage"] = df.apply(stage, axis=1)
    return df.sort_values("score", ascending=False)


def stage(r):
    if r["excess20"] <= 0:
        return "偏弱"
    if r["r5"] > 0:
        return "加速" if r["persist"] >= 6 or r["limit_ups10"] >= 0.3 else "启动"
    return "分歧" if r["amount_ratio"] >= 1 else "退潮"


def dedupe(scored, members, n=TOP_THEMES, overlap=0.5):
    """成分股重叠超过一半的概念算同一条主线，只留分高的。"""
    keep = []
    for concept in scored.index:
        s = set(members[concept])
        if all(len(s & set(members[k])) / min(len(s), len(members[k])) < overlap for k in keep):
            keep.append(concept)
        if len(keep) >= n:
            break
    return keep


# ---------- 主线里挑股票 ----------

def pick_stocks(p, codes, names, n=PICKS):
    """龙头：涨停次数、20 日涨幅、成交额综合靠前；潜力：20 日涨得少于主线中位，但近 5 日转强、放量、站上 20 日线。"""
    cols = [c for c in codes if c in p.close.columns]
    c = p.close[cols]
    ok = p.traded[cols].iloc[-1] & ~p.is_st[cols].iloc[-1]
    amount20 = p.amount[cols].iloc[-20:].mean()
    ok &= amount20 >= 1e8  # 日均成交额至少 1 亿，太冷门的不列
    lu = _limit_up(p)[cols]
    df = pd.DataFrame({
        "r5": c.iloc[-1] / c.iloc[-6] - 1,
        "r20": c.iloc[-1] / c.iloc[-21] - 1,
        "limit_ups20": lu.iloc[-20:].sum(),
        "amount20": amount20 / 1e8,
        "vol_ratio": p.amount[cols].iloc[-5:].mean() / p.amount[cols].iloc[-20:].mean(),
        "above_ma20": c.iloc[-1] > c.rolling(20, min_periods=15).mean().iloc[-1],
        "ma20_up": c.rolling(20, min_periods=15).mean().iloc[-1] > c.rolling(20, min_periods=15).mean().iloc[-6],
        "streak": lu.iloc[-3:].sum(),
    })[ok].dropna(subset=["r5", "r20"])
    if df.empty:
        return [], []
    lead = (df["limit_ups20"].rank(pct=True) + df["r20"].rank(pct=True) + df["amount20"].rank(pct=True)).sort_values(ascending=False)
    leaders = list(lead.index[:n])
    pool = df.drop(index=leaders)
    pool = pool[(pool["r20"] < df["r20"].median()) & (pool["r5"] > 0) & pool["above_ma20"] & pool["ma20_up"]
                & (pool["vol_ratio"] > 1.1) & (pool["streak"] < 2)]
    pot = (pool["r5"].rank(pct=True) + pool["vol_ratio"].rank(pct=True) + (-pool["r20"]).rank(pct=True)).sort_values(ascending=False)
    potentials = list(pot.index[:n])

    def row(code, kind):
        r = df.loc[code]
        if kind == "龙头":
            why = f"近20日涨{r['r20']:.0%}，涨停{int(r['limit_ups20'])}次，日均成交{r['amount20']:.0f}亿"
        else:
            why = f"近20日只涨{r['r20']:.0%}，近5日涨{r['r5']:.0%}，量能放大到{r['vol_ratio']:.1f}倍，站上20日线"
        return dict(code=code, name=str(names.get(code, "")), kind=kind, r5=round(float(r["r5"]), 4),
                    r20=round(float(r["r20"]), 4), limit_ups20=int(r["limit_ups20"]), why=why)

    return [row(c, "龙头") for c in leaders], [row(c, "潜力") for c in potentials]


# ---------- 快讯热度 ----------

def short_name(concept):
    return re.sub(r"(概念|板块|产业链|行业)$", "", concept)


def news_heat(raw, concepts, days=5):
    """近几天 7x24 快讯里提到概念名（去掉“概念”等后缀）的条数，取 log 压缩。"""
    if not raw:
        return {}
    cutoff = (pd.Timestamp.now(tz="Asia/Shanghai") - pd.Timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
    texts = [r["title"] + r["body"] for r in raw if r.get("time", "") >= cutoff]
    out = {}
    for c in concepts:
        k = short_name(c)
        if len(k) >= 2:
            out[c] = math.log1p(sum(1 for t in texts if k in t))
    return out


# ---------- 汇总 ----------

def load_members(df=None):
    df = df if df is not None else pd.read_csv(CONCEPT_FILE, dtype=str)
    df = df[~df["concept"].str.contains(NOT_THEME)]
    m = df.groupby("concept")["code"].apply(lambda s: sorted(set(s)))
    return {k: v for k, v in m.items() if MIN_MEMBERS <= len(v) <= MAX_MEMBERS}


def as_of(p, t):
    """把 Panel 截到 t（含）为止，用来回看历史某一天。"""
    import copy
    q = copy.copy(p)
    keep = p.dates <= t
    q.dates = p.dates[keep]
    for col in ("traded", "is_st", "open", "high", "low", "close", "preclose", "amount", "turn", "volume"):
        setattr(q, col, getattr(p, col).loc[keep])
    return q


def build(p, members, names, raw_news=None, n=TOP_THEMES):
    cr = concept_returns(p, members)
    heat = news_heat(raw_news, cr.columns)
    scored = score_concepts(p, members, cr, heat)
    top = dedupe(scored, members, n)
    themes, picks, chart = [], [], []
    base = cr.iloc[-CHART_DAYS - 1:]
    for rank, concept in enumerate(top, 1):
        r = scored.loc[concept]
        leaders, potentials = pick_stocks(p, members[concept], names)
        themes.append(dict(rank=rank, theme=concept, stage=r["stage"], score=round(float(r["score"]), 1),
                           r5=round(float(r["r5"]), 4), r20=round(float(r["r20"]), 4), excess20=round(float(r["excess20"]), 4),
                           persist=int(r["persist"]), limit_ups10=round(float(r["limit_ups10"]), 2),
                           amount_ratio=round(float(r["amount_ratio"]), 2), breadth=round(float(r["breadth"]), 3),
                           heat=round(float(r["heat"]), 2), members=int(r["members"]),
                           leaders="、".join(x["name"] for x in leaders), potentials="、".join(x["name"] for x in potentials)))
        for x in leaders + potentials:
            picks.append(dict(theme=concept, **x))
        idx = (1 + base[concept].fillna(0)).cumprod()
        idx = idx / idx.iloc[0] - 1
        chart += [dict(date=str(d.date()), theme=concept, value=round(float(v), 4)) for d, v in idx.iloc[1:].items()]
    return dict(signal_date=str(p.dates[-1].date()), themes=themes, picks=picks, chart=chart,
                concepts=len(cr.columns), ranking=[dict(theme=k, score=round(float(v), 1), stage=s)
                                                  for k, v, s in zip(scored.index[:30], scored["score"][:30], scored["stage"][:30])])


def history_check(p, members, names, dates, n=3, horizon=20):
    """回看：在给定的日子里排出前几条主线，并算之后 horizon 天主线指数相对全部概念平均的超额。"""
    cr_all = concept_returns(p, members)
    rows = []
    for t in dates:
        t = p.dates[p.dates <= pd.Timestamp(t)][-1]
        q = as_of(p, t)
        cr = cr_all.loc[:t]
        scored = score_concepts(q, members, cr)
        top = dedupe(scored, members, n)
        pos = cr_all.index.get_loc(t)
        fwd = cr_all.iloc[pos + 1:pos + 1 + horizon]
        fwd_ret = (1 + fwd).prod() - 1 if len(fwd) else pd.Series(dtype=float)
        avg = float(fwd_ret.mean()) if len(fwd_ret) else np.nan
        for rank, c in enumerate(top, 1):
            leaders, _ = pick_stocks(q, members[c], names)
            rows.append(dict(date=str(t.date()), rank=rank, theme=c, stage=scored.loc[c, "stage"],
                             score=round(float(scored.loc[c, "score"]), 1), r20=round(float(scored.loc[c, "r20"]), 4),
                             fwd=round(float(fwd_ret.get(c, np.nan)), 4) if len(fwd) else None,
                             fwd_vs_avg=round(float(fwd_ret.get(c, np.nan) - avg), 4) if len(fwd) else None,
                             leaders="、".join(x["name"] for x in leaders)))
    return pd.DataFrame(rows)


def write(out):
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False))
