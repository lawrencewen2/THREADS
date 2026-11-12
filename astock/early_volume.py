"""早盘放量研究（2026-11-10）：用集合竞价（开盘价）和 9:30–10:30 的成交量，在 10:30 选股。

问题：10:30 时看到“竞价高开/低开 + 第一个小时放量 + 第一个小时涨跌”，能不能挑出当天收盘比 10:30 更高、
而且按 A股 T+1 规则第二天卖出后（带止损止盈、扣成本）还赚钱的股票？

数据：
- baostock 60 分钟线（2021 年起），只留每天第一根（9:30–10:30）：10:30 价格、第一小时最高最低、成交量、成交额
- 日线缓存（data/astock/cache/kline.pkl.gz）：开盘价 = 集合竞价成交价、昨收、收盘、第二天的开高低收
- 集合竞价的成交量只有腾讯最近 5 天的 1 分钟线里有（9:30 那根），历史回测拿不到，只能实盘开始逐日记录

训练期 2021–2023 定规则，2024 年起全部是样本外。
"""

import json
import logging
import math
from itertools import product

import numpy as np
import pandas as pd

from . import config, data

log = logging.getLogger(__name__)

HOUR_FILE = config.CACHE_DIR / "hour1.pkl.gz"
HOUR_START = "2021-01-01"  # 从境外 runner 拉 baostock 60 分钟线很慢（每只约 10 秒），只拉 2021 年起
HOUR_FIELDS = "date,time,code,open,high,low,close,volume,amount"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
# 回测选出的规则只写进报告目录。实盘名单读 astock/early_volume_model.json（early_live.MODEL_FILE），
# 只有回测真的过关、人工确认后才复制过去；2026-11-10 回测没过关，所以实盘只观察不出名单
MODEL_FILE = config.REPORT_DIR / "early_volume_rule.json"
REPORT_FILE = config.REPORT_DIR / "early_volume.md"
PAGE_FILE = config.PAGE_DIR / "early_volume.json"

MIN_TRADES = 300  # 训练期至少这么多笔，才算一条规则
MIN_DAY_SHARE = 0.15  # 至少 15% 的交易日有信号
TOP_N = 5  # 每天最多买几只（按量比从高到低）
CAPITAL_SHARE = 0.5  # T 日买、T+1 卖，资金分两半轮流用

# 规则网格
LB_MIN = (1.5, 2.0, 3.0, 5.0)  # 10:30 量比下限（每分钟成交额 / 过去 5 日每分钟平均成交额）
LB_MAX = (math.inf, 10.0)
GAP = {"不限": (-1, 1), "低开": (-1, -0.005), "平开±2%": (-0.02, 0.02), "高开0–5%": (0.0, 0.05), "高开2–7%": (0.02, 0.07)}
R1 = {"不限": (-1, 1), "第一小时涨": (0.0, 1), "第一小时跌": (-1, 0.0)}
POS = {"不限": (-0.01, 1.01), "收在高位≥0.7": (0.7, 1.01), "收在低位≤0.3": (-0.01, 0.3)}
GATE = {"不限": (-0.01, 1.01), "大盘强 上涨家数≥50%": (0.5, 1.01), "大盘弱 上涨家数<50%": (-0.01, 0.5)}
SL_GRID = (0.03, 0.05, 0.07)
TP_GRID = (0.03, 0.05, 0.08, 0.12)


# ---------------------------------------------------------------- 数据

def _fetch_hour_chunk(job):
    items, end = job
    parts, failed = [], []
    try:
        sess = data._session()
    except Exception as exc:
        log.warning("baostock unavailable: %s", exc)
        return parts, [c for c, _ in items]
    for n, (code, start) in enumerate(items):
        try:
            df = sess.query("query_history_k_data_plus", code, HOUR_FIELDS, start_date=start, end_date=end,
                            frequency="60", adjustflag="3")
        except data.BaostockBlocked as exc:
            log.warning("baostock blocked: %s", exc)
            return parts, failed + [c for c, _ in items[n:]]
        except Exception as exc:
            log.warning("hour %s failed: %s", code, exc)
            failed.append(code)
            continue
        if len(df):
            df = df[df["time"].str[8:12] == "1030"]
            parts.append(clean_hour(df))
    return parts, failed


def clean_hour(df):
    out = pd.DataFrame({"date": pd.to_datetime(df["date"]), "code": df["code"].values})
    for src, dst in (("open", "h1_open"), ("high", "h1_high"), ("low", "h1_low"), ("close", "h1_close"),
                     ("volume", "h1_vol"), ("amount", "h1_amount")):
        out[dst] = pd.to_numeric(df[src].values, errors="coerce").astype("float32")
    return out


def load_hour(path=HOUR_FILE):
    return pd.read_pickle(path) if path.exists() else None


def fetch_codes(basic):
    """要拉 60 分钟线的股票：沪深 A股，2021 年前已退市的不要。"""
    b = basic[(basic["type"] == "1") & basic["code"].str[:5].isin(config.BOARD_PREFIXES)]
    out = pd.to_datetime(b["outDate"], errors="coerce")
    b = b[out.isna() | (out >= pd.Timestamp(HOUR_START))]
    return sorted(b["code"])


SINA_URL = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
SINA_UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}


def _sina_one(code, session):
    import time as _t

    for i in range(4):
        try:
            r = session.get(SINA_URL, params=dict(symbol=code.replace(".", ""), scale=60, ma="no", datalen=1023),
                            headers=SINA_UA, timeout=20)
            if r.status_code == 200 and r.text.strip().startswith("["):
                rows = json.loads(r.text)
                break
            if r.status_code == 200 and r.text.strip() in ("null", "[]", ""):
                return None
        except Exception:
            pass
        _t.sleep(2 * 2 ** i)
    else:
        return None
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    df = df[df["day"].str[11:16] == "10:30"]
    out = pd.DataFrame({"date": pd.to_datetime(df["day"].str[:10]).values, "code": code})
    for src, dst in (("open", "h1_open"), ("high", "h1_high"), ("low", "h1_low"), ("close", "h1_close"), ("volume", "h1_vol")):
        out[dst] = pd.to_numeric(df[src].values, errors="coerce").astype("float32")
    # 新浪没有成交额：用第一小时均价近似（量比只看相对大小，影响很小）
    out["h1_amount"] = (out["h1_vol"] * (out["h1_open"] + out["h1_close"]) / 2).astype("float32")
    return out


def fetch_sina(codes, workers=6):
    """新浪 60 分钟线：每只最多 1023 根，约一年。速度快（几分钟拉完全市场），但历史短。"""
    import requests
    from concurrent.futures import ThreadPoolExecutor

    session = requests.Session()
    parts, failed = [], []
    with ThreadPoolExecutor(workers) as ex:
        for i, (code, df) in enumerate(zip(codes, ex.map(lambda c: _sina_one(c, session), codes)), 1):
            if df is None:
                failed.append(code)
            else:
                parts.append(df)
            if i % 500 == 0:
                log.info("sina %d/%d", i, len(codes))
    return pd.concat(parts, ignore_index=True), failed


def merge_parts(files):
    parts = [pd.read_pickle(f) for f in files]
    old = load_hour()
    return _save_hour(old, parts)


def _save_hour(old, parts, path=HOUR_FILE):
    out = pd.concat(([old] if old is not None else []) + parts, ignore_index=True)
    out = out.drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"]).reset_index(drop=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(path)
    return out


def update_hour(codes, end=None, workers=config.FETCH_WORKERS, fresh_since=None, path=HOUR_FILE):
    """每只股票一次查询（整段历史），已有缓存的从末尾前 5 天续拉。每只股票只查一次，避免再被 baostock 拉黑。

    第一次全量约需几个小时：每拉完约 300 只落盘一次，任务超时中断后下次只拉还没拉到的（缓存末尾 ≥ fresh_since 的跳过）。"""
    from multiprocessing import Pool

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    old = load_hour(path)
    last = old.groupby("code")["date"].max() if old is not None else pd.Series(dtype="datetime64[ns]")
    items = []
    for code in codes:
        s = HOUR_START
        if code in last.index:
            if fresh_since is not None and last[code] >= fresh_since:
                continue
            s = (last[code] - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        items.append((code, s))
    log.info("hour bars: %d codes to fetch", len(items))
    if not items:
        return old, []
    chunk = 50
    jobs = [(items[i:i + chunk], end) for i in range(0, len(items), chunk)]
    parts, failed = [], []
    with Pool(max(1, workers), initializer=data._init_worker) as pool:
        for i, (got, bad) in enumerate(pool.imap_unordered(_fetch_hour_chunk, jobs), 1):
            parts += got
            failed += bad
            log.info("hour chunks %d/%d", i, len(jobs))
            if len(parts) >= 300:
                old, parts = _save_hour(old, parts, path), []
    return _save_hour(old, parts, path), failed


def limit_pct(code, date, st):
    """涨跌停幅度：ST 5%，科创板 20%，创业板 2020-08-24 起 20%，其他 10%。"""
    pct = np.where(code.str.startswith("sh.68"), 0.2, 0.1)
    pct = np.where(code.str.startswith("sz.30") & (date >= pd.Timestamp("2020-08-24")), 0.2, pct)
    return np.where(st == 1, 0.05, pct)


def stamp(date):
    return np.where(date >= pd.Timestamp(config.STAMP_DUTY_CHANGE), config.STAMP_DUTY_NEW, config.STAMP_DUTY_OLD)


def build_panel(kline, hour, basic):
    """每行一个股票日：10:30 能看到的特征 + 之后的结果。"""
    k = kline[kline["date"] >= pd.Timestamp(HOUR_START) - pd.Timedelta(days=60)].copy()
    k = k[k["code"].str[:5].isin(config.BOARD_PREFIXES)].sort_values(["code", "date"])
    g = k.groupby("code", sort=False)
    k["vol5"] = g["volume"].transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
    k["amt5"] = g["amount"].transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
    k["amt20"] = g["amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    k["ret5"] = k["preclose"] / g["close"].shift(5) - 1
    for c in ("open", "high", "low", "close", "date", "tradestatus"):
        k["n_" + c] = g[c].shift(-1)
    k["nth"] = g.cumcount()
    p = k.merge(hour, on=["code", "date"], how="inner")
    p = p[p["date"] >= pd.Timestamp(HOUR_START)]
    hg = p.groupby("code", sort=False)
    p["h1v5"] = p["h1_vol"] / hg["h1_vol"].transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())

    ipo = pd.to_datetime(basic.set_index("code")["ipoDate"], errors="coerce")
    listed = (p["date"] - p["code"].map(ipo)).dt.days
    pct = limit_pct(p["code"], p["date"], p["isST"])
    up_px = np.round(p["preclose"] * (1 + pct) + 1e-6, 2)
    ok = ((p["tradestatus"] == 1) & (p["isST"] == 0) & (listed >= config.MIN_LISTED_DAYS)
          & (p["amt20"] >= config.MIN_AMOUNT_20D) & (p["h1_amount"] > 0) & (p["amt5"] > 0)
          & (p["preclose"] > 0) & (p["open"] > 0)
          & (p["h1_close"] < up_px - 0.005))  # 10:30 封涨停的买不到
    p = p[ok].copy()
    # 量比：第一小时每分钟成交额 / 过去 5 日每分钟平均成交额 =(h1/60)/(amt5/240)。用成交额不用成交量，
    # 因为腾讯实时行情的成交量单位各板块不一，成交额统一是万元，实盘和回测口径一致
    p["lb"] = 4 * p["h1_amount"] / p["amt5"]
    p["gap"] = p["open"] / p["preclose"] - 1
    p["r1"] = p["h1_close"] / p["open"] - 1
    p["rp"] = p["h1_close"] / p["preclose"] - 1
    rng = (p["h1_high"] - p["h1_low"]).replace(0, np.nan)
    p["pos"] = ((p["h1_close"] - p["h1_low"]) / rng).fillna(0.5)
    p["breadth"] = p.groupby("date")["rp"].transform(lambda s: (s > 0).mean())
    # 结果
    p["same"] = p["close"] / p["h1_close"] - 1  # 10:30 买到收盘
    entry = p["h1_close"] * (1 + config.SLIPPAGE) * (1 + config.COMMISSION)
    p["entry"] = entry
    nxt_ok = (p["n_tradestatus"] == 1) & (p["n_open"] > 0) & ((p["n_date"] - p["date"]).dt.days <= 7)
    p["sell_cost"] = (1 - config.SLIPPAGE) * (1 - config.COMMISSION - stamp(p["date"]))
    p["net_close"] = (p["n_close"] * p["sell_cost"] / entry - 1).where(nxt_ok)
    keep = ["date", "code", "lb", "h1v5", "gap", "r1", "rp", "pos", "breadth", "ret5", "same", "net_close",
            "h1_close", "entry", "sell_cost", "n_open", "n_high", "n_low", "n_close"]
    p = p[keep].dropna(subset=["net_close", "lb"]).reset_index(drop=True)
    return p


def tp_sl_return(p, sl, tp):
    """T+1 卖出：开盘就破止损/到止盈按开盘价，盘中先看止损（保守），否则收盘卖。止损止盈价按 10:30 买入价算。"""
    base = p["h1_close"]
    sl_px, tp_px = base * (1 - sl), base * (1 + tp)
    px = np.where(p["n_open"] <= sl_px, p["n_open"],
         np.where(p["n_open"] >= tp_px, p["n_open"],
         np.where(p["n_low"] <= sl_px, sl_px,
         np.where(p["n_high"] >= tp_px, tp_px, p["n_close"]))))
    return pd.Series(px * p["sell_cost"] / p["entry"] - 1, index=p.index)


# ---------------------------------------------------------------- 研究

def stats(ret, dates=None, same=None):
    ret = pd.Series(ret).dropna()
    if ret.empty:
        return dict(n=0)
    out = dict(n=int(len(ret)), win=float((ret > 0).mean()), mean=float(ret.mean()), median=float(ret.median()))
    if same is not None:
        out["same_up"] = float((pd.Series(same).loc[ret.index] > 0).mean())
    if dates is not None:
        out["days"] = int(pd.Series(dates).loc[ret.index].nunique())
    return out


def portfolio(sel, ret, all_days):
    """每天买量比最高的 TOP_N 只，等权，用一半资金（另一半在 T+1 卖出中）。返回净值序列。"""
    df = sel.assign(ret=ret)
    top = df.sort_values(["date", "lb"], ascending=[True, False]).groupby("date").head(TOP_N)
    day = top.groupby("date")["ret"].mean().reindex(all_days).fillna(0) * CAPITAL_SHARE
    return (1 + day).cumprod(), top


def curve_stats(eq):
    if eq.empty:
        return {}
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    dd = (eq / eq.cummax() - 1).min()
    return dict(ann=float(eq.iloc[-1] ** (1 / years) - 1), dd=float(dd), total=float(eq.iloc[-1] - 1))


def masks(p):
    """预先算好各维度的布尔条件，网格搜索时只做 AND。"""
    m = {}
    for v in LB_MIN:
        m[("lbmin", v)] = (p["lb"] >= v).values
    for v in LB_MAX:
        m[("lbmax", v)] = (p["lb"] < v).values
    for name, (col, grid) in {"gap": ("gap", GAP), "r1": ("r1", R1), "pos": ("pos", POS), "gate": ("breadth", GATE)}.items():
        for k, (lo, hi) in grid.items():
            m[(name, k)] = ((p[col] >= lo) & (p[col] < hi)).values
    return m


def search(p, train):
    m = masks(p)
    tr = train.values
    ndays = p.loc[train, "date"].nunique()
    ret = p["net_close"].values
    rows = []
    for lbmin, lbmax, gap, r1, pos, gate in product(LB_MIN, LB_MAX, GAP, R1, POS, GATE):
        if lbmax <= lbmin:
            continue
        sel = m[("lbmin", lbmin)] & m[("lbmax", lbmax)] & m[("gap", gap)] & m[("r1", r1)] & m[("pos", pos)] & m[("gate", gate)]
        s = sel & tr
        n = int(s.sum())
        if n < MIN_TRADES:
            continue
        days = p.loc[s, "date"].nunique()
        if days < MIN_DAY_SHARE * ndays:
            continue
        r = ret[s]
        rows.append(dict(lbmin=lbmin, lbmax=lbmax, gap=gap, r1=r1, pos=pos, gate=gate, n=n, days=int(days),
                         mean=float(r.mean()), win=float((r > 0).mean())))
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True), m


def rule_mask(m, r):
    return (m[("lbmin", r["lbmin"])] & m[("lbmax", r["lbmax"])] & m[("gap", r["gap"])] & m[("r1", r["r1"])]
            & m[("pos", r["pos"])] & m[("gate", r["gate"])])


def buckets(p, idx):
    """训练期：量比分组 × 竞价分组的胜率和平均收益（不挑参数，看规律是否单调）。"""
    d = p[idx].copy()
    d["量比"] = pd.cut(d["lb"], [0, 0.8, 1.2, 2, 3, 5, np.inf], labels=["<0.8", "0.8–1.2", "1.2–2", "2–3", "3–5", "≥5"])
    d["竞价"] = pd.cut(d["gap"], [-1, -0.03, -0.005, 0.005, 0.03, 0.07, 1],
                     labels=["低开>3%", "低开0.5–3%", "平开", "高开0.5–3%", "高开3–7%", "高开>7%"])
    d["第一小时"] = np.where(d["r1"] > 0, "涨", "跌")
    out = {}
    for col in ("量比", "竞价"):
        g = d.groupby(col, observed=True)
        out[col] = pd.DataFrame({"笔数": g.size(), "当天收盘高于10:30": g["same"].apply(lambda s: (s > 0).mean()),
                                 "T+1收盘卖胜率": g["net_close"].apply(lambda s: (s > 0).mean()),
                                 "平均净收益": g["net_close"].mean()})
    g = d.groupby(["量比", "第一小时"], observed=True)
    out["量比×第一小时"] = pd.DataFrame({"笔数": g.size(), "当天收盘高于10:30": g["same"].apply(lambda s: (s > 0).mean()),
                                    "T+1收盘卖胜率": g["net_close"].apply(lambda s: (s > 0).mean()),
                                    "平均净收益": g["net_close"].mean()})
    return out


def run(p):
    train = p["date"] <= pd.Timestamp(TRAIN_END)
    test = p["date"] >= pd.Timestamp(TEST_START)
    res = dict(rows=int(len(p)), start=str(p["date"].min().date()), end=str(p["date"].max().date()),
               baseline=dict(train=stats(p.loc[train, "net_close"], p["date"], p["same"]),
                             test=stats(p.loc[test, "net_close"], p["date"], p["same"])))
    res["buckets"] = buckets(p, train)
    res["buckets_test"] = buckets(p, test)
    grid, m = search(p, train)
    res["grid_size"] = int(len(grid))
    res["top_train"] = grid.head(10)
    # 训练期前 10 条规则在样本外的表现：看第一名是不是运气
    oos = []
    for _, r in grid.head(10).iterrows():
        sel = rule_mask(m, r) & test.values
        oos.append(stats(p.loc[sel, "net_close"], p["date"], p["same"]))
    res["top_test"] = oos
    best = grid.iloc[0].to_dict()
    sel_all = rule_mask(m, best)
    # 止损止盈只在训练期挑
    tp_rows = []
    for sl, tp in product(SL_GRID, TP_GRID):
        r = tp_sl_return(p[sel_all & train.values], sl, tp)
        tp_rows.append(dict(sl=sl, tp=tp, mean=float(r.mean()), win=float((r > 0).mean())))
    tp_df = pd.DataFrame(tp_rows).sort_values("mean", ascending=False)
    sl, tp = float(tp_df.iloc[0]["sl"]), float(tp_df.iloc[0]["tp"])
    best.update(sl=sl, tp=tp)
    res["best"] = best
    res["tp_grid"] = tp_df
    sp = p[sel_all].copy()
    sp["ret"] = tp_sl_return(sp, sl, tp)
    for name, idx in (("train", sp["date"] <= pd.Timestamp(TRAIN_END)), ("test", sp["date"] >= pd.Timestamp(TEST_START))):
        res[f"rule_{name}"] = stats(sp.loc[idx, "ret"], sp["date"], sp["same"])
        days = pd.DatetimeIndex(sorted(p.loc[train if name == "train" else test, "date"].unique()))
        eq, top = portfolio(sp[idx], sp.loc[idx, "ret"], days)
        res[f"port_{name}"] = dict(curve_stats(eq), **stats(top["ret"], top["date"], top["same"]))
        res[f"eq_{name}"] = eq
    sp["year"] = sp["date"].dt.year
    res["yearly"] = sp.groupby("year")["ret"].agg(["size", "mean", lambda s: (s > 0).mean()]).rename(
        columns={"size": "笔数", "mean": "平均净收益", "<lambda_0>": "胜率"})
    res["recent"] = sp.sort_values("date").tail(30)[["date", "code", "lb", "gap", "r1", "same", "ret"]]
    return res


# ---------------------------------------------------------------- 输出

def _pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def _srow(name, s):
    if not s or not s.get("n"):
        return f"| {name} | 0 | — | — | — | — |"
    return (f"| {name} | {s['n']} | {_pct(s.get('same_up'))} | {_pct(s['win'])} | {_pct(s['mean'], 2)} | "
            f"{s.get('days', '—')} |")


def _df_md(df, pct_cols=()):
    cols = list(df.columns)
    idx_name = " / ".join(str(n) for n in df.index.names if n) or ""
    lines = ["| " + " | ".join([idx_name] + [str(c) for c in cols]) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, row in df.iterrows():
        key = " / ".join(map(str, i)) if isinstance(i, tuple) else str(i)
        vals = [(_pct(v, 2) if c in pct_cols else (f"{v:.0f}" if isinstance(v, (int, float, np.integer, np.floating)) else str(v)))
                for c, v in row.items()]
        lines.append("| " + " | ".join([key] + vals) + " |")
    return "\n".join(lines)


def describe(r):
    lbmax = "" if r["lbmax"] == math.inf else f"、< {r['lbmax']:g}"
    return (f"10:30 量比 ≥ {r['lbmin']:g}{lbmax}；竞价 {r['gap']}；{r['r1']}；第一小时位置 {r['pos']}；{r['gate']}")


def to_markdown(res):
    b = res["best"]
    L = [f"# 早盘放量选股回测（集合竞价 + 9:30–10:30 成交量）", "",
         f"数据 {res['start']} 至 {res['end']}，{res['rows']:,} 个可交易的股票日。训练期到 {TRAIN_END} 定规则，{TEST_START} 起是样本外。",
         "10:30 按第一小时收盘价买入（加滑点和佣金），A股 T+1，第二天卖出。“当天收盘高于10:30”是用户关心的当天上涨，"
         "“胜率”是第二天卖出扣完成本后是否赚钱。集合竞价只用了开盘价（高开低开），竞价成交量没有历史数据。", "",
         "## 不挑股票的基准（所有股票 10:30 买、第二天收盘卖）", "",
         "| 区间 | 笔数 | 当天收盘高于10:30 | 胜率 | 平均净收益 | 天数 |", "|---|---|---|---|---|---|",
         _srow("训练期", res["baseline"]["train"]), _srow("样本外", res["baseline"]["test"]), "",
         "## 规律：量比、竞价分组（训练期，不挑参数）", ""]
    pc = ("当天收盘高于10:30", "T+1收盘卖胜率", "平均净收益")
    for k, df in res["buckets"].items():
        L += [f"### {k}", "", _df_md(df, pc), ""]
    L += ["## 同样的分组在样本外", ""]
    for k, df in res["buckets_test"].items():
        L += [f"### {k}", "", _df_md(df, pc), ""]
    L += [f"## 训练期最好的规则（{res['grid_size']} 条候选里挑第一）", "", describe(b),
          f"，止损 {b['sl'] * 100:.0f}%、止盈 {b['tp'] * 100:.0f}%（按 10:30 买入价）。", "",
          "| 区间 | 笔数 | 当天收盘高于10:30 | 胜率 | 平均净收益 | 天数 |", "|---|---|---|---|---|---|",
          _srow("训练期", res["rule_train"]), _srow("样本外", res["rule_test"]), "",
          f"每天最多买量比最高的 {TOP_N} 只、用一半资金：", "",
          "| 区间 | 年化 | 最大回撤 | 胜率 | 笔数 |", "|---|---|---|---|---|"]
    for name, k in (("训练期", "port_train"), ("样本外", "port_test")):
        s = res[k]
        L.append(f"| {name} | {_pct(s.get('ann'))} | {_pct(s.get('dd'))} | {_pct(s.get('win'))} | {s.get('n', 0)} |")
    L += ["", "### 训练期前 10 条规则在样本外", "",
          "| # | 规则 | 训练平均 | 训练胜率 | 样本外笔数 | 样本外胜率 | 样本外平均 |", "|---|---|---|---|---|---|---|"]
    for i, ((_, r), s) in enumerate(zip(res["top_train"].iterrows(), res["top_test"]), 1):
        L.append(f"| {i} | {describe(r)} | {_pct(r['mean'], 2)} | {_pct(r['win'])} | {s.get('n', 0)} | "
                 f"{_pct(s.get('win'))} | {_pct(s.get('mean'), 2)} |")
    L += ["", "### 选中规则按年", "", _df_md(res["yearly"], ("平均净收益", "胜率")), ""]
    return "\n".join(L)


def page_data(res):
    def clean(d):
        return {k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v) for k, v in d.items()}

    eq = pd.concat([res["eq_train"], res["eq_test"] * res["eq_train"].iloc[-1]]) if len(res["eq_train"]) else res["eq_test"]
    return dict(
        generated=pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
        start=res["start"], end=res["end"], rule=clean(res["best"]), rule_text=describe(res["best"]),
        baseline={k: clean(v) for k, v in res["baseline"].items()},
        rule_train=clean(res["rule_train"]), rule_test=clean(res["rule_test"]),
        port_train=clean(res["port_train"]), port_test=clean(res["port_test"]),
        top_test=[clean(s) for s in res["top_test"]],
        curve=[[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in eq.resample("W").last().dropna().items()],
    )


def _json(o):
    return o.item() if hasattr(o, "item") else str(o)


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(to_markdown(res), encoding="utf-8")
    page = page_data(res)
    PAGE_FILE.write_text(json.dumps(page, ensure_ascii=False, indent=1, default=_json), encoding="utf-8")
    b = res["best"]
    model = dict(rule={k: (None if v == math.inf else v) for k, v in b.items() if k in
                       ("lbmin", "lbmax", "gap", "r1", "pos", "gate", "sl", "tp")},
                 gap_range=GAP[b["gap"]], r1_range=R1[b["r1"]], pos_range=POS[b["pos"]], gate_range=GATE[b["gate"]],
                 rule_text=describe(b), top_n=TOP_N, oos=page["rule_test"], oos_port=page["port_test"], fitted=page["generated"])
    MODEL_FILE.write_text(json.dumps(model, ensure_ascii=False, indent=1, default=_json), encoding="utf-8")
