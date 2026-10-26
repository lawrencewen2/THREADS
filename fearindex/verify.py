"""数据核查：原始数据体检、计算复核、发布文件一致性、跨数据源比对。

每天在更新后运行，结果写入 reports/verification.md。核查不改动任何数据，只报告。
"""

import io
import json
import logging
import re

import numpy as np
import pandas as pd

from . import compute, config, publish

log = logging.getLogger(__name__)

OK, WARN, FAIL = "✅", "⚠️", "❌"


# ---------- 1. 原始数据体检 ----------

def raw_checks(series, today=None):
    """每个序列：最新日期、周末行、非正值、近两年缺口、近两年最大单日波动。"""
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    rows = []
    for key, s in series.items():
        recent = s[s.index >= today - pd.Timedelta(days=730)]
        lag = len(pd.bdate_range(s.index[-1], today)) - 1
        weekend = int((s.index.dayofweek >= 5).sum())
        nonpos = int((s <= 0).sum())
        gaps = recent.index.to_series().diff().dt.days
        max_gap = int(gaps.max()) if len(gaps.dropna()) else 0
        r = np.log(recent).diff().dropna()
        big = r.abs().sort_values(ascending=False).head(3)
        moves = "，".join(f"{d:%Y-%m-%d} {r[d] * 100:+.1f}%" for d in big.index)
        problems = []
        if lag > 3:
            problems.append(f"已 {lag} 个工作日没有新数据")
        if weekend:
            problems.append(f"{weekend} 行周末数据")
        if nonpos:
            problems.append(f"{nonpos} 个非正值")
        if max_gap > 16:  # 春节/国庆长假约 10 天，超过两周视为断档
            problems.append(f"近两年最长断档 {max_gap} 天")
        rows.append(dict(key=key, rows=len(s), first=s.index[0].date(), last=s.index[-1].date(),
                         status=WARN if problems else OK, problems="；".join(problems) or "—", moves=moves))
    return rows


# ---------- 2. 计算复核 ----------

def _pct_rank_naive(values, x):
    """与 pandas rank(pct=True, method='average') 等价的直接计算，用来独立复核。"""
    less = (values < x).sum()
    equal = (values == x).sum()
    return (less + (equal + 1) / 2) / len(values) * 100


def recompute_percentiles(series, comp_df, days=5):
    """对每个成分最近 days 个交易日，用朴素方法重算分位，与发布值比较（发布值为 100 − 恐慌分位）。"""
    raw = compute.raw_components(series)
    worst = {}
    for cid, s in raw.items():
        if cid not in comp_df:
            continue
        diffs = []
        for d in s.index[-days:]:
            hist = s.loc[:d].iloc[-config.RANK_WINDOW:]
            if len(hist) < config.RANK_MIN_PERIODS or d not in comp_df.index:
                continue
            mine = 100 - _pct_rank_naive(hist.values, hist.iloc[-1])
            diffs.append(abs(mine - comp_df.at[d, cid]))
        if diffs:
            worst[cid] = max(diffs)
    return worst


def recompute_index(comp_df):
    """只用发布的成分分数，按配置权重从头重算区域分和全球分（含平滑）。"""
    fear = 100 - comp_df
    reg = compute.region_scores(fear)
    idx = reg[list(config.REGIONS)].copy()
    idx.insert(0, "global", compute.global_score(reg))
    idx["common"] = reg["global"]
    return 100 - compute.smooth(idx)


def realtime_replay(series, index_df, back=10):
    """把数据截断到 back 个交易日前重算，确认当天能算出的分数和现在发布的历史值一致（无未来数据）。"""
    cut = index_df.index[-back - 1]
    truncated = {k: s[s.index <= cut] for k, s in series.items()}
    past, _, _ = compute.build(truncated)
    common = past.index.intersection(index_df.index)
    diff = (past.loc[common].iloc[-1] - index_df.loc[common].iloc[-1]).abs()
    return cut.date(), diff.drop("common", errors="ignore")


# ---------- 3. 发布文件一致性 ----------

def published_consistency(root=publish.ROOT):
    """latest.json、index.csv、README 表格、各区域折线图的最后一个点，四处数字应一致。"""
    problems, checked = [], 0
    data = root / "data"
    latest = json.loads((data / "latest.json").read_text())
    index_df = pd.read_csv(data / "index.csv", parse_dates=["date"], index_col="date")
    last = index_df.iloc[-1]
    if str(index_df.index[-1].date()) != latest["date"]:
        problems.append(f"latest.json 日期 {latest['date']} ≠ index.csv {index_df.index[-1].date()}")
    readme = (root / "README.md").read_text()
    block = re.search(r"<!-- latest:start -->(.*)<!-- latest:end -->", readme, re.S).group(1)
    for key, r in latest["regions"].items():
        if r["score"] is None:
            continue
        checked += 1
        if abs(r["score"] - last[key]) > 0.05:
            problems.append(f"{r['name']}：latest.json {r['score']} ≠ index.csv {last[key]}")
        week = index_df[key].iloc[-6]
        if r["change_5d"] is not None and abs(r["change_5d"] - round(last[key] - week, 1)) > 0.11:
            problems.append(f"{r['name']}：5日变化 {r['change_5d']} 与 index.csv 不符")
        m = re.search(rf"^\| {r['name']} \| (\d+) \|", block, re.M)
        if not m or int(m.group(1)) != round(r["score"]):
            problems.append(f"{r['name']}：README 汇总表读数与数据不符")
        if key in config.REGIONS:
            h = re.search(rf"^### {r['name']}：(\d+)", block, re.M)
            if not h or int(h.group(1)) != round(r["score"]):
                problems.append(f"{r['name']}：README 分区域表标题读数与数据不符")
            svg = root / "docs" / "regions" / f"{key}.svg"
            # 折线上每个点的数值标签用 #374151 着色（见 publish.svg_recent），最后一个即最新读数
            labels = re.findall(r'fill="#374151">(\d+)<', svg.read_text()) if svg.exists() else []
            if not labels or int(labels[-1]) != round(r["score"]):
                problems.append(f"{r['name']}：折线图最后一个点与数据不符")
    return checked, problems


# ---------- 4. 跨数据源比对 ----------

def _http_csv(url, **kw):
    import requests

    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}, **kw)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def _series(df, date_col, value_col):
    s = pd.to_numeric(df[value_col], errors="coerce")
    s.index = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize()
    return s.dropna().sort_index()


def _cboe(name):
    df = _http_csv(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv")
    return _series(df, "DATE", "CLOSE")


def _stooq(sym):
    return _series(_http_csv(f"https://stooq.com/q/d/l/?s={sym}&i=d"), "Date", "Close")


def _ecb(cur):
    """欧洲央行参考汇率（frankfurter.app），每天 14:15 CET 定价。cur 形如 'CHF' 或 'JPY/USD'。"""
    import requests

    start = (pd.Timestamp.today() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    r = requests.get(f"https://api.frankfurter.app/{start}..", params={"from": "EUR", "to": "CHF,JPY,USD"}, timeout=30)
    r.raise_for_status()
    rates = pd.DataFrame(r.json()["rates"]).T
    rates.index = pd.to_datetime(rates.index)
    if "/" in cur:
        a, b = cur.split("/")
        return (rates[a] / rates[b]).sort_index()
    return rates[cur].sort_index()


def _treasury10y():
    year = pd.Timestamp.today().year
    frames = []
    for y in (year - 1, year):
        url = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
               f"{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv")
        frames.append(_http_csv(url))
    return _series(pd.concat(frames), "Date", "10 Yr")


def _ak(fn, *args, date_col="date", value_col="close", **kw):
    import akshare as ak

    return _series(getattr(ak, fn)(*args, **kw), date_col, value_col)


def _ak_sum(fn, symbols, value_col):
    parts = [_ak(fn, symbol=s, value_col=value_col) for s in symbols]
    total = parts[0]
    for p in parts[1:]:
        total = total + p
    return total.dropna()


def _sina_us(symbol):
    return _ak("stock_us_daily", symbol=symbol, adjust="")


def _first(*getters):
    """依次尝试多个第二来源，返回第一个取到的。"""
    def go():
        errors = []
        for g in getters:
            try:
                return g()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}")
        raise RuntimeError("、".join(errors))
    return go


def _sse_margin():
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=60)
    return _ak("stock_margin_sse", start_date=f"{start:%Y%m%d}", end_date=f"{end:%Y%m%d}",
               date_col="信用交易日期", value_col="融资余额")


# (序列, 第二来源说明, 取数函数, 比较方式, 容差)
# level：逐日比较数值，容差为相对偏差上限；ratio：两边单位可能不同（如手与股），比较两者比值的稳定性。
CROSS_CHECKS = [
    ("spx", "新浪 标普500", _first(lambda: _ak("index_us_stock_sina", symbol=".INX"), lambda: _stooq("^spx")),
     "level", 0.005),
    ("vix", "CBOE 官方 VIX 历史", lambda: _cboe("VIX"), "level", 0.01),
    ("vix3m", "CBOE 官方 VIX3M 历史", lambda: _cboe("VIX3M"), "level", 0.01),
    ("hyg", "新浪 美股 HYG", _first(lambda: _sina_us("HYG"), lambda: _stooq("hyg.us")), "level", 0.005),
    ("ief", "新浪 美股 IEF", _first(lambda: _sina_us("IEF"), lambda: _stooq("ief.us")), "level", 0.005),
    ("us10y", "美国财政部 10 年期收益率", _treasury10y, "level", 0.03),
    ("sx5e", "新浪 欧洲Stoxx50指数", lambda: _ak("index_global_hist_sina", symbol="欧洲Stoxx50指数"), "level", 0.005),
    ("eurchf", "欧洲央行参考汇率", lambda: _ecb("CHF"), "level", 0.015),
    ("n225", "新浪 日经225指数", lambda: _ak("index_global_hist_sina", symbol="日经225指数"), "level", 0.005),
    ("usdjpy", "欧洲央行参考汇率（JPY/USD 交叉）", lambda: _ecb("JPY/USD"), "level", 0.015),
    ("csi300", "新浪 沪深300", lambda: _ak("stock_zh_index_daily", symbol="sh000300"), "level", 0.005),
    ("a_turnover", "新浪 上证+深成 成交量", lambda: _ak_sum("stock_zh_index_daily", ["sh000001", "sz399001"], "volume"),
     "ratio", 0.03),
    ("sh_margin", "上交所官网 融资余额", _sse_margin, "ratio", 0.005),
    ("hsi", "新浪 恒生指数", lambda: _ak("stock_hk_index_daily_sina", symbol="HSI"), "level", 0.005),
    ("hstech", "新浪 恒生科技指数（3033.HK 为跟踪 ETF）", lambda: _ak("stock_hk_index_daily_sina", symbol="HSTECH"),
     "ratio", 0.02),
    ("gold", "新浪 COMEX 黄金期货", lambda: _ak("futures_foreign_hist", symbol="GC"), "level", 0.01),
]

NO_SECOND_SOURCE = {
    "qvix50": "50ETF 期权波动率只有期权论坛（optbbs）一家免费发布",
    "move": "MOVE 指数为 ICE 专有数据，免费渠道只有 Yahoo",
}


def compare(ours, theirs, mode, tol, days=20):
    ours, theirs = (s[~s.index.duplicated(keep="last")] for s in (ours, theirs))
    both = pd.concat([ours, theirs], axis=1, join="inner").dropna().iloc[-days:]
    if len(both) < 5:
        return dict(n=len(both), status=WARN, detail="重叠日期不足 5 天，无法比较")
    ratio = both.iloc[:, 0] / both.iloc[:, 1]
    if mode == "level":
        dev = (ratio - 1).abs()
    else:
        dev = (ratio / ratio.median() - 1).abs()
    worst = dev.idxmax()
    detail = f"中位偏差 {dev.median() * 100:.2f}%，最大 {dev.max() * 100:.2f}%（{worst:%m-%d}）"
    if mode == "ratio":
        detail += f"，两边比值约 {ratio.median():.4g}"
    return dict(n=len(both), status=OK if dev.max() <= tol else WARN, detail=detail,
                worst_date=worst.date(), ours=both.loc[worst].iloc[0], theirs=both.loc[worst].iloc[1])


def cross_source(series):
    rows = []
    for key, label, getter, mode, tol in CROSS_CHECKS:
        if key not in series:
            continue
        try:
            theirs = getter()
        except Exception as exc:
            rows.append(dict(key=key, source=label, status="—", detail=f"第二来源取数失败：{type(exc).__name__} {str(exc)[:60]}"))
            continue
        try:
            res = compare(series[key], theirs, mode, tol)
        except Exception as exc:
            rows.append(dict(key=key, source=label, status="—", detail=f"比较出错：{type(exc).__name__} {str(exc)[:60]}"))
            continue
        if res["status"] == WARN and "ours" in res:
            res["detail"] += f"；我方 {res['ours']:.6g} vs 对方 {res['theirs']:.6g}"
        rows.append(dict(key=key, source=label, tol=f"{tol * 100:g}%", **res))
    return rows


# ---------- 报告 ----------

def intraday_reconcile(log_dir, index_df, days=10):
    """盘中估算的最后一个读数与收盘后正式读数相差多少（每个区域、最近 days 个交易日）。"""
    from pathlib import Path

    files = sorted(Path(log_dir).glob("*.csv")) if log_dir else []
    if not files:
        return None
    log = pd.concat([pd.read_csv(f) for f in files[-2:]], ignore_index=True)
    rows = []
    for key in ["global"] + list(config.REGIONS):
        if key not in log or f"{key}_date" not in log:
            continue
        sub = log.dropna(subset=[key, f"{key}_date"])
        last = sub.groupby(f"{key}_date")[key].last()
        last.index = pd.to_datetime(last.index)
        common = last.index.intersection(index_df.index)[-days:]
        if len(common):
            diff = (last.loc[common] - index_df.loc[common, key]).abs()
            rows.append(dict(key=key, n=len(common), median=float(diff.median()), max=float(diff.max()),
                             worst=str(diff.idxmax().date())))
    return rows


def run(series, index_df, comp_df, online=True, intraday_log=None):
    report = dict(raw=raw_checks(series))
    report["pct"] = recompute_percentiles(series, comp_df)
    recomputed = recompute_index(comp_df)
    tail = index_df.index[-21:]
    report["index_diff"] = (recomputed.loc[tail] - index_df.loc[tail]).abs().max()
    report["replay"] = realtime_replay(series, index_df)
    report["published"] = published_consistency()
    report["cross"] = cross_source(series) if online else None
    report["intraday"] = intraday_reconcile(intraday_log, index_df)
    return report


def to_markdown(rep, date):
    lines = [f"# 数据核查报告（{date}）", "",
             "每天更新后自动运行。核查只报告、不改数据；发现问题会在下次修复数据源或清洗规则后重算。", ""]

    pct_bad = {c: v for c, v in rep["pct"].items() if v > 0.1}
    idx_bad = rep["index_diff"][rep["index_diff"] > 0.2]
    cut, replay = rep["replay"]
    replay_bad = replay[replay > 0.1]
    n_pub, pub_problems = rep["published"]
    raw_warn = [r for r in rep["raw"] if r["status"] != OK]
    cross = rep["cross"]
    summary = [
        ("原始数据体检", f"{len(rep['raw'])} 个序列", raw_warn and WARN or OK,
         "；".join(f"{r['key']} {r['problems']}" for r in raw_warn) or "无周末行、无非正值、无长断档、都已更新"),
        ("成分分位复核", f"{len(rep['pct'])} 个成分 × 最近 5 天", pct_bad and FAIL or OK,
         "；".join(f"{c} 差 {v:.2f}" for c, v in pct_bad.items()) or
         f"独立重算与发布值最大相差 {max(rep['pct'].values(), default=0):.2f} 分（四舍五入误差）"),
        ("区域与全球分复核", "最近 21 天", len(idx_bad) and FAIL or OK,
         "；".join(f"{k} 差 {v:.2f}" for k, v in idx_bad.items()) or
         f"按权重从成分重算，最大相差 {rep['index_diff'].max():.2f} 分"),
        ("实时回放（无未来数据）", f"截断到 {cut} 重算", len(replay_bad) and FAIL or OK,
         "；".join(f"{k} 差 {v:.1f}" for k, v in replay_bad.items()) or
         f"当天能算出的分数与现在发布的历史值最大相差 {replay.max():.2f} 分"),
        ("发布文件一致性", f"{n_pub} 个读数 × 4 处", pub_problems and FAIL or OK,
         "；".join(pub_problems) or "latest.json、index.csv、README 表格、折线图数字一致"),
    ]
    if cross is None:
        summary.append(("跨数据源比对", "—", "—", "离线运行，未比对"))
    else:
        warn = [r for r in cross if r["status"] == WARN]
        failed = [r for r in cross if r["status"] == "—"]
        ok_n = sum(r["status"] == OK for r in cross)
        detail = f"{ok_n} 个序列与第二来源一致"
        if warn:
            detail += "；偏差超容差：" + "、".join(r["key"] for r in warn)
        if failed:
            detail += "；第二来源取数失败：" + "、".join(r["key"] for r in failed)
        summary.append(("跨数据源比对", f"{len(cross)} 个序列 × 最近 20 天", warn and WARN or OK, detail))

    intra = rep.get("intraday")
    if intra:
        # 美欧日港所有成分都有盘中行情，应基本一致；A股的融资余额、全球的 MOVE 只能沿用前一天
        tol = {"cn": 2.0, "global": 1.5}
        bad = [r for r in intra if r["max"] > tol.get(r["key"], 1.0)]
        summary.append(("盘中估算与收盘读数", f"{len(intra)} 个区域 × 最近 {max(r['n'] for r in intra)} 天",
                        bad and WARN or OK,
                        "；".join(f"{r['key']} 最大差 {r['max']:.1f}（{r['worst']}）" for r in bad) or
                        "最后一个盘中读数与正式读数最大相差 " + "、".join(f"{r['key']} {r['max']:.1f}" for r in intra)))
    else:
        summary.append(("盘中估算与收盘读数", "—", "—", "还没有盘中记录"))

    lines += ["## 汇总", "", "| 核查项 | 范围 | 结果 | 说明 |", "| --- | --- | --- | --- |"]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in summary]

    if cross is not None:
        lines += ["", "## 跨数据源比对", "",
                  "| 序列 | 第二来源 | 容差 | 天数 | 结果 | 说明 |", "| --- | --- | --- | --- | --- | --- |"]
        for r in cross:
            lines.append(f"| {r['key']} | {r['source']} | {r.get('tol', '—')} | {r.get('n', '—')} | "
                         f"{r['status']} | {r['detail']} |")
        for key, why in NO_SECOND_SOURCE.items():
            lines.append(f"| {key} | 无 | — | — | — | {why}，只做原始数据体检 |")

    lines += ["", "## 原始数据体检", "",
              "| 序列 | 行数 | 起始 | 最新 | 结果 | 问题 | 近两年最大单日波动（供人工核对是否为真实行情） |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rep["raw"]:
        lines.append(f"| {r['key']} | {r['rows']} | {r['first']} | {r['last']} | {r['status']} | "
                     f"{r['problems']} | {r['moves']} |")
    return "\n".join(lines) + "\n"
