"""回测：事件覆盖率、误报率、分档未来收益、简单仓位模拟。

样本内 2005–2018 用于调参，样本外 2019 以后只看不调。
"""

import numpy as np
import pandas as pd

from . import config

PRICE_KEYS = {"us": "spx", "eu": "sx5e", "jp": "n225", "cn": "csi300", "hk": "hsi"}
SPLIT = pd.Timestamp("2019-01-01")
BUCKETS = [0, 20, 40, 60, 80, 100.01]
BUCKET_LABELS = ["0–20", "20–40", "40–60", "60–80", "80–100"]


def price_panel(series, calendar):
    px = {}
    for region, key in PRICE_KEYS.items():
        s = series.get(key)
        if s is not None:
            px[region] = s.reindex(s.index.union(calendar)).ffill(limit=config.FFILL_LIMIT).reindex(calendar)
    px = pd.DataFrame(px)
    # 全球股票合成：各区域本币日收益按区域权重加权
    rets = np.log(px).diff()
    w = pd.Series(config.REGION_WEIGHTS).reindex(rets.columns)
    avail = rets.notna().mul(w, axis=1).sum(axis=1)
    g = rets.fillna(0).mul(w, axis=1).sum(axis=1) / avail.replace(0, np.nan)
    px["global"] = np.exp(g.fillna(0).cumsum())
    return px


def drawdown_episodes(price, threshold=0.15, rebound=0.25):
    """返回回撤超过 threshold 的 (峰值日, 谷底日, 最大回撤) 列表。

    一段回撤在创出新高时结束；若长期未回到前高（如 A股 2007 年后），
    从谷底反弹超过 rebound 也算结束，之后重新找峰值，避免多次熊市被并成一段。
    """
    price = price.dropna()
    episodes = []
    peak_date, peak = price.index[0], price.iloc[0]
    trough_date, trough = peak_date, peak
    for d, p in price.items():
        deep = 1 - trough / peak >= threshold
        if p > peak or (deep and p / trough - 1 >= rebound):
            if deep:
                episodes.append((peak_date, trough_date, 1 - trough / peak))
            peak_date, peak, trough_date, trough = d, p, d, p
        elif p < trough:
            trough_date, trough = d, p
    if 1 - trough / peak >= threshold:
        episodes.append((peak_date, trough_date, 1 - trough / peak))
    return episodes


def event_capture(score, price, threshold=0.15, level=30):
    rows = []
    for peak, trough, dd in drawdown_episodes(price, threshold):
        window = score.loc[peak: trough + pd.Timedelta(days=30)].dropna()
        if window.empty:
            continue
        hit = _first_hit(window, level)
        dd_at_hit = None if hit is None else round((1 - price.loc[pd.Timestamp(hit)] / price.loc[peak]) * 100, 1)
        rows.append(dict(peak=peak.date(), trough=trough.date(), drawdown=round(dd * 100, 1),
                         min_score=round(window.min(), 1), first_hit=hit, dd_at_hit=dd_at_hit,
                         score_at_trough=round(score.get(trough, float("nan")), 1),
                         captured=bool(window.min() <= level)))
    return rows


def _first_hit(window, level):
    hit = window[window <= level]
    return hit.index[0].date() if len(hit) else None


def false_alarms(score, price, level=20, min_dd=0.05):
    """分数下穿 level（进入极度恐慌）时，前20日到后60日内市场从高点回撤从未超过 min_dd，记为误报。"""
    s = score.dropna()
    entries = s.index[(s <= level) & (s.shift(1) > level)]
    roll_dd = 1 - price / price.rolling(252, min_periods=20).max()
    rows = []
    for d in entries:
        i = price.index.get_loc(d)
        w = roll_dd.iloc[max(0, i - 20): i + 61]
        rows.append(dict(date=d.date(), max_dd=round(w.max() * 100, 1), false_alarm=bool(w.max() < min_dd)))
    return rows


def forward_table(score, price, horizons=(20, 60, 120)):
    df = pd.DataFrame({"score": score, "price": price}).dropna()
    df["bucket"] = pd.cut(df["score"], BUCKETS, labels=BUCKET_LABELS, right=False)
    for h in horizons:
        df[f"r{h}"] = df["price"].shift(-h) / df["price"] - 1
    fwd_min = df["price"][::-1].rolling(60, min_periods=1).min()[::-1].shift(-1)
    df["mdd60"] = 1 - fwd_min / df["price"]
    g = df.groupby("bucket", observed=False)
    out = pd.DataFrame({"days": g.size()})
    out.index.name = "分数段"
    for h in horizons:
        out[f"avg_r{h}%"] = (g[f"r{h}"].mean() * 100).round(2)
        out[f"win_r{h}%"] = (g[f"r{h}"].apply(lambda x: (x.dropna() > 0).mean()) * 100).round(1)
    out["avg_mdd60%"] = (g["mdd60"].mean() * 100).round(2)
    return out


def _perf(ret):
    ret = ret.dropna()
    if len(ret) < 60:
        return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, mdd=np.nan)
    eq = (1 + ret).cumprod()
    years = len(ret) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(252)
    mdd = (1 - eq / eq.cummax()).max()
    return dict(cagr=round(cagr * 100, 2), vol=round(vol * 100, 2),
                sharpe=round(cagr / vol, 2) if vol else np.nan, mdd=round(mdd * 100, 2))


def position_sim(score, price):
    """两种简单仓位规则，t 日收盘的分数决定 t+1 日仓位（无交易成本）。

    逆向：仓位 = 0.3 + 0.7 × (100 − 分数)/100，越恐慌（分数越低）越重仓。
    避险：config.risk_off 为真（分数 ≤ 40 且低于 20 日前，连续 2 天）时半仓，否则满仓。
    对照：买入持有，以及与逆向规则平均仓位相同的固定仓位。
    """
    df = pd.DataFrame({"score": score, "price": price}).dropna()
    r = df["price"].pct_change()
    contra = (0.3 + 0.7 * (100 - df["score"]) / 100).shift(1)
    riskoff = np.where(config.risk_off(df["score"]), 0.5, 1.0)
    riskoff = pd.Series(riskoff, index=df.index).shift(1)
    rules = {
        "买入持有": r,
        "逆向加仓": contra * r,
        f"固定仓位{contra.mean():.0%}": contra.mean() * r,
        "恐慌加深减仓": riskoff * r,
    }
    rows = {}
    for period, mask in (("样本内(–2018)", df.index < SPLIT), ("样本外(2019–)", df.index >= SPLIT)):
        for name, ret in rules.items():
            rows[(period, name)] = _perf(ret[mask])
    return pd.DataFrame(rows).T


def run(index_df, series):
    px = price_panel(series, index_df.index)
    report = {}
    for region in ["global"] + list(config.REGIONS):
        if region not in px or index_df[region].dropna().empty:
            continue
        score, price = index_df[region], px[region]
        start = score.first_valid_index()
        score, price = score.loc[start:], price.loc[start:]
        events = event_capture(score, price)
        alarms = false_alarms(score, price)
        report[region] = dict(
            start=start.date(),
            events=events,
            capture_rate=round(np.mean([e["captured"] for e in events]) * 100, 1) if events else None,
            alarms=alarms,
            false_alarm_rate=round(np.mean([a["false_alarm"] for a in alarms]) * 100, 1) if alarms else None,
            forward_in=forward_table(score[score.index < SPLIT], price[price.index < SPLIT]),
            forward_out=forward_table(score[score.index >= SPLIT], price[price.index >= SPLIT]),
            positions=position_sim(score, price),
            daily_change_median=round(score.diff().abs().median(), 2),
        )
    return report


def extreme_stats(index_df, series):
    """触底/触顶提示的历史表现（全球及各区域合并统计）。"""
    px = price_panel(series, index_df.index)
    rows = []
    for kind, label in (("bottom", "触底提示"), ("short_bottom", "短期触底提示"), ("top", "触顶提示")):
        for per, in_per in (("样本内(–2018)", lambda d: d < SPLIT), ("样本外(2019–)", lambda d: d >= SPLIT)):
            r60, drops, base_r60, base_drop = [], [], [], []
            for region in ["global"] + list(config.REGIONS):
                if region not in px:
                    continue
                p = px[region].reindex(index_df.index)
                f60 = p.shift(-60) / p - 1
                fmin = p[::-1].rolling(60, min_periods=1).min()[::-1].shift(-1)
                drop = (1 - fmin / p) > 0.10
                m = in_per(p.index) & f60.notna()
                base_r60.append(f60[m].mean())
                base_drop.append(drop[m].mean())
                for d in config.extreme_events(index_df[region], kind):
                    if in_per(d) and pd.notna(f60.get(d)):
                        r60.append(f60[d])
                        drops.append(bool(drop[d]))
            rows.append({"提示": label, "区间": per, "次数": len(r60),
                         "之后60日平均涨跌%": round(np.mean(r60) * 100, 1) if r60 else np.nan,
                         "所有交易日60日平均涨跌%": round(np.nanmean(base_r60) * 100, 1),
                         "之后60日上涨概率%": round(np.mean(np.array(r60) > 0) * 100, 0) if r60 else np.nan,
                         "之后60日再跌超10%概率%": round(np.mean(drops) * 100, 0) if drops else np.nan,
                         "所有交易日再跌超10%概率%": round(np.nanmean(base_drop) * 100, 0)})
    df = pd.DataFrame(rows).set_index(["提示", "区间"])
    for c in ("次数", "之后60日上涨概率%", "之后60日再跌超10%概率%", "所有交易日再跌超10%概率%"):
        df[c] = df[c].astype("Int64")
    return df


def extremes_markdown(stats):
    return "\n".join([
        "## 触底、触顶提示的历史表现", "",
        f"触底：分数在 {config.EXTREME_MEMORY} 个交易日内曾 ≤ {config.BOTTOM_LEVEL}，之后回升到 {config.BOTTOM_EXIT} 以上。"
        f"短期触底：{config.SHORT_WINDOW} 个交易日内分数下跌至少 {config.SHORT_DROP} 分、低点 ≤ {config.SHORT_LEVEL}，"
        f"之后从低点回升至少 {config.SHORT_REBOUND} 分。"
        f"触顶：曾 ≥ {config.TOP_LEVEL}，之后回落到 {config.TOP_EXIT} 以下。全球与五个区域合并统计。",
        "这些提示仅供参考，不参与减仓规则。触底、触顶提示没有可靠的预测力；短期触底提示之后多数时候守住了前期低点"
        "（scripts/experiment_short_bottom.py），但之后 60 日的表现并不比平时好，不是买入信号。", "",
        _md_table(stats), ""])


def _md_table(df):
    idx_name = df.index.name or ""
    head = [idx_name] + [str(c) for c in df.columns]
    rows = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"]
    for idx, row in df.iterrows():
        label = " ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
        rows.append("| " + " | ".join([label] + ["—" if pd.isna(v) else str(v) for v in row]) + " |")
    return "\n".join(rows)


def to_markdown(report, index_df):
    names = {"global": "全球", **config.REGIONS}
    lines = ["# 回测报告", "",
             f"数据区间：{index_df.index[0].date()} 至 {index_df.index[-1].date()}。"
             "样本内 2018 年底以前，样本外 2019 年以后。", "",
             f"当前参数：分位窗口 {config.RANK_WINDOW} 个交易日，平滑 {config.SMOOTH_SPAN} 日，"
             f"分数 0 = 极度恐慌、100 = 极度贪婪；减仓信号为分数 ≤ {config.RISK_OFF_LEVEL} 且低于 {config.RISK_OFF_LOOKBACK} 日前，"
             f"连续 {config.RISK_OFF_CONFIRM} 天成立。"
             "参数只依据样本内结果选定，复现见 `scripts/tune_params.py`。", ""]
    lines += ["## 汇总", "", "| 区域 | 起始 | 回撤>15% 事件数 | 覆盖率(≤30) | 下穿20 次数 | 误报率 | 日变动中位数 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for region, r in report.items():
        lines.append(f"| {names[region]} | {r['start']} | {len(r['events'])} | "
                     f"{r['capture_rate']}% | {len(r['alarms'])} | {r['false_alarm_rate']}% | {r['daily_change_median']} |")
    for region, r in report.items():
        lines += ["", f"## {names[region]}", "", "### 回撤事件", "",
                  "| 峰值 | 谷底 | 回撤% | 期间最低分 | 谷底当天分 | 首次≤30 | 首次≤30时已跌% | 捕捉 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for e in r["events"]:
            lines.append(f"| {e['peak']} | {e['trough']} | {e['drawdown']} | {e['min_score']} | {e['score_at_trough']} | "
                         f"{e['first_hit'] or '—'} | {e['dd_at_hit'] if e['dd_at_hit'] is not None else '—'} | "
                         f"{'是' if e['captured'] else '否'} |")
        for title, key in (("分档未来收益（样本内）", "forward_in"), ("分档未来收益（样本外）", "forward_out")):
            lines += ["", f"### {title}", "", _md_table(r[key])]
        lines += ["", "### 仓位模拟", "", _md_table(r["positions"])]
    return "\n".join(lines) + "\n"
