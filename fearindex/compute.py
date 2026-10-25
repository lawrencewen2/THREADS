"""成分计算、滚动分位数标准化、区域与全球合成。

每一步只用当天及以前的数据，所以历史值与实盘值算法一致，原始数据不变时历史分数不会漂移。
"""

import numpy as np
import pandas as pd

from . import config


def _align(*series):
    df = pd.concat(series, axis=1, join="inner").dropna()
    return [df.iloc[:, i] for i in range(df.shape[1])]


def _ret(s, n):
    return s / s.shift(n) - 1


TRANSFORMS = {
    "level": lambda s: s,
    "ratio": lambda a, b: (lambda a, b: a / b)(*_align(a, b)),
    "realized_vol": lambda s: np.log(s).diff().rolling(20, min_periods=15).std() * np.sqrt(252),
    "drawdown": lambda s: 1 - s / s.rolling(252, min_periods=20).max(),
    "below_ma": lambda s: 1 - s / s.rolling(125, min_periods=60).mean(),
    "ratio_drawdown": lambda a, b: (lambda a, b: 1 - (a / b) / (a / b).rolling(252, min_periods=20).max())(*_align(a, b)),
    "fall_20d": lambda s: -_ret(s, 20),
    "drop_20d": lambda s: -(s - s.shift(20)),
    # inputs [x, y]：x 跑输 y 越多越恐慌
    "underperform_20d": lambda x, y: (lambda x, y: _ret(y, 20) - _ret(x, 20))(*_align(x, y)),
    "turnover_shrink": lambda s: -np.log(s.rolling(20, min_periods=15).mean() / s.rolling(250, min_periods=120).mean()),
}


def raw_components(series):
    """返回 DataFrame：每列一个成分的原始值（各自交易日历）。缺输入的成分跳过。"""
    cols = {}
    for cid, spec in config.COMPONENTS.items():
        inputs = [series.get(k) for k in spec["inputs"]]
        if any(s is None or s.empty for s in inputs):
            continue
        cols[cid] = TRANSFORMS[spec["transform"]](*inputs).replace([np.inf, -np.inf], np.nan).dropna()
    return cols


def rolling_pct_rank(s, window=None, min_periods=None):
    """当天值在过去 window 个观测里的分位（含当天），0–100。"""
    window = window or config.RANK_WINDOW
    min_periods = min_periods or config.RANK_MIN_PERIODS
    return s.rolling(window, min_periods=min_periods).rank(pct=True) * 100


def component_scores(series, calendar=None):
    raw = raw_components(series)
    scores = {cid: rolling_pct_rank(s).dropna() for cid, s in raw.items()}
    # 历史不足 RANK_MIN_PERIODS 的成分暂不出分
    scores = {cid: s for cid, s in scores.items() if len(s)}
    raw = {cid: raw[cid] for cid in scores}
    if calendar is None:
        last = max(s.index[-1] for s in scores.values())
        first = min(s.index[0] for s in scores.values())
        calendar = pd.bdate_range(first, last)
    # 各市场节假日不同：先并入统一日历，再有限向前填充
    df = pd.DataFrame({cid: s.reindex(s.index.union(calendar)).ffill(limit=config.FFILL_LIMIT).reindex(calendar)
                       for cid, s in scores.items()})
    raw_df = pd.DataFrame({cid: s.reindex(s.index.union(calendar)).ffill(limit=config.FFILL_LIMIT).reindex(calendar)
                           for cid, s in raw.items()})
    return df, raw_df


def _weighted(df, weights, min_coverage):
    w = pd.Series(weights).reindex(df.columns).fillna(0.0)
    avail = df.notna().mul(w, axis=1)
    covered = avail.sum(axis=1)
    total = df.fillna(0).mul(w, axis=1).sum(axis=1) / covered.replace(0, np.nan)
    total[covered < min_coverage * w.sum()] = np.nan
    return total


def region_scores(comp):
    out = {}
    for region in list(config.REGIONS) + ["global"]:
        ids = [c for c, s in config.COMPONENTS.items() if s["region"] == region]
        weights = {c: config.COMPONENTS[c]["weight"] for c in ids}
        cols = [c for c in ids if c in comp.columns]
        all_w = sum(weights.values())
        sub = comp[cols] if cols else pd.DataFrame(index=comp.index)
        # 覆盖率按区域全部成分的权重算，避免缺失成分时少数成分独撑
        avail = sub.notna().mul(pd.Series(weights).reindex(cols), axis=1).sum(axis=1)
        val = _weighted(sub, weights, 0) if cols else pd.Series(np.nan, index=comp.index)
        val[avail < config.MIN_REGION_COVERAGE * all_w] = np.nan
        out[region] = val
    return pd.DataFrame(out)


def global_score(reg):
    regional = _weighted(reg[list(config.REGIONS)], config.REGION_WEIGHTS, config.MIN_REGION_COVERAGE)
    common = reg["global"]
    g = regional * (1 - config.COMMON_WEIGHT) + common * config.COMMON_WEIGHT
    return g.where(common.notna(), regional)


def smooth(df, span=None):
    span = span or config.SMOOTH_SPAN
    if span <= 1:
        return df
    return df.ewm(span=span, adjust=False, ignore_na=True).mean().where(df.notna())


def build(series):
    """series: dict key->Series。返回 (index_df, component_score_df, component_raw_df)。

    index_df 列：global, us, eu, jp, cn, hk, common
    """
    comp, raw = component_scores(series)
    reg = region_scores(comp)
    idx = reg[list(config.REGIONS)].copy()
    idx.insert(0, "global", global_score(reg))
    idx["common"] = reg["global"]
    idx = smooth(idx).round(1)
    idx = idx.dropna(how="all")
    return idx, comp.loc[idx.index].round(1), raw.loc[idx.index]
