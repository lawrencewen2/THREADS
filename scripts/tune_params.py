"""调参诊断：对比滚动窗口、平滑、弱成分权重与减仓规则（只应依据 2018 年以前的结果做选择）。

用法：python scripts/tune_params.py   （需先有 data/raw 缓存）
"""

import copy
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fearindex import backtest, compute, config, fetch  # noqa: E402

SPLIT = pd.Timestamp("2019-01-01")
REGIONS = ["global"] + list(config.REGIONS)
WEAK = {"cn_rvol", "eu_chf", "gl_bond", "gl_gold", "jp_yen"}


def auc(score, label):
    d = pd.DataFrame({"s": score, "y": label}).dropna()
    if d.y.nunique() < 2:
        return np.nan
    r = d.s.rank()
    n1 = d.y.sum()
    n0 = len(d) - n1
    return (r[d.y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate(series, base, window, span, weak_w):
    """AUC：分数区分“已回撤>10%”与“未来60日将回撤>10%”的能力；cap：回撤>15% 事件中分数达到70的比例。"""
    config.RANK_WINDOW, config.SMOOTH_SPAN = window, span
    config.COMPONENTS = copy.deepcopy(base)
    for c in WEAK:
        config.COMPONENTS[c]["weight"] = weak_w
    idx, _, _ = compute.build(series)
    px = backtest.price_panel(series, idx.index)
    rows = {}
    for r in REGIONS:
        p = px[r]
        dd = 1 - p / p.rolling(252, min_periods=20).max()
        fmdd = 1 - p[::-1].rolling(60, min_periods=1).min()[::-1].shift(-1) / p
        for per, m in (("in", idx.index < SPLIT), ("out", idx.index >= SPLIT)):
            rows[(r, per)] = dict(auc_dd=auc(idx[r][m], (dd[m] > 0.1).astype(int)),
                                  auc_fwd=auc(idx[r][m], (fmdd[m] > 0.1).astype(int)))
        start = idx[r].first_valid_index()
        ev = backtest.event_capture(idx[r].loc[start:], p.loc[start:])
        rows[(r, "cap")] = dict(auc_dd=np.mean([e["captured"] for e in ev]), auc_fwd=np.nan)
    return idx, px, pd.DataFrame(rows).T


def rule_grid(idx, px):
    def perf(ret):
        ret = ret.dropna()
        eq = (1 + ret).cumprod()
        vol = ret.std() * np.sqrt(252)
        return (eq.iloc[-1] ** (252 / len(ret)) - 1) / vol, (1 - eq / eq.cummax()).max()

    rows = []
    for r in REGIONS:
        df = pd.DataFrame({"s": idx[r], "p": px[r]}).dropna()
        ret = df.p.pct_change()
        for level, lookback in itertools.product((60, 70, 80), (5, 10, 20)):
            expo = pd.Series(np.where((df.s >= level) & (df.s.diff(lookback) > 0), 0.5, 1.0), index=df.index).shift(1)
            for per, m in (("in", df.index < SPLIT), ("out", df.index >= SPLIT)):
                sh, mdd = perf((expo * ret)[m])
                bsh, bmdd = perf(ret[m])
                rows.append(dict(level=level, lookback=lookback, per=per, d_sharpe=sh - bsh, d_mdd=bmdd - mdd))
    return pd.DataFrame(rows).groupby(["level", "lookback", "per"])[["d_sharpe", "d_mdd"]].mean().unstack("per")


def main():
    series = fetch.load_all()
    base = copy.deepcopy(config.COMPONENTS)
    for window, span, weak_w in itertools.product((756, 1260), (3, 5), (1.0, 0.5)):
        _, _, df = evaluate(series, base, window, span, weak_w)
        ins, outs = df.xs("in", level=1).mean(), df.xs("out", level=1).mean()
        cap = df.xs("cap", level=1).auc_dd.mean()
        print(f"window={window} span={span} weak={weak_w}: IN dd {ins.auc_dd:.3f} fwd {ins.auc_fwd:.3f} | "
              f"OUT dd {outs.auc_dd:.3f} fwd {outs.auc_fwd:.3f} | capture {cap:.2f}")
    idx, px, _ = evaluate(series, base, config.RANK_WINDOW, config.SMOOTH_SPAN, 0.5)
    print(rule_grid(idx, px).round(3).to_string())


if __name__ == "__main__":
    main()
