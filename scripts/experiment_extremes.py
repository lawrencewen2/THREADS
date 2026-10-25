"""试验：“触底”和“触顶”仓位信号的触发条件。只依据 2018 年以前的结果挑选，2019 年以后检验。

触底：分数先跌到 low 以下，之后回升到 low + rebound 以上（恐慌见底回升）。
触顶：分数先升到 high 以上，之后回落到 high - drop 以下（贪婪见顶回落）。
同一区域两次信号至少间隔 20 个交易日。

用法：python scripts/experiment_extremes.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fearindex import backtest, compute, config, fetch  # noqa: E402

SPLIT = pd.Timestamp("2019-01-01")
MEMORY = 60   # 极端读数在多少个交易日内仍算“刚出现过”
GAP = 20


def events(score, kind, level, move):
    """返回信号日期。kind = 'bottom' 或 'top'。"""
    s = score.dropna()
    if kind == "bottom":
        armed = (s <= level).astype(int).rolling(MEMORY, min_periods=1).max().astype(bool).shift(1, fill_value=False)
        fire = armed & (s >= level + move) & (s.shift(1) < level + move)
    else:
        armed = (s >= level).astype(int).rolling(MEMORY, min_periods=1).max().astype(bool).shift(1, fill_value=False)
        fire = armed & (s <= level - move) & (s.shift(1) > level - move)
    out, last = [], None
    for d in s.index[fire]:
        if last is None or s.index.get_loc(d) - s.index.get_loc(last) >= GAP:
            out.append(d)
            last = d
    return out


def forward(price, h):
    return price.shift(-h) / price - 1


def fwd_mdd(price, h=60):
    fmin = price[::-1].rolling(h, min_periods=1).min()[::-1].shift(-1)
    return 1 - fmin / price


def evaluate(idx, px, kind, level, move):
    rows = []
    for r in ["global"] + list(config.REGIONS):
        p = px[r].reindex(idx.index)
        f20, f60, mdd = forward(p, 20), forward(p, 60), fwd_mdd(p)
        for d in events(idx[r], kind, level, move):
            rows.append(dict(region=r, date=d, per="in" if d < SPLIT else "out",
                             r20=f20.get(d), r60=f60.get(d), mdd60=mdd.get(d),
                             base20=f20[(p.index < SPLIT) == (d < SPLIT)].mean(),
                             base60=f60[(p.index < SPLIT) == (d < SPLIT)].mean(),
                             basemdd=(mdd[(p.index < SPLIT) == (d < SPLIT)] > 0.10).mean()))
    return pd.DataFrame(rows)


def summarize(ev):
    out = {}
    for per, g in ev.groupby("per"):
        g = g.dropna(subset=["r60"])
        out[per] = dict(n=len(g), r20=g.r20.mean() * 100, r60=g.r60.mean() * 100,
                        ex60=(g.r60 - g.base60).mean() * 100, win60=(g.r60 > 0).mean() * 100,
                        p_mdd10=(g.mdd60 > 0.10).mean() * 100, base_mdd10=g.basemdd.mean() * 100)
    return out


def main():
    series = fetch.load_all()
    idx, _, _ = compute.build(series)
    px = backtest.price_panel(series, idx.index)
    rows = []
    for kind, levels in (("bottom", (10, 15, 20)), ("top", (80, 85, 90))):
        for level, move in itertools.product(levels, (0, 5, 10)):
            s = summarize(evaluate(idx, px, kind, level, move))
            for per, m in s.items():
                rows.append(dict(kind=kind, level=level, move=move, per=per, **m))
    df = pd.DataFrame(rows).set_index(["kind", "level", "move", "per"]).unstack("per")
    pd.set_option("display.width", 250)
    print(df.round(1).to_string())


if __name__ == "__main__":
    main()
