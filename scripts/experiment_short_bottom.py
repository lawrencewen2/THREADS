"""试验：“短期触底”提示。不要求分数跌到 20 以下，只要求短期内明显下跌后开始回升。

定义：过去 window 个交易日里，分数从高点下跌至少 drop 分到低点（低点不高于 level），
之后从低点回升至少 rebound 分的那天触发。同一区域两次至少间隔 20 个交易日。

评价（价格用各区域主指数）：
- 守住低点率：触发后 20 个交易日内，价格没有跌破触发前 window 日内的最低价；
- 之后 5/20 日平均涨跌，与所有交易日的平均相比；
- 触发时价格已从低点反弹了多少（越小越及时）；每个区域每年触发几次。
只依据 2018 年以前的结果挑选，2019 年以后检验。用法：python scripts/experiment_short_bottom.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fearindex import backtest, compute, config, fetch  # noqa: E402

SPLIT = pd.Timestamp("2019-01-01")
GAP = 20
HOLD = 20


def events(score, window, drop, rebound, level):
    s = score.dropna()
    low = s.rolling(window, min_periods=window).min()
    # 低点之前的高点：低点出现前 window 日内的最高分
    out, last = [], None
    vals = s.values
    for i in range(window, len(s)):
        seg = vals[i - window: i + 1]
        j = int(np.argmin(seg))
        trough = seg[j]
        peak = seg[: j + 1].max()
        if trough > level or peak - trough < drop:
            continue
        if vals[i] - trough >= rebound and vals[i - 1] - trough < rebound:
            if last is None or i - last >= GAP:
                out.append(s.index[i])
                last = i
    return out


def evaluate(idx, px, window, drop, rebound, level):
    rows, base = [], []
    for r in ["global"] + list(config.REGIONS):
        p = px[r].reindex(idx.index).ffill()
        prior_min = p.rolling(window + 1, min_periods=1).min()
        fut_min = p[::-1].rolling(HOLD, min_periods=HOLD).min()[::-1].shift(-1)
        f5, f20 = p.shift(-5) / p - 1, p.shift(-20) / p - 1
        held = fut_min >= prior_min
        bounce = p / prior_min - 1
        for per, f in (("in", lambda d: d < SPLIT), ("out", lambda d: d >= SPLIT)):
            m = f(p.index) & fut_min.notna() & idx[r].notna()
            base.append(dict(region=r, per=per, years=m.sum() / 252, held=held[m].mean(),
                             f5=f5[m].mean(), f20=f20[m].mean()))
        for d in events(idx[r], window, drop, rebound, level):
            if pd.isna(fut_min.get(d)):
                continue
            rows.append(dict(region=r, per="in" if d < SPLIT else "out", date=d, held=bool(held[d]),
                             f5=f5[d], f20=f20[d], bounce=bounce[d]))
    return pd.DataFrame(rows), pd.DataFrame(base)


def summarize(ev, base):
    out = {}
    for per in ("in", "out"):
        g = ev[ev.per == per] if len(ev) else ev
        b = base[base.per == per]
        yrs = b.years.sum()
        out[per] = dict(n=len(g), per_yr=len(g) / yrs * 6 if yrs else np.nan,  # 6 个序列，折算成每区域每年
                        held=g.held.mean() * 100 if len(g) else np.nan, base_held=b.held.mean() * 100,
                        ex5=(g.f5.mean() - b.f5.mean()) * 100 if len(g) else np.nan,
                        ex20=(g.f20.mean() - b.f20.mean()) * 100 if len(g) else np.nan,
                        win20=(g.f20 > 0).mean() * 100 if len(g) else np.nan,
                        bounce=g.bounce.median() * 100 if len(g) else np.nan)
    return out


def main():
    series = fetch.load_all()
    idx, _, _ = compute.build(series)
    px = backtest.price_panel(series, idx.index)
    rows = []
    for window, drop, rebound, level in itertools.product((10, 20), (10, 15, 20, 25), (5, 8, 12), (40, 50)):
        ev, base = evaluate(idx, px, window, drop, rebound, level)
        s = summarize(ev, base)
        row = dict(window=window, drop=drop, rebound=rebound, level=level)
        for per, m in s.items():
            row.update({f"{k}_{per}": v for k, v in m.items()})
        rows.append(row)
        print(row)
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "reports" / "experiments" / "short_bottom_grid.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 300)
    print(df.round(1).sort_values("held_in", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
