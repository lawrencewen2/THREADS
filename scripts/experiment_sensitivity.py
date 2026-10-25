"""调参试验：各区域加“N 日急跌”成分，以及“减仓”信号防抖。只依据 2018 年以前的结果挑选。

用法：python scripts/experiment_sensitivity.py
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
PRICE = {"us": "spx", "eu": "sx5e", "jp": "n225", "cn": "csi300", "hk": "hsi"}
REGIONS = ["global"] + list(config.REGIONS)

for n in (5, 10, 20):
    compute.TRANSFORMS.setdefault(f"fall_{n}d", lambda s, n=n: -compute._ret(s, n))


def auc(score, label):
    d = pd.DataFrame({"s": score, "y": label}).dropna()
    if d.y.nunique() < 2:
        return np.nan
    r = d.s.rank()
    n1 = d.y.sum()
    n0 = len(d) - n1
    return (r[d.y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fear_entries(score, price, level=30, min_dd=0.05):
    """分数下穿 level 的日子；前 5 日到后 20 日内回撤始终不到 min_dd 记为误报。"""
    s = score.dropna()
    entries = s.index[(s <= level) & (s.shift(1) > level)]
    dd = 1 - price / price.rolling(252, min_periods=20).max()
    out = []
    for d in entries:
        i = price.index.get_loc(d)
        out.append(dd.iloc[max(0, i - 5): i + 21].max() < min_dd)
    return np.array(out, dtype=bool)


def metrics(score, price, mask_fn):
    """score 为发布口径（0 = 恐慌）。"""
    start = score.first_valid_index()
    score = score.loc[start:]
    price = price.reindex(score.index)
    out = {}
    events = backtest.event_capture(score, price)
    dd = 1 - price / price.rolling(252, min_periods=20).max()
    for per, f in mask_fn.items():
        ev = [e for e in events if f(pd.Timestamp(e["peak"]))]
        m = f(score.index)
        fe = fear_entries(score[m], price[m])
        hits = [e["dd_at_hit"] for e in ev if e["captured"] and e["dd_at_hit"] is not None]
        out[per] = dict(
            cap=np.mean([e["captured"] for e in ev]) if ev else np.nan,
            n_ev=len(ev),
            lead=np.median(hits) if hits else np.nan,
            auc=auc(-score[m], (dd[m] > 0.10).astype(int)),
            entries=len(fe),
            false=fe.mean() if len(fe) else np.nan,
        )
    return out


def risk_rule(score, confirm=1, hold=1):
    """减仓信号：≤40 且低于 20 日前；confirm = 连续几天成立才开启，hold = 开启后至少保持几天。"""
    raw = config.risk_off(score).astype(int)
    on = raw.rolling(confirm, min_periods=confirm).min().fillna(0).astype(bool) if confirm > 1 else raw.astype(bool)
    if hold > 1:
        out, left = [], 0
        for v in on.values:
            left = hold if v else max(left - 1, 0)
            out.append(left > 0)
        on = pd.Series(out, index=on.index)
    return on


def rule_perf(score, price, on, mask):
    df = pd.DataFrame({"s": score, "p": price, "on": on}).dropna()
    df = df[mask(df.index)]
    ret = df.p.pct_change()
    expo = (1 - 0.5 * df.on.astype(float)).shift(1)

    def perf(r):
        r = r.dropna()
        eq = (1 + r).cumprod()
        vol = r.std() * np.sqrt(252)
        return (eq.iloc[-1] ** (252 / len(r)) - 1) / vol, (1 - eq / eq.cummax()).max()

    sh, mdd = perf(expo * ret)
    bsh, bmdd = perf(ret)
    switches = df.on.astype(int).diff().abs().sum() / (len(df) / 252)
    return dict(d_sharpe=sh - bsh, d_mdd=bmdd - mdd, switches=switches)


PERIODS = {"in": lambda d: d < SPLIT, "out": lambda d: d >= SPLIT}


def build(series, base, crash):
    """crash: {region: (n, weight)}。"""
    # 基准为不含急跌成分的旧配置
    config.COMPONENTS = {c: s for c, s in copy.deepcopy(base).items() if not c.endswith("_crash")}
    for region, (n, w) in crash.items():
        if n:
            config.COMPONENTS[f"{region}_crash"] = dict(region=region, transform=f"fall_{n}d",
                                                        inputs=[PRICE[region]], weight=w, name=f"{n}日急跌")
    idx, _, _ = compute.build(series)
    return idx


def main():
    series = fetch.load_all()
    base = copy.deepcopy(config.COMPONENTS)
    px = None
    rows = []
    # 1) 各区域单独扫描：窗口 × 权重
    for n, w in [(0, 0)] + list(itertools.product((5, 10, 20), (0.5, 1.0))):
        idx = build(series, base, {r: (n, w) for r in config.REGIONS})
        px = backtest.price_panel(series, idx.index)
        for r in REGIONS:
            m = metrics(idx[r], px[r], PERIODS)
            rows.append(dict(region=r, n=n, w=w, **{f"{k}_{p}": v for p, d in m.items() for k, v in d.items()}))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    cols = ["region", "n", "w", "cap_in", "lead_in", "auc_in", "entries_in", "false_in",
            "cap_out", "lead_out", "auc_out", "entries_out", "false_out"]
    print(df[cols].round(3).to_string(index=False))
    df.to_csv(Path(__file__).resolve().parent.parent / "reports" / "experiments" / "sensitivity_grid.csv", index=False)
    config.COMPONENTS = base


if __name__ == "__main__":
    main()
