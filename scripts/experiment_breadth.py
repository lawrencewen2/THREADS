"""试验：加入等权/中小盘指标，让指数反映多数股票的感受，而不是少数权重股。

对每个区域比较：现行 vs 加“宽基回撤”或“宽基跑输权重指数”成分（权重 0.5 / 1）。
评价同时看两种价格：权重指数（标普500、日经225、沪深300、恒指）和宽基（RSP 等权、TOPIX、中证500、恒生综合）。
只依据 2018 年以前的结果挑选。用法：python scripts/experiment_breadth.py
"""

import copy
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fearindex import backtest, compute, config, fetch  # noqa: E402
import experiment_sensitivity as E  # noqa: E402

BROAD = {"us": "rsp", "jp": "topix", "cn": "csi500", "hk": "hscomp"}
HEAD = {"us": "spx", "jp": "n225", "cn": "csi300", "hk": "hsi"}
CANDIDATES = {
    "us": {"等权回撤": ("drawdown", ["rsp"]), "小盘回撤": ("drawdown", ["iwm"]),
           "等权跑输": ("underperform_20d", ["rsp", "spx"])},
    "jp": {"TOPIX回撤": ("drawdown", ["topix"]), "TOPIX跑输": ("underperform_20d", ["topix", "n225"])},
    "cn": {"中证500回撤": ("drawdown", ["csi500"]), "中证1000回撤": ("drawdown", ["csi1000"]),
           "中证500跑输": ("underperform_20d", ["csi500", "csi300"])},
    "hk": {"恒生综合回撤": ("drawdown", ["hscomp"]), "恒生综合跑输": ("underperform_20d", ["hscomp", "hsi"])},
}


def broad_price(series, key, calendar):
    s = series[key]
    return s.reindex(s.index.union(calendar)).ffill(limit=config.FFILL_LIMIT).reindex(calendar)


def main():
    series = fetch.load_all()
    base = copy.deepcopy(config.COMPONENTS)
    rows = []
    for region, cands in CANDIDATES.items():
        options = [("现行", None, 0)] + [(n, c, w) for (n, c), w in itertools.product(cands.items(), (0.5, 1.0))]
        for name, cand, w in options:
            config.COMPONENTS = copy.deepcopy(base)
            if cand:
                config.COMPONENTS[f"{region}_breadth"] = dict(region=region, transform=cand[0], inputs=cand[1],
                                                              weight=w, name=name)
            idx, _, _ = compute.build(series)
            px = backtest.price_panel(series, idx.index)
            head = E.metrics(idx[region], px[region], E.PERIODS)
            # 宽基价格只在其有数据的区间评价
            bp = broad_price(series, BROAD[region], idx.index)
            sc = idx[region].where(bp.notna())
            broad = E.metrics(sc.dropna(), bp, E.PERIODS)
            row = dict(region=region, variant=name, w=w)
            for per in ("in", "out"):
                row[f"cap_{per}"] = head[per]["cap"]
                row[f"bcap_{per}"] = broad[per]["cap"]
                row[f"nb_{per}"] = broad[per]["n_ev"]
                row[f"lead_{per}"] = head[per]["lead"]
                row[f"blead_{per}"] = broad[per]["lead"]
                row[f"auc_{per}"] = head[per]["auc"]
                row[f"bauc_{per}"] = broad[per]["auc"]
                row[f"false_{per}"] = head[per]["false"]
            last = idx[region].dropna()
            row["10-23"] = round(last.iloc[-1], 1)
            rows.append(row)
            print(row)
    config.COMPONENTS = base
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "reports" / "experiments" / "breadth_grid.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 300)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
