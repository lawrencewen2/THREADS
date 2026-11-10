"""早盘放量（集合竞价 + 9:30–10:30 成交量）选股回测。在 GitHub Actions 里跑（需要 baostock 和日线缓存）。

python scripts/astock_early_volume.py [--no-fetch]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from astock import data, early_volume  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="只用缓存里的 60 分钟线")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    basic = data.load_basic()
    kline = data.load_kline()
    if kline is None:
        raise SystemExit("no daily kline cache")
    hour = early_volume.load_hour()
    if a.no_fetch:
        logging.info("using cached hour bars (to %s)", None if hour is None else hour["date"].max().date())
    else:
        # 已经拉到近几天的股票跳过；没拉完的（上次超时）接着拉
        hour, failed = early_volume.update_hour(data.stock_codes(basic), fresh_since=kline["date"].max() - pd.Timedelta(days=4))
        logging.info("hour bars: %d rows, %d codes failed", len(hour), len(failed))
    panel = early_volume.build_panel(kline, hour, basic)
    logging.info("panel %d rows", len(panel))
    res = early_volume.run(panel)
    early_volume.write(res)
    print(early_volume.to_markdown(res))


if __name__ == "__main__":
    main()
