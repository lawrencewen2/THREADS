"""盘中做T回测：拉线上持仓和随机股票的 5 分钟线，写 reports/astock/t0.md 和 docs/astock/t0.json。"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from astock import data, t0  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not data.KINDS["kline"][1].exists():
        raise SystemExit("no price cache")
    basic = data.load_basic()
    ctx = t0.live_holdings(data.load_kline(), basic, data.load_valuation(), data.load_forecast())
    trades = ctx[2]
    by_year = pd.Series([t["entry"].year for t in trades]).value_counts().sort_index()
    logging.info("live-rule trades by entry year: %s", by_year.to_dict())
    if not args.no_fetch:
        end = str(ctx[3].dates[-1].date())
        _, failed = t0.update_m5(t0.fetch_items(ctx, end))
        logging.info("m5 failed: %d", len(failed))
    if args.fetch_only:
        return
    res = t0.run(ctx, t0.load_m5())
    t0.write(res)
    print(t0.to_markdown(res))


if __name__ == "__main__":
    main()
