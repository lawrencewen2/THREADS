"""另类因子对照实验：拉业绩预告后，只用本地行情和估值缓存，写 reports/astock/alt_study.md 和 docs/astock/alt_study.json。

  python scripts/astock_alt_study.py            # 先拉业绩预告
  python scripts/astock_alt_study.py --offline  # 用已缓存的业绩预告
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import alt_study, config, data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not data.KINDS["kline"][1].exists():
        raise SystemExit("no price cache")
    basic = data.load_basic()
    if not args.offline:
        _, failed = data.update_forecast(data.stock_codes(basic))
        if failed:
            logging.warning("%d forecast codes failed", len(failed))
    industry = pd.read_csv(config.DATA_DIR / "industry.csv", dtype=str).set_index("code")["industry"]
    res = alt_study.run(data.load_kline, basic, data.load_valuation, data.load_forecast, industry)
    alt_study.write(res)
    print(alt_study.to_markdown(res))


if __name__ == "__main__":
    main()
