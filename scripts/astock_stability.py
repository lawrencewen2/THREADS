"""线上规则稳定性 + 胜率检验：重拉业绩预告后，只用本地行情和估值缓存，
写 reports/astock/stability.md、winrate.md 和 docs/astock/stability.json、winrate.json。

  python scripts/astock_stability.py            # 先拉全部历史业绩预告
  python scripts/astock_stability.py --offline  # 用已缓存的业绩预告
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import data, stability, winrate  # noqa: E402


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
    res = stability.run(data.load_kline, basic, data.load_valuation, data.load_forecast)
    stability.write(res)
    print(stability.to_markdown(res))
    del res
    gc.collect()
    res = winrate.run(data.load_kline, basic, data.load_valuation, data.load_forecast)
    winrate.write(res)
    print(winrate.to_markdown(res)[:6000])


if __name__ == "__main__":
    main()
