"""带止盈止损的真实账户模拟：重拉业绩预告后，写 reports/astock/slots.md 和 docs/astock/slots.json。"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import data, slots  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not data.KINDS["kline"][1].exists():
        raise SystemExit("no price cache")
    basic = data.load_basic()
    if not args.offline and data.load_forecast() is None:
        _, failed = data.update_forecast(data.stock_codes(basic))
        if failed:
            logging.warning("%d forecast codes failed", len(failed))
    res = slots.run(data.load_kline, basic, data.load_valuation, data.load_forecast)
    slots.write(res)
    print(slots.to_markdown(res))


if __name__ == "__main__":
    main()
