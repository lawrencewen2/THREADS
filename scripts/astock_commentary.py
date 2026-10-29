"""盘面点评里的方法回测：重拉业绩预告后，写 reports/astock/commentary.md 和 docs/astock/commentary.json。"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import commentary, data  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not data.KINDS["kline"][1].exists():
        raise SystemExit("no price cache")
    basic = data.load_basic()
    # 每日运行结束后会把业绩预告存进缓存；有缓存就不再全量重拉（2026-10-29 全量重拉太多次被 baostock 拉黑）
    if not args.offline and data.load_forecast() is None:
        _, failed = data.update_forecast(data.stock_codes(basic))
        if failed:
            logging.warning("%d forecast codes failed", len(failed))
    res = commentary.run(data.load_kline, basic, data.load_valuation, data.load_forecast)
    commentary.write(res)
    print(commentary.to_markdown(res))


if __name__ == "__main__":
    main()
