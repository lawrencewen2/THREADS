"""主升浪选股 + 市场阶段仓位的回测：只读行情和业绩预告缓存（不重拉，见 baostock 黑名单），
写 reports/astock/mainwave.md 和 docs/astock/mainwave.json。"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import data, mainwave  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not data.KINDS["kline"][1].exists():
        raise SystemExit("no price cache")
    res = mainwave.run(data.load_kline, data.load_basic(), data.load_valuation, data.load_forecast)
    mainwave.write(res)
    print(mainwave.to_markdown(res))


if __name__ == "__main__":
    main()
