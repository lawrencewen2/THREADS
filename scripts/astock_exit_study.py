"""出场规则研究：python scripts/astock_exit_study.py（要行情、估值、业绩预告缓存，在 Actions 上跑）。"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import data, exit_study  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = exit_study.run(data.load_kline, data.load_basic(), data.load_valuation, data.load_forecast)
    exit_study.write(res)
    print(exit_study.to_markdown(res))


if __name__ == "__main__":
    main()
