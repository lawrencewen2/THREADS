"""修正同日腾位后的全量重跑：python scripts/astock_rerun.py（要行情、估值、业绩预告缓存，在 Actions 上跑）。"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import data, rerun  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = rerun.run(data.load_kline, data.load_basic(), data.load_valuation, data.load_forecast)
    rerun.write(res)
    print(rerun.to_markdown(res))


if __name__ == "__main__":
    main()
