"""A股选股回测：拉全部历史（有缓存则增量），训练期定参数，样本外检验，写报告和 astock/model.json。

  python scripts/astock_backtest.py            # 拉数据 + 回测
  python scripts/astock_backtest.py --offline  # 只用本地缓存
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import backtest, data, model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.offline:
        with data.Session() as sess:
            universe = data.update_universe(sess)
            data.update_basic(sess)
            codes = sorted(universe["code"].unique())
            _, failed = data.update_kline(sess, codes)
            if failed:
                logging.warning("%d codes failed: %s", len(failed), failed[:20])
    res = backtest.run(data.load_kline(), data.load_universe(), data.load_basic())
    backtest.write(res)
    print(backtest.to_markdown(res))
    print(model.public(res["chosen"]["test"]))


if __name__ == "__main__":
    main()
