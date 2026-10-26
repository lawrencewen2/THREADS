"""A股选股回测：拉全部历史（有缓存则增量），训练期定参数，样本外检验，写报告和 astock/model.json。

  python scripts/astock_backtest.py            # 拉数据 + 回测
  python scripts/astock_backtest.py --offline  # 只用本地缓存
  python scripts/astock_backtest.py --fetch-only  # 只拉数据（Actions 上先存缓存再回测，回测出错也不丢数据）
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
    ap.add_argument("--fetch-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.offline:
        with data.Session() as sess:
            basic = data.update_basic(sess)
        codes = data.stock_codes(basic)
        logging.info("fetching %d codes", len(codes))
        _, failed = data.update_kline(codes)
        if failed:
            logging.warning("%d codes failed: %s", len(failed), failed[:20])
        _, failed = data.update_kline(codes, kind="valuation")
        if failed:
            logging.warning("%d valuation codes failed: %s", len(failed), failed[:20])
    if args.fetch_only:
        return
    res = backtest.run(data.load_kline, data.load_basic(), data.load_valuation)
    backtest.write(res)
    print(backtest.to_markdown(res))
    print(model.public(res["chosen"]["test"]))


if __name__ == "__main__":
    main()
