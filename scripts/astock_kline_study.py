"""纯 K 线（量价）选股研究：写 reports/astock/kline_study.md、docs/astock/kline_study.json、data/research/kline_daily.csv.gz。"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import config, data, kline_study  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(__file__).resolve().parent.parent / "data" / "research" / "kline_daily.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    industry = pd.read_csv(config.DATA_DIR / "industry.csv", dtype=str).set_index("code")["industry"]
    res = kline_study.run(data.load_kline, data.load_basic(), data.load_valuation, data.load_forecast, industry, dump=out)
    kline_study.write(res)
    print(kline_study.to_markdown(res))


if __name__ == "__main__":
    main()
