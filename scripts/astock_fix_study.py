"""选股改进对照实验：只用本地行情缓存，写 reports/astock/fix_study.md 和 docs/astock/fix_study.json。

  python scripts/astock_fix_study.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import data, fix_study  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not data.KINDS["kline"][1].exists():
        raise SystemExit("no price cache")
    res = fix_study.run(data.load_kline, data.load_basic(), data.load_valuation)
    fix_study.write(res)
    print(fix_study.to_markdown(res))


if __name__ == "__main__":
    main()
