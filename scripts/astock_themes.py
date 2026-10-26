"""主线行情：读行情缓存和概念成分，写 docs/astock/themes.json。

  python scripts/astock_themes.py                    # 先更新概念成分，再算主线
  python scripts/astock_themes.py --offline          # 用已有的 concepts.csv
  python scripts/astock_themes.py --history 2025-12-31 2026-03-31   # 回看这些日子排出的主线和之后 20 天表现
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import data, news, themes  # noqa: E402
from astock.factors import Panel  # noqa: E402

DAYS = 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--history", nargs="*", default=[])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.offline:
        themes.update_members()
    if not themes.CONCEPT_FILE.exists():
        raise SystemExit("no concepts.csv")
    kline = data.load_kline()
    if kline is None:
        raise SystemExit("no price cache")
    kline = kline[kline["date"] >= kline["date"].max() - pd.Timedelta(days=DAYS + 200 * bool(args.history))]
    p = Panel(kline)
    names = data.load_basic().set_index("code")["code_name"]
    members = themes.load_members()
    out = themes.build(p, members, names, news.LAST_RAW or None)
    themes.write(out)
    print(json.dumps({k: out[k] for k in ("signal_date", "concepts", "themes")}, ensure_ascii=False, indent=1))
    if args.history:
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 20)
        print(themes.history_check(p, members, names, args.history).to_string())


if __name__ == "__main__":
    main()
