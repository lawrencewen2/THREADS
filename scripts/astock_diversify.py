"""分散化回测：读 data/research（scripts/astock_diversify_dump.py 和 astock_etf_history.py 导出），
写 reports/astock/diversify.md、diversify_scan.csv、docs/astock/diversify.json。--check-quotes 只检查实盘 ETF 行情能不能拉到。"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import diversify, live_sleeve  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "--check-quotes" in sys.argv:
        got = live_sleeve.fetch_closes(["sh513100", "sh518880", "sh511010"])
        for c, s in got.items():
            print(c, len(s), s.index[-1].date(), s.iloc[-1])
        if len(got) < 3:
            raise SystemExit("ETF quotes missing")
        return
    res = diversify.run()
    diversify.write(res)
    if "--chart" in sys.argv:
        diversify.chart(res, sys.argv[sys.argv.index("--chart") + 1])
    print(diversify.to_markdown(res))


if __name__ == "__main__":
    main()
