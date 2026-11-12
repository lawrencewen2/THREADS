"""J Law 方法回测：写 reports/astock/jlaw.md、docs/astock/jlaw.json、data/research/jlaw_daily.csv.gz。只读行情、估值、业绩预告缓存。"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import data, insight, jlaw  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "--ensure-forecast" in sys.argv:
        fc = data.load_forecast()
        old = 0 if fc is None else int((fc["pub"].dt.year < 2021).sum())
        logging.info("forecast rows before 2021: %d", old)
        if old < 2000:
            data.update_forecast(data.stock_codes(data.load_basic()), workers=2, merge=True)
        return
    ind = insight.load_industry()
    industry = None if ind is None else dict(zip(ind["code"], ind["industry"]))
    out = Path(__file__).resolve().parent.parent / "data" / "research" / "jlaw_daily.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = jlaw.run(data.load_kline, data.load_basic(), data.load_valuation, data.load_forecast, industry, dump=out)
    jlaw.write(res)
    print(jlaw.to_markdown(res))


if __name__ == "__main__":
    main()
