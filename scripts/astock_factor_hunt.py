"""新因子挖掘：写 reports/astock/factor_hunt.md 和 docs/astock/factor_hunt.json。只读行情、估值、业绩预告缓存。"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import data, factor_hunt, insight  # noqa: E402


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
    if "--combos" in sys.argv:
        out = Path(__file__).resolve().parent.parent / "data" / "research" / "combo_daily.csv.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        res = factor_hunt.run(data.load_kline, data.load_basic(), data.load_valuation, data.load_forecast, industry,
                              combos=True, dump=out)
        factor_hunt.write(res, "factor_combos")
        print(factor_hunt.to_markdown(res))
        return
    res = factor_hunt.run(data.load_kline, data.load_basic(), data.load_valuation, data.load_forecast, industry)
    factor_hunt.write(res)
    print(factor_hunt.to_markdown(res))


if __name__ == "__main__":
    main()
