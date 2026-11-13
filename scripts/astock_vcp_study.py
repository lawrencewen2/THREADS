"""VCP 叠加线上选股：写 reports/astock/vcp_study.md、docs/astock/vcp_study.json、data/research/vcp_daily.csv.gz。"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import data, vcp_study  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "--ensure-forecast" in sys.argv:
        fc = data.load_forecast()
        old = 0 if fc is None else int((fc["pub"].dt.year < 2021).sum())
        logging.info("forecast rows before 2021: %d", old)
        if old < 2000:
            data.update_forecast(data.stock_codes(data.load_basic()), workers=2, merge=True)
        return
    out = Path(__file__).resolve().parent.parent / "data" / "research" / "vcp_daily.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = vcp_study.run(data.load_kline, data.load_basic(), data.load_valuation, data.load_forecast, dump=out)
    vcp_study.write(res)
    print(vcp_study.to_markdown(res))


if __name__ == "__main__":
    main()
