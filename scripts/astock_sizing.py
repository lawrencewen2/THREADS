"""动态仓位回测：写 reports/astock/sizing.md 和 docs/astock/sizing.json。

只读行情、估值和业绩预告缓存。分支缓存里业绩预告不够早（2021 年前不到 2,000 条）时，
用 2 个连接把全部业绩预告重拉一次（约 5,400 次 baostock 查询，2026-11-02 拉过一次约 5 分钟、没出问题），
存进本分支缓存，之后的运行直接用。"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import data, sizing  # noqa: E402


def forecast_rows_before(year=2021):
    fc = data.load_forecast()
    if fc is None or not len(fc):
        return 0
    return int((fc["pub"].dt.year < year).sum())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not data.KINDS["kline"][1].exists():
        raise SystemExit("no price cache")
    basic = data.load_basic()
    if "--ensure-forecast" in sys.argv:
        old = forecast_rows_before()
        logging.info("forecast rows before 2021: %d", old)
        if old < 2000:
            df, failed = data.update_forecast(data.stock_codes(basic), workers=2, merge=True)
            logging.info("forecast refetched: %d rows, %d codes failed, before 2021 now %d", len(df), len(failed),
                         forecast_rows_before())
        return
    logging.info("forecast rows before 2021: %d", forecast_rows_before())
    res = sizing.run(data.load_kline, basic, data.load_valuation, data.load_forecast)
    sizing.write(res)
    print(sizing.to_markdown(res))


if __name__ == "__main__":
    main()
