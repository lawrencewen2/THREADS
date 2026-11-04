"""主升浪选股 + 市场阶段仓位的回测：写 reports/astock/mainwave.md 和 docs/astock/mainwave.json。

默认只读行情和业绩预告缓存。--fetch-forecast 只做一件事：把 2014 年以来全部股票（含退市）的业绩预告重拉一遍存进缓存
（2026-11-02 用户同意为了完整对比线上规则拉一次；约 5,400 次 baostock 查询，用 2 个连接慢慢拉）。"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import data, mainwave  # noqa: E402


def describe():
    for name, df, col in (("kline", data.load_kline(), "date"), ("valuation", data.load_valuation(), "date"),
                          ("forecast", data.load_forecast(), "pub")):
        if df is None or not len(df):
            logging.info("%s: empty", name)
            continue
        by_year = df[col].dt.year.value_counts().sort_index()
        logging.info("%s: %d rows, %s to %s, per year %s", name, len(df), df[col].min().date(), df[col].max().date(),
                     by_year.to_dict())
        del df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-forecast", action="store_true")
    ap.add_argument("--industry", action="store_true", help="只跑线上规则 + 同一行业持股上限")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not data.KINDS["kline"][1].exists():
        raise SystemExit("no price cache")
    basic = data.load_basic()
    if args.fetch_forecast:
        df, failed = data.update_forecast(data.stock_codes(basic), workers=2, merge=True)
        logging.info("forecast refetched: %d rows, %d codes failed", len(df), len(failed))
        describe()
        return
    describe()
    if args.industry:
        res = mainwave.run_industry(data.load_kline, basic, data.load_valuation, data.load_forecast)
        mainwave.write_industry(res)
        print(mainwave.industry_markdown(res))
        return
    res = mainwave.run(data.load_kline, basic, data.load_valuation, data.load_forecast)
    mainwave.write(res)
    print(mainwave.to_markdown(res))


if __name__ == "__main__":
    main()
