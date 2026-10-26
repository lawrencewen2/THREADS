"""选股面板的决策辅助板块：读行情缓存和当天选股结果，写 docs/astock/insight.json 和 news.json。

  python scripts/astock_insight.py              # 拉一次行业分类（baostock），其余用本地缓存
  python scripts/astock_insight.py --offline    # 全部用本地文件
  python scripts/astock_insight.py --fetch-days 500  # 没有行情缓存时先拉近 500 天行情和估值（单独测试用）
  python scripts/astock_insight.py --news-only  # 只更新行情新闻
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import config, data, insight, news  # noqa: E402


def update_news(out):
    """行情新闻出错只记日志，不让整个步骤失败。"""
    picks_file = config.PAGE_DIR / "picks.json"
    picks = json.loads(picks_file.read_text()) if picks_file.exists() else None
    if out is None and insight.OUT_FILE.exists():
        out = json.loads(insight.OUT_FILE.read_text())
    try:
        res = news.build(picks, (out or {}).get("sectors"))
        news.write(res)
        print(json.dumps(res["status"], ensure_ascii=False))
    except Exception as exc:
        logging.exception("news failed: %s", exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--fetch-days", type=int, default=0)
    ap.add_argument("--news-only", action="store_true", help="只更新行情新闻（用已有的选股清单和 insight.json）")
    ap.add_argument("--no-news", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.news_only:
        update_news(None)
        return
    if not args.offline:
        with data.Session() as sess:
            basic = data.update_basic(sess)
            ind = insight.clean_industry(sess.query("query_stock_industry"))
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        ind.to_csv(insight.INDUSTRY_FILE, index=False)
        if args.fetch_days:
            start = (pd.Timestamp.today() - pd.Timedelta(days=args.fetch_days)).strftime("%Y-%m-%d")
            codes = data.stock_codes(basic, listed_only=True)
            data.update_kline(codes, start=start)
            data.update_kline(codes, start=start, kind="valuation")
    kline = data.load_kline()
    if kline is None:
        raise SystemExit("no price cache; run with --fetch-days")
    scores = config.PAGE_DIR / "scores.csv"
    ranking = pd.read_csv(scores, dtype={"code": str}) if scores.exists() else None
    picks_file = config.PAGE_DIR / "picks.json"
    picks = json.loads(picks_file.read_text()) if picks_file.exists() else None
    bt_file = config.PAGE_DIR / "backtest.json"
    bt = json.loads(bt_file.read_text()) if bt_file.exists() else None
    out = insight.build(kline, data.load_basic(), data.load_valuation(), insight.load_industry(), ranking, picks, bt)
    insight.write(out)
    if not args.offline and not args.no_news:
        update_news(out)
    print(json.dumps({k: out[k] for k in ("signal_date", "temperature", "contrarian", "checklist")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
