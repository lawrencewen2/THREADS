"""早盘放量（集合竞价 + 9:30–10:30 成交量）选股回测。在 GitHub Actions 里跑（需要 baostock 和日线缓存）。

从境外 runner 拉 baostock 60 分钟线每只约 10 秒，全市场要十几个小时，所以分片并行：
  python scripts/astock_early_volume.py --shard 3 --shards 8 --out parts/   # 只拉第 3 片，写 parts/hour1_3.pkl.gz
  python scripts/astock_early_volume.py --merge parts/                      # 合并各片进缓存，然后回测
  python scripts/astock_early_volume.py --no-fetch                          # 只用缓存回测
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import data, early_volume  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="只用缓存里的 60 分钟线")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", default="parts")
    ap.add_argument("--merge", help="合并这个目录里的 hour1_*.pkl.gz 再回测")
    ap.add_argument("--sina", action="store_true", help="用新浪 60 分钟线（约一年）快速回测：前一半训练、后一半样本外，结果另存 *_sina")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    basic = data.load_basic()
    if a.shard is not None:
        codes = early_volume.fetch_codes(basic)[a.shard::a.shards]
        path = Path(a.out) / f"hour1_{a.shard}.pkl.gz"
        hour, failed = early_volume.update_hour(codes, path=path)
        logging.info("shard %d: %d codes, %d rows, %d failed", a.shard, len(codes), len(hour), len(failed))
        return
    kline = data.load_kline()
    if kline is None:
        raise SystemExit("no daily kline cache")
    if a.sina:
        codes = [c for c in early_volume.fetch_codes(basic) if c in set(kline["code"].unique())]
        hour, failed = early_volume.fetch_sina(codes)
        logging.info("sina: %d rows, %d codes, %d failed", len(hour), hour["code"].nunique(), len(failed))
        # 成交稀少的股票 1023 根能回溯好几年；只用大多数股票都有数据的日子（约最近一年）
        cnt = hour.groupby("date")["code"].size()
        hour = hour[hour["date"].isin(cnt.index[cnt >= 0.5 * cnt.max()])]
        days = sorted(hour["date"].unique())
        # 前 5 个交易日给 5 日均量预热，剩下的对半分
        mid = pd.Timestamp(days[5 + (len(days) - 5) // 2])
        early_volume.TRAIN_END = str((mid - pd.Timedelta(days=1)).date())
        early_volume.TEST_START = str(mid.date())
        early_volume.HOUR_START = str(pd.Timestamp(days[0]).date())
        early_volume.MIN_TRADES = 150
        for k in ("REPORT_FILE", "PAGE_FILE", "MODEL_FILE"):
            f = getattr(early_volume, k)
            setattr(early_volume, k, f.with_name(f.name.replace("early_volume", "early_volume_sina")))
    elif a.merge:
        files = sorted(Path(a.merge).rglob("hour1_*.pkl.gz"))
        logging.info("merging %d parts", len(files))
        hour = early_volume.merge_parts(files)
    elif a.no_fetch:
        hour = early_volume.load_hour()
    else:
        hour, failed = early_volume.update_hour(early_volume.fetch_codes(basic),
                                                fresh_since=kline["date"].max() - pd.Timedelta(days=4))
        logging.info("hour bars: %d rows, %d codes failed", len(hour), len(failed))
    logging.info("hour bars to %s, %d codes", hour["date"].max().date(), hour["code"].nunique())
    panel = early_volume.build_panel(kline, hour, basic)
    logging.info("panel %d rows", len(panel))
    res = early_volume.run(panel)
    early_volume.write(res)
    print(early_volume.to_markdown(res))


if __name__ == "__main__":
    main()
