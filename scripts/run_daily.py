"""每日任务：拉数据 → 算指数 → 检查 → 发布。失败时以非零退出码结束，让 GitHub Actions 发邮件提醒。"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fearindex import check, compute, fetch, publish  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="重新拉取全部历史")
    ap.add_argument("--offline", action="store_true", help="只用缓存数据，不联网")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.offline:
        series, errors = fetch.load_all(), {}
    else:
        series, errors = fetch.update_all(full=args.full)

    prev_path = publish.DATA / "index.csv"
    previous = pd.read_csv(prev_path, parse_dates=["date"], index_col="date") if prev_path.exists() else None

    index_df, comp_df, raw_df = compute.build(series)
    status = check.run_checks(series, errors, index_df, previous)
    latest = publish.write_data(index_df, comp_df, raw_df, status)
    publish.write_dashboard(index_df, latest)
    publish.update_readme(latest)

    for w in status["warnings"]:
        logging.warning(w)
    for f in status["failures"]:
        logging.error(f)
    g = latest["regions"]["global"]
    logging.info("%s 全球恐慌指数 %s（%s）", latest["date"], g["score"], g["band"])
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
