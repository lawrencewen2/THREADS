"""数据核查：在每日更新之后运行，结果写入 reports/verification.md。只报告，不改数据，总是以 0 退出。"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fearindex import compute, fetch, publish, verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="不做跨数据源比对")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    series = fetch.load_all()
    index_df, comp_df, _ = compute.build(series)
    rep = verify.run(series, index_df, comp_df, online=not args.offline)
    md = verify.to_markdown(rep, index_df.index[-1].date())
    out = publish.ROOT / "reports" / "verification.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(md)
    print(md.split("## 跨数据源比对")[0].split("## 原始数据体检")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
