"""拉中国宏观月度数据写 data/raw/cn_macro.csv（公开免密钥接口，只在 GitHub Actions 上跑）。"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astock import cn_macro, config  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df, check = cn_macro.fetch()
    cn_macro.write(df)
    info = dict(check, first={c: str(df[c].dropna().index.min().date()) for c in df},
                last={c: str(df[c].dropna().index.max().date()) for c in df})
    (config.REPORT_DIR / "cn_macro_check.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(json.dumps(info, ensure_ascii=False, indent=1))
    print(df.tail(6).to_string())


if __name__ == "__main__":
    main()
