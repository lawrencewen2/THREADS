"""回测：基于缓存数据重算历史指数并生成 reports/backtest.md。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fearindex import backtest, compute, fetch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    series = fetch.load_all()
    index_df, _, _ = compute.build(series)
    report = backtest.run(index_df, series)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    stats = backtest.extreme_stats(index_df, series)
    (out / "backtest.md").write_text(backtest.to_markdown(report, index_df) + "\n" + backtest.extremes_markdown(stats))
    summary = {r: dict(capture_rate=v["capture_rate"], false_alarm_rate=v["false_alarm_rate"],
                       events=len(v["events"]), alarms=len(v["alarms"])) for r, v in report.items()}
    (out / "backtest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
