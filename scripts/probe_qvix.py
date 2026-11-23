"""临时探针：optbbs QVIX 是否停更；上交所官方隐含波动率补尾的效果。只在分支上跑，不进 main。"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fearindex import akcompat, fetch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
pd.set_option("display.width", 200)

for url in ("http://1.optbbs.com/d/csv/d/k.csv", "http://1.optbbs.com/d/csv/d/vix50.csv"):
    try:
        r = requests.get(url, timeout=20)
        print(url, r.status_code, {k: v for k, v in r.headers.items() if k.lower() in ("last-modified", "date", "etag", "content-length")})
        lines = r.content.decode("gbk", "replace").strip().splitlines()
        print("  rows", len(lines), "| first:", lines[0][:80], "| last:", [x[:60] for x in lines[-2:]])
    except Exception as exc:
        print(url, "FAILED", type(exc).__name__, str(exc)[:150])

for d in ("20261118", "20261119", "20261120"):
    try:
        df = akcompat.call("option_risk_indicator_sse", date=d)
        sub = df[df["合约交易代码"].astype(str).str.startswith("510050")]
        print(d, "rows", len(df), "50ETF rows", len(sub))
        if d == "20261120":
            print(sub.head(6).to_string())
            print(sorted(sub["合约交易代码"].astype(str).str[6:11].unique()))
            print(repr(sub["合约交易代码"].iloc[0]))
        print("  atm_iv_30d:", fetch.atm_iv_30d(df, pd.Timestamp(d)))
    except Exception as exc:
        print(d, "FAILED", type(exc).__name__, str(exc)[:200])

raw = fetch.load_cached("qvix50")
print("cached qvix tail:\n", raw.tail(4).to_string())
import akshare as ak  # noqa: E402
try:
    live = ak.index_option_50etf_qvix().set_index("date")["close"].dropna()
    print("optbbs daily tail:\n", live.tail(3).to_string())
except Exception as exc:
    print("optbbs daily FAILED", type(exc).__name__, str(exc)[:150])

out = fetch.fetch_akshare("qvix50", "2026-01-01")
print("filled series tail:\n", out.tail(8).to_string())
iv = fetch.load_cached(fetch.SSE_IV_KEY)
q = pd.Series(raw.to_numpy(float), index=pd.to_datetime(raw.index))
both = pd.concat([q, iv], axis=1, keys=["qvix", "sse_iv"]).dropna().tail(fetch.SSE_IV_OVERLAP)
ratio = (both["qvix"] / both["sse_iv"]).median()
both["scaled"] = both["sse_iv"] * ratio
both["err%"] = (both["scaled"] / both["qvix"] - 1) * 100
print(both.round(2).tail(30).to_string())
print("overlap days", len(both), "ratio", round(ratio, 4), "abs err% median/p90/max",
      both["err%"].abs().median().round(2), both["err%"].abs().quantile(0.9).round(2), both["err%"].abs().max().round(2))
print("corr of daily changes", round(float(np.corrcoef(both["qvix"].diff().dropna(), both["scaled"].diff().dropna())[0, 1]), 3),
      "corr of levels", round(float(np.corrcoef(both["qvix"], both["scaled"])[0, 1]), 3))
# 样本外：只用前一半的比值去估后一半
half = len(both) // 2
r0 = (both["qvix"].iloc[:half] / both["sse_iv"].iloc[:half]).median()
e = (both["sse_iv"].iloc[half:] * r0 / both["qvix"].iloc[half:] - 1).abs() * 100
print("out-of-sample (ratio from first half): abs err% median/max", e.median().round(2), e.max().round(2))
print("2026-11-19 minute-feed close was 14.75; SSE fill gives:", out.get(pd.Timestamp("2026-11-19")))
