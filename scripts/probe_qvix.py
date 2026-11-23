"""临时探针：optbbs QVIX 是否停更；上交所官方隐含波动率补尾的效果。只在分支上跑，不进 main。"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fearindex import akcompat, fetch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
pd.set_option("display.width", 200)

import time
t0 = time.monotonic()
for d in ("20261120", "20261009"):
    df = akcompat.call("option_risk_indicator_sse", date=d)
    print(d, "rows", len(df), "atm", fetch.atm_iv_30d(df, pd.Timestamp(d)), "elapsed", round(time.monotonic() - t0, 1))
raw = fetch.load_cached("qvix50")
print("cached qvix tail:\n", raw.tail(4).to_string())
fetch.SSE_IV_BUDGET = 900
t0 = time.monotonic()
iv = fetch._sse_iv_series(pd.Timestamp("2026-08-20"), pd.Timestamp("2026-11-23"))
print("sse series rows", len(iv), "elapsed", round(time.monotonic() - t0, 1))
t0 = time.monotonic()
out = fetch._sse_iv_tail(raw)
print("tail fill elapsed", round(time.monotonic() - t0, 1))
print("filled series tail:\n", out.tail(8).to_string())
iv = fetch.load_cached(fetch.SSE_IV_KEY)
q = pd.Series(raw.to_numpy(float), index=pd.to_datetime(raw.index))
both = pd.concat([q, iv], axis=1, keys=["qvix", "sse_iv"]).dropna().tail(200)
ratio = (both["qvix"] / both["sse_iv"]).median()
both["scaled"] = both["sse_iv"] * ratio
both["err%"] = (both["scaled"] / both["qvix"] - 1) * 100
print(both.round(2).tail(70).to_string())
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
