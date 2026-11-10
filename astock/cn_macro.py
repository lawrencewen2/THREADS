"""中国宏观月度数据（2026-11-10 用户：没测到就找能授权的数据源，注意安全）。

只用公开、免密钥的只读接口，不需要任何账号或 token，也不加新的依赖：
- 东方财富数据中心（转载国家统计局 / 人民银行）：制造业和非制造业 PMI、M1 / M2 同比、CPI / PPI 同比、新增人民币贷款；
- 商务部数据中心：社会融资规模增量；
- 国家统计局官网 easyquery：制造业 PMI，用来和东方财富的核对（境外 IP 有时连不上，连不上就跳过核对）。
只在 GitHub Actions 上跑（云端容器连不上这些网站）。结果写 data/raw/cn_macro.csv（一行一个月）。

公布日期按保守的口径算（之前用不到）：PMI 统计局在月末最后一天 9:30 公布，按下一个交易日起可用；
CPI / PPI 一般次月 9–10 日，按次月 12 日起可用；M1 / M2、贷款、社融一般次月 10–15 日，按次月 18 日起可用。
"""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

FILE = config.ROOT / "data" / "raw" / "cn_macro.csv"
EM_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
UA = {"User-Agent": "Mozilla/5.0"}
EM_REPORTS = {
    # reportName: {接口字段: 我们的列名}
    "RPT_ECONOMY_PMI": {"MAKE_INDEX": "pmi", "NMAKE_INDEX": "pmi_non"},
    "RPT_ECONOMY_CURRENCY_SUPPLY": {"BASIC_CURRENCY_SAME": "m2_yoy", "CURRENCY_SAME": "m1_yoy"},
    "RPT_ECONOMY_PPI": {"BASE_SAME": "ppi_yoy"},
    "RPT_ECONOMY_CPI": {"NATIONAL_SAME": "cpi_yoy"},
    "RPT_ECONOMY_RMB_LOAN": {"RMB_LOAN": "loan_new"},
}
# 列名 -> (次月第几天起可用；0 = 当月月末的下一天)
AVAILABLE = {"pmi": 0, "pmi_non": 0, "cpi_yoy": 12, "ppi_yoy": 12, "m2_yoy": 18, "m1_yoy": 18, "loan_new": 18, "tsf": 18}


class _Resp:
    def __init__(self, body):
        self.body = body

    def json(self):
        return json.loads(self.body)


def _get(url, tries=4, method="GET", params=None):
    """标准库 urllib 发只读请求（不加第三方依赖）。"""
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    for i in range(tries):
        try:
            req = urllib.request.Request(full, headers=UA, method=method, data=b"" if method == "POST" else None)
            with urllib.request.urlopen(req, timeout=30) as r:
                return _Resp(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            log.warning("%s failed (%d/%d): %s", url, i + 1, tries, exc)
            time.sleep(2 ** i)
    raise RuntimeError(f"{url} unreachable")


def fetch_eastmoney(report, cols):
    params = dict(columns="REPORT_DATE," + ",".join(cols), pageNumber=1, pageSize=2000, sortColumns="REPORT_DATE",
                  sortTypes=-1, source="WEB", client="WEB", reportName=report)
    data = _get(EM_URL, params=params).json()["result"]["data"]
    df = pd.DataFrame(data)
    month = pd.to_datetime(df["REPORT_DATE"]).dt.to_period("M").dt.to_timestamp()
    out = pd.DataFrame({v: pd.to_numeric(df[k], errors="coerce").to_numpy() for k, v in cols.items()}, index=month)
    return out[~out.index.duplicated()].sort_index()


def fetch_tsf():
    r = _get("https://data.mofcom.gov.cn/datamofcom/front/gnmy/shrzgmQuery", method="POST")
    df = pd.DataFrame(r.json())
    month = pd.to_datetime(df["date"].astype(str).str.replace(r"\D", "", regex=True).str[:6], format="%Y%m")
    s = pd.Series(pd.to_numeric(df["tiosfs"], errors="coerce").to_numpy(), index=month, name="tsf")
    return s[~s.index.duplicated()].sort_index()


def fetch_nbs_pmi():
    """国家统计局 easyquery：近 13 个月制造业 PMI（A0B0101），用来核对。"""
    params = dict(m="QueryData", dbcode="hgyd", rowcode="zb", colcode="sj", wds="[]",
                  dfwds='[{"wdcode":"zb","valuecode":"A0B0101"},{"wdcode":"sj","valuecode":"LAST13"}]',
                  k1=str(int(time.time() * 1000)))
    js = _get("https://data.stats.gov.cn/easyquery.htm", tries=2, params=params).json()
    rows = {}
    for node in js["returndata"]["datanodes"]:
        sj = next(w["valuecode"] for w in node["wds"] if w["wdcode"] == "sj")
        if node["data"]["hasdata"]:
            rows[pd.Timestamp(f"{sj[:4]}-{sj[4:6]}-01")] = float(node["data"]["data"])
    return pd.Series(rows).sort_index()


def validate(df):
    """范围检查：数值离谱或月份断档太多就报错，不写文件。"""
    problems = []
    rng = dict(pmi=(35, 60), pmi_non=(25, 65), m2_yoy=(0, 35), m1_yoy=(-15, 50), ppi_yoy=(-15, 20), cpi_yoy=(-5, 10))
    for c, (lo, hi) in rng.items():
        if c not in df:
            problems.append(f"{c} missing")
            continue
        s = df[c].dropna()
        bad = s[(s < lo) | (s > hi)]
        if len(bad):
            problems.append(f"{c} out of range: {bad.to_dict()}")
        if len(s) and s.index.min() > pd.Timestamp("2013-01-01"):
            problems.append(f"{c} starts only at {s.index.min().date()}")
        recent = s.loc["2014-01-01":]
        gaps = len(pd.date_range(recent.index.min(), recent.index.max(), freq="MS")) - len(recent) if len(recent) else 99
        if gaps > 3:
            problems.append(f"{c} has {gaps} missing months since 2014")
    # 统计局官网境外 IP 常被拒（403），所以再用几个公开的历史值核对口径（统计局 / 人民银行公布数）
    for (c, m), v in ANCHORS.items():
        got = df[c].get(pd.Timestamp(m)) if c in df else None
        if got is None or not np.isfinite(got) or abs(got - v) > 0.051:
            problems.append(f"{c} {m}: got {got}, official {v}")
    return problems


ANCHORS = {("pmi", "2020-02-01"): 35.7, ("pmi", "2024-12-01"): 50.1, ("pmi_non", "2020-02-01"): 29.6,
           ("m2_yoy", "2023-12-01"): 9.7, ("m1_yoy", "2023-12-01"): 1.3, ("cpi_yoy", "2023-12-01"): -0.3,
           ("ppi_yoy", "2023-12-01"): -2.7}


def fetch():
    parts = []
    for report, cols in EM_REPORTS.items():
        parts.append(fetch_eastmoney(report, cols))
        log.info("%s: %d months", report, len(parts[-1]))
    try:
        parts.append(fetch_tsf().to_frame())
        log.info("tsf: %d months", len(parts[-1]))
    except Exception as exc:  # 社融只是附加，拿不到不影响其他
        log.warning("tsf unavailable: %s", exc)
    df = pd.concat(parts, axis=1).sort_index()
    df.index.name = "month"
    problems = validate(df)
    check = dict(nbs="skipped")
    try:
        nbs = fetch_nbs_pmi()
        both = pd.concat([df["pmi"], nbs.rename("nbs")], axis=1).dropna()
        diff = (both["pmi"] - both["nbs"]).abs()
        check = dict(nbs="ok", months=len(both), max_abs_diff=float(diff.max()) if len(diff) else None)
        if len(diff) and diff.max() > 0.051:
            problems.append(f"PMI differs from NBS: {diff[diff > 0.051].to_dict()}")
    except Exception as exc:
        log.warning("NBS cross-check skipped: %s", exc)
    if problems:
        raise RuntimeError("; ".join(problems))
    return df, check


def write(df):
    FILE.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(FILE)


def load():
    try:
        return pd.read_csv(FILE, parse_dates=["month"]).set_index("month").sort_index()
    except FileNotFoundError:
        return None


def available_series(df, col):
    """把月度值挂到“公布后可用”的日期上（保守口径）。"""
    s = df[col].dropna()
    if not len(s):
        return s
    nxt = s.index + pd.offsets.MonthBegin(1)
    day = AVAILABLE.get(col, 18)
    idx = nxt if day == 0 else nxt + pd.Timedelta(days=day - 1)
    return pd.Series(s.to_numpy(), index=idx).sort_index()


def expanding_z(s, min_periods=24):
    """只用当时及以前的数据算 z 分数（扩展窗口），没有偷看。"""
    m = s.expanding(min_periods).mean()
    sd = s.expanding(min_periods).std()
    return (s - m) / sd.replace(0, np.nan)


def indicators(df):
    """月度指标（按月份），之后再挂到公布日期上。"""
    out = {}
    if "pmi" in df:
        out["pmi"] = df["pmi"]
        out["pmi_chg"] = df["pmi"] - df["pmi"].shift(3)
    if {"m1_yoy", "m2_yoy"} <= set(df):
        sc = df["m1_yoy"] - df["m2_yoy"]
        out["m1m2"] = sc
        out["m1m2_chg"] = sc - sc.shift(3)
        out["m2_chg"] = df["m2_yoy"] - df["m2_yoy"].shift(6)
    if "tsf" in df and df["tsf"].notna().sum() > 36:
        s12 = df["tsf"].rolling(12, min_periods=12).sum()
        out["credit"] = s12 / s12.shift(12) - 1  # 社融 12 个月累计同比（信贷脉冲）
    if "ppi_yoy" in df:
        out["ppi_chg"] = df["ppi_yoy"] - df["ppi_yoy"].shift(3)
    return out


def multipliers(dates, lo=0.5, hi=1.5):
    """各宏观指标的仓位系数：扩展窗口 z 分数 × 0.25 + 1（经济/流动性越好仓位越高），按公布日对齐到交易日。"""
    df = load()
    if df is None:
        return {}
    avail_day = dict(pmi="pmi", pmi_chg="pmi", m1m2="m1_yoy", m1m2_chg="m1_yoy", m2_chg="m2_yoy", credit="tsf",
                     ppi_chg="ppi_yoy")
    out = {}
    for k, s in indicators(df).items():
        z = expanding_z(s.dropna())
        tmp = pd.DataFrame({avail_day[k]: z})
        a = available_series(tmp, avail_day[k])
        a = a[~a.index.duplicated(keep="last")]
        m = np.clip(1 + 0.25 * a, lo, hi)
        aligned = m.reindex(m.index.union(dates)).ffill().reindex(dates)
        out["cn_" + k] = aligned.fillna(1.0).astype(float)
    return out
