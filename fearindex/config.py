"""指数定义：原始数据序列、成分指标、区域与权重。

成分内部都调成“数值越高越恐慌”再做滚动分位数标准化，发布前整体翻转：
0 = 极度恐慌，100 = 极度贪婪（与 CNN 恐惧贪婪指数方向一致）。
"""

import numpy as np
import pandas as pd

# 原始序列：key -> [(数据源, 代码), ...]，按顺序尝试，第一个成功的生效。
# 换了数据源（量纲可能不同）时会整段重拉，不与旧缓存拼接。
# yahoo: yfinance 收盘价；akshare: 见 fetch.py 里的专用函数
# 注：FRED 的 ICE 高收益债利差在 Actions 上常超时，且公开历史有限，改用 HYG/IEF 比值衡量信用压力。
SERIES = {
    # 美国
    "spx": [("yahoo", "^GSPC")],
    "vix": [("yahoo", "^VIX")],
    "vix3m": [("yahoo", "^VIX3M")],
    "hyg": [("yahoo", "HYG")],
    "ief": [("yahoo", "IEF")],
    "rsp": [("yahoo", "RSP")],     # 标普500 等权 ETF：反映多数成分股，而不是少数权重股
    "iwm": [("yahoo", "IWM")],     # 罗素2000 小盘 ETF
    # 欧洲
    "sx5e": [("yahoo", "^STOXX50E")],
    "eurchf": [("yahoo", "EURCHF=X")],
    # 日本
    "n225": [("yahoo", "^N225")],
    "topix": [("yahoo", "1306.T"), ("yahoo", "^TOPX")],  # TOPIX（市值加权、覆盖全市场；日经225 为价格加权，受少数高价股左右）
    "usdjpy": [("yahoo", "JPY=X")],
    # A股（东方财富常拒绝境外 IP，依次退到腾讯、新浪、Yahoo）
    "csi300": [("akshare", "em_close:sh000300"), ("akshare", "tx_close:sh000300"),
               ("akshare", "sina_close:sh000300"), ("yahoo", "000300.SS")],
    "a_turnover": [("akshare", "em_amount:sh000001+sz399001"), ("akshare", "tx_amount:sh000001+sz399001"),
                   ("akshare", "sina_volume:sh000001+sz399001")],
    "sh_margin": [("akshare", "sh_margin")],
    "csi1000": [("akshare", "tx_close:sh000852"), ("akshare", "sina_close:sh000852"), ("yahoo", "000852.SS")],
    "csi500": [("akshare", "tx_close:sh000905"), ("akshare", "sina_close:sh000905"), ("yahoo", "000905.SS")],
    "qvix50": [("akshare", "qvix50")],
    # 港股
    "hsi": [("yahoo", "^HSI")],
    "hstech": [("yahoo", "HSTECH.HK"), ("yahoo", "3033.HK")],
    "hscomp": [("yahoo", "^HSCI"), ("yahoo", "2833.HK")],  # 恒生综合指数（覆盖大中小盘）
    # 全球公共
    "gold": [("yahoo", "GC=F")],
    "move": [("yahoo", "^MOVE")],
    "us10y": [("yahoo", "^TNX")],
}

# 成分：id -> 定义
# *_crash（10 日急跌）于 2026-10-25 加入，让指数对急跌反应更快；A股、港股缺少快速反应的成分，权重取 1，
# 其余区域取 0.5。依据只用 2018 年以前的回测（见 scripts/experiment_sensitivity.py）。
#   transform: 见 compute.TRANSFORMS
#   inputs: 用到的原始序列
#   weight: 区域内相对权重（缺失时剩余成分重新归一）
COMPONENTS = {
    # ---- 美国 ----
    "us_vix": dict(region="us", transform="level", inputs=["vix"], weight=1.0, name="VIX"),
    "us_term": dict(region="us", transform="ratio", inputs=["vix", "vix3m"], weight=0.75, name="VIX/VIX3M 期限结构"),
    "us_rvol": dict(region="us", transform="realized_vol", inputs=["spx"], weight=0.75, name="标普500 20日实际波动率"),
    "us_dd": dict(region="us", transform="drawdown", inputs=["spx"], weight=1.0, name="标普500 距52周高点回撤"),
    "us_credit": dict(region="us", transform="ratio_drawdown", inputs=["hyg", "ief"], weight=1.0, name="高收益债相对国债回撤（HYG/IEF）"),
    "us_crash": dict(region="us", transform="fall_10d", inputs=["spx"], weight=0.5, name="标普500 10日急跌"),
    # ---- 欧洲 ----
    "eu_rvol": dict(region="eu", transform="realized_vol", inputs=["sx5e"], weight=1.25, name="欧洲斯托克50 20日实际波动率"),
    "eu_dd": dict(region="eu", transform="drawdown", inputs=["sx5e"], weight=1.0, name="欧洲斯托克50 回撤"),
    "eu_ma": dict(region="eu", transform="below_ma", inputs=["sx5e"], weight=0.75, name="欧洲斯托克50 低于125日均线幅度"),
    "eu_chf": dict(region="eu", transform="fall_20d", inputs=["eurchf"], weight=0.5, name="瑞郎避险升值"),
    "eu_crash": dict(region="eu", transform="fall_10d", inputs=["sx5e"], weight=0.5, name="欧洲斯托克50 10日急跌"),
    # ---- 日本 ----
    "jp_rvol": dict(region="jp", transform="realized_vol", inputs=["n225"], weight=1.25, name="日经225 20日实际波动率"),
    "jp_dd": dict(region="jp", transform="drawdown", inputs=["n225"], weight=1.0, name="日经225 回撤"),
    "jp_ma": dict(region="jp", transform="below_ma", inputs=["n225"], weight=0.75, name="日经225 低于125日均线幅度"),
    "jp_yen": dict(region="jp", transform="fall_20d", inputs=["usdjpy"], weight=0.5, name="日元20日急升"),
    "jp_crash": dict(region="jp", transform="fall_10d", inputs=["n225"], weight=0.5, name="日经225 10日急跌"),
    # 日经225 是价格加权，少数高价股（如迅销、东京电子）影响很大；TOPIX 覆盖全市场，更接近多数股票的感受
    "jp_breadth": dict(region="jp", transform="drawdown", inputs=["topix"], weight=1.0, name="TOPIX 回撤（全市场）"),
    # ---- A股 ----
    "cn_qvix": dict(region="cn", transform="level", inputs=["qvix50"], weight=1.0, name="50ETF 期权波动率 QVIX"),
    "cn_rvol": dict(region="cn", transform="realized_vol", inputs=["csi300"], weight=0.5, name="沪深300 20日实际波动率"),
    "cn_dd": dict(region="cn", transform="drawdown", inputs=["csi300"], weight=1.0, name="沪深300 回撤"),
    "cn_turnover": dict(region="cn", transform="turnover_shrink", inputs=["a_turnover"], weight=0.75, name="两市成交量萎缩"),
    "cn_margin": dict(region="cn", transform="fall_20d", inputs=["sh_margin"], weight=0.75, name="沪市融资余额下降"),
    "cn_crash": dict(region="cn", transform="fall_10d", inputs=["csi300"], weight=1.0, name="沪深300 10日急跌"),
    # ---- 港股 ----
    "hk_rvol": dict(region="hk", transform="realized_vol", inputs=["hsi"], weight=1.25, name="恒指 20日实际波动率"),
    "hk_dd": dict(region="hk", transform="drawdown", inputs=["hsi"], weight=1.0, name="恒指 回撤"),
    "hk_ma": dict(region="hk", transform="below_ma", inputs=["hsi"], weight=0.75, name="恒指 低于125日均线幅度"),
    "hk_tech": dict(region="hk", transform="underperform_20d", inputs=["hstech", "hsi"], weight=0.5, name="恒生科技跑输恒指"),
    "hk_crash": dict(region="hk", transform="fall_10d", inputs=["hsi"], weight=1.0, name="恒指 10日急跌"),
    # ---- 全球公共 ----
    "gl_move": dict(region="global", transform="level", inputs=["move"], weight=1.0, name="MOVE 美债波动率"),
    "gl_gold": dict(region="global", transform="underperform_20d", inputs=["spx", "gold"], weight=0.5, name="黄金跑赢股票"),
    "gl_bond": dict(region="global", transform="drop_20d", inputs=["us10y"], weight=0.5, name="美债收益率急跌（避险买盘）"),
}

REGIONS = {
    "us": "美国",
    "eu": "欧洲",
    "jp": "日本",
    "cn": "A股",
    "hk": "港股",
}

# 全球分 = (1 - COMMON_WEIGHT) * 区域加权 + COMMON_WEIGHT * 全球公共成分
REGION_WEIGHTS = {"us": 0.45, "eu": 0.20, "jp": 0.10, "cn": 0.15, "hk": 0.10}
COMMON_WEIGHT = 0.10

# 标准化参数
RANK_WINDOW = 1260       # 约5年交易日（回测中优于3年窗口）
RANK_MIN_PERIODS = 252   # 至少1年历史才出分
MIN_REGION_COVERAGE = 0.5  # 区域内可用成分权重低于该比例则该区域当天不出分
FFILL_LIMIT = 5          # 节假日最多向前填充的天数
SMOOTH_SPAN = 3          # 最终指数的 EMA 平滑跨度；1 表示不平滑

START_DATE = "2000-01-01"

# 单日坏点过滤：当天偏离前后两天均值超过阈值、而前后两天彼此接近（差距小于阈值一半）时剔除。
# 只用于外汇：Yahoo 2008 年的 USDJPY/EURCHF 有日月颠倒的报价。股指和波动率指数的单日暴跌暴涨是真实行情，不过滤。
SPIKE_FILTER = {"usdjpy": 0.04, "eurchf": 0.04}

# 仓位信号：分数 ≤ RISK_OFF_LEVEL 且低于 RISK_OFF_LOOKBACK 个交易日前（恐慌在加深）→ 建议仓位减半。
# 参数只用 2018 年以前的数据选定（见 reports/backtest.md）。
RISK_OFF_LEVEL = 40
RISK_OFF_LOOKBACK = 20
RISK_OFF_CONFIRM = 2     # 条件连续成立几天才发出信号，减少来回开关

# 极端行情提示（参考，不参与减仓规则；回测显示两者都没有可靠的预测力，统计见 reports/backtest.md）：
#   触底：分数在过去 EXTREME_MEMORY 个交易日内曾 ≤ BOTTOM_LEVEL，之后回升到 BOTTOM_EXIT 以上（恐慌解除）
#   触顶：分数在过去 EXTREME_MEMORY 个交易日内曾 ≥ TOP_LEVEL，之后回落到 TOP_EXIT 以下（贪婪退潮）
# 触发后显示 EXTREME_SHOW 个交易日；同一种提示至少间隔 EXTREME_GAP 个交易日。阈值只依据 2018 年以前的回测选定。
BOTTOM_LEVEL, BOTTOM_EXIT = 20, 50
TOP_LEVEL, TOP_EXIT = 85, 65
EXTREME_MEMORY = 60
EXTREME_SHOW = 5
EXTREME_GAP = 20
# 短期触底（参考）：过去 SHORT_WINDOW 个交易日里分数从高点下跌至少 SHORT_DROP 分、低点不高于 SHORT_LEVEL，
# 之后从低点回升至少 SHORT_REBOUND 分。不要求跌到极度恐慌。参数只依据 2018 年以前的回测选定
# （scripts/experiment_short_bottom.py）：在每区域每年至少约 1 次的方案里，前几名守住低点的比例相差不到 1 个百分点，
# 取触发最早（触发时价格离低点最近）的一个。2018 年以前触发后 20 日内守住前期低点约 65%，平时约 46%。
SHORT_WINDOW = 10
SHORT_DROP = 10
SHORT_REBOUND = 8
SHORT_LEVEL = 50
SIGNAL_TEXT = {"risk_off": "恐慌加深，减仓", "bottom": "恐慌解除（可能触底，仅供参考）",
               "short_bottom": "短期触底（仅供参考）",
               "top": "贪婪退潮（可能触顶，仅供参考）", "normal": "正常"}

BANDS = [
    (20, "极度恐慌"),
    (40, "恐慌"),
    (60, "中性"),
    (80, "贪婪"),
    (101, "极度贪婪"),
]


def band(score):
    for upper, label in BANDS:
        if score < upper:
            return label
    return BANDS[-1][1]


def fingerprint():
    """算法参数指纹。参数变化后历史分数本应改变，健康检查据此跳过“历史值漂移”。"""
    import hashlib
    import json

    params = dict(series=SERIES, components=COMPONENTS, region_weights=REGION_WEIGHTS, common=COMMON_WEIGHT,
                  window=RANK_WINDOW, min_periods=RANK_MIN_PERIODS, coverage=MIN_REGION_COVERAGE,
                  ffill=FFILL_LIMIT, smooth=SMOOTH_SPAN, spike=SPIKE_FILTER, risk_off=(RISK_OFF_LEVEL, RISK_OFF_LOOKBACK, RISK_OFF_CONFIRM),
                  extremes=(BOTTOM_LEVEL, BOTTOM_EXIT, TOP_LEVEL, TOP_EXIT, EXTREME_MEMORY, EXTREME_SHOW, EXTREME_GAP),
                  short=(SHORT_WINDOW, SHORT_DROP, SHORT_REBOUND, SHORT_LEVEL))
    return hashlib.sha1(json.dumps(params, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def risk_off(score):
    """返回布尔序列：True 表示恐慌在加深，建议降低仓位（条件需连续 RISK_OFF_CONFIRM 天成立）。"""
    raw = (score <= RISK_OFF_LEVEL) & (score.diff(RISK_OFF_LOOKBACK) < 0)
    if RISK_OFF_CONFIRM <= 1:
        return raw
    return raw.astype(int).rolling(RISK_OFF_CONFIRM, min_periods=RISK_OFF_CONFIRM).min().fillna(0).astype(bool)


def extreme_events(score, kind):
    """触底（kind='bottom'）、短期触底（'short_bottom'）或触顶（'top'）提示的触发日期列表。"""
    s = score.dropna()
    if kind == "short_bottom":
        fire = _short_bottom_fire(s)
    elif kind == "bottom":
        seen = (s <= BOTTOM_LEVEL).astype(int).rolling(EXTREME_MEMORY, min_periods=1).max().shift(1, fill_value=0) > 0
        fire = seen & (s >= BOTTOM_EXIT) & (s.shift(1) < BOTTOM_EXIT)
    else:
        seen = (s >= TOP_LEVEL).astype(int).rolling(EXTREME_MEMORY, min_periods=1).max().shift(1, fill_value=0) > 0
        fire = seen & (s <= TOP_EXIT) & (s.shift(1) > TOP_EXIT)
    out = []
    for d in s.index[fire]:
        if not out or s.index.get_loc(d) - s.index.get_loc(out[-1]) >= EXTREME_GAP:
            out.append(d)
    return out


def _short_bottom_fire(s):
    vals = s.to_numpy()
    fire = np.zeros(len(s), dtype=bool)
    for i in range(SHORT_WINDOW, len(s)):
        seg = vals[i - SHORT_WINDOW: i + 1]
        j = int(np.argmin(seg))
        trough = seg[j]
        if trough > SHORT_LEVEL or seg[: j + 1].max() - trough < SHORT_DROP:
            continue
        fire[i] = vals[i] - trough >= SHORT_REBOUND and vals[i - 1] - trough < SHORT_REBOUND
    return pd.Series(fire, index=s.index)


def signals(score):
    """每天的仓位信号：risk_off / bottom / short_bottom / top / normal。
    减仓优先，其次是触底，再次是短期触底；提示在触发后显示 EXTREME_SHOW 天。"""
    s = score.dropna()
    out = pd.Series("normal", index=s.index)
    pos = pd.Series(range(len(s)), index=s.index)
    for kind in ("top", "short_bottom", "bottom"):
        for d in extreme_events(s, kind):
            i = pos[d]
            out.iloc[i:i + EXTREME_SHOW] = kind
    out[risk_off(s)] = "risk_off"
    return out
