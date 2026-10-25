"""指数定义：原始数据序列、成分指标、区域与权重。

所有成分都调成“数值越高越恐慌”，再做滚动分位数标准化。
"""

# 原始序列：key -> (数据源, 代码)
# yahoo: yfinance 收盘价；fred: FRED 公开 CSV；akshare: 见 fetch.py 里的专用函数
SERIES = {
    # 美国
    "spx": ("yahoo", "^GSPC"),
    "vix": ("yahoo", "^VIX"),
    "vix3m": ("yahoo", "^VIX3M"),
    "us_hy_oas": ("fred", "BAMLH0A0HYM2"),
    # 欧洲
    "sx5e": ("yahoo", "^STOXX50E"),
    "vstoxx": ("yahoo", "^V2TX"),
    "eurchf": ("yahoo", "EURCHF=X"),
    "eu_hy_oas": ("fred", "BAMLHE00EHYIOAS"),
    # 日本
    "n225": ("yahoo", "^N225"),
    "usdjpy": ("yahoo", "JPY=X"),
    # A股
    "csi300": ("akshare", "index_close:sh000300"),
    "a_turnover": ("akshare", "a_turnover"),
    "sh_margin": ("akshare", "sh_margin"),
    "qvix50": ("akshare", "qvix50"),
    # 港股
    "hsi": ("yahoo", "^HSI"),
    "hstech": ("yahoo", "^HSTECH"),
    # 全球公共
    "gold": ("yahoo", "GC=F"),
    "move": ("yahoo", "^MOVE"),
    "us10y": ("yahoo", "^TNX"),
}

# 成分：id -> 定义
#   transform: 见 compute.TRANSFORMS
#   inputs: 用到的原始序列
#   weight: 区域内相对权重（缺失时剩余成分重新归一）
COMPONENTS = {
    # ---- 美国 ----
    "us_vix": dict(region="us", transform="level", inputs=["vix"], weight=1.0, name="VIX"),
    "us_term": dict(region="us", transform="ratio", inputs=["vix", "vix3m"], weight=0.75, name="VIX/VIX3M 期限结构"),
    "us_rvol": dict(region="us", transform="realized_vol", inputs=["spx"], weight=0.75, name="标普500 20日实际波动率"),
    "us_dd": dict(region="us", transform="drawdown", inputs=["spx"], weight=1.0, name="标普500 距52周高点回撤"),
    "us_credit": dict(region="us", transform="level", inputs=["us_hy_oas"], weight=1.0, name="美国高收益债利差"),
    # ---- 欧洲 ----
    "eu_vstoxx": dict(region="eu", transform="level", inputs=["vstoxx"], weight=1.0, name="VSTOXX"),
    "eu_rvol": dict(region="eu", transform="realized_vol", inputs=["sx5e"], weight=1.0, name="欧洲斯托克50 20日实际波动率"),
    "eu_dd": dict(region="eu", transform="drawdown", inputs=["sx5e"], weight=1.0, name="欧洲斯托克50 回撤"),
    "eu_credit": dict(region="eu", transform="level", inputs=["eu_hy_oas"], weight=1.0, name="欧元区高收益债利差"),
    "eu_chf": dict(region="eu", transform="fall_20d", inputs=["eurchf"], weight=0.5, name="瑞郎避险升值"),
    # ---- 日本 ----
    "jp_rvol": dict(region="jp", transform="realized_vol", inputs=["n225"], weight=1.25, name="日经225 20日实际波动率"),
    "jp_dd": dict(region="jp", transform="drawdown", inputs=["n225"], weight=1.0, name="日经225 回撤"),
    "jp_ma": dict(region="jp", transform="below_ma", inputs=["n225"], weight=0.75, name="日经225 低于125日均线幅度"),
    "jp_yen": dict(region="jp", transform="fall_20d", inputs=["usdjpy"], weight=0.75, name="日元20日急升"),
    # ---- A股 ----
    "cn_qvix": dict(region="cn", transform="level", inputs=["qvix50"], weight=1.0, name="50ETF 期权波动率 QVIX"),
    "cn_rvol": dict(region="cn", transform="realized_vol", inputs=["csi300"], weight=1.0, name="沪深300 20日实际波动率"),
    "cn_dd": dict(region="cn", transform="drawdown", inputs=["csi300"], weight=1.0, name="沪深300 回撤"),
    "cn_turnover": dict(region="cn", transform="turnover_shrink", inputs=["a_turnover"], weight=0.75, name="两市成交额萎缩"),
    "cn_margin": dict(region="cn", transform="fall_20d", inputs=["sh_margin"], weight=0.75, name="沪市融资余额下降"),
    # ---- 港股 ----
    "hk_rvol": dict(region="hk", transform="realized_vol", inputs=["hsi"], weight=1.25, name="恒指 20日实际波动率"),
    "hk_dd": dict(region="hk", transform="drawdown", inputs=["hsi"], weight=1.0, name="恒指 回撤"),
    "hk_ma": dict(region="hk", transform="below_ma", inputs=["hsi"], weight=0.75, name="恒指 低于125日均线幅度"),
    "hk_tech": dict(region="hk", transform="underperform_20d", inputs=["hstech", "hsi"], weight=0.5, name="恒生科技跑输恒指"),
    # ---- 全球公共 ----
    "gl_move": dict(region="global", transform="level", inputs=["move"], weight=1.0, name="MOVE 美债波动率"),
    "gl_gold": dict(region="global", transform="underperform_20d", inputs=["spx", "gold"], weight=0.75, name="黄金跑赢股票"),
    "gl_bond": dict(region="global", transform="drop_20d", inputs=["us10y"], weight=0.75, name="美债收益率急跌（避险买盘）"),
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
RANK_WINDOW = 756        # 约3年交易日
RANK_MIN_PERIODS = 252   # 至少1年历史才出分
MIN_REGION_COVERAGE = 0.5  # 区域内可用成分权重低于该比例则该区域当天不出分
FFILL_LIMIT = 5          # 节假日最多向前填充的天数
SMOOTH_SPAN = 3          # 最终指数的 EMA 平滑跨度；1 表示不平滑

START_DATE = "2000-01-01"

BANDS = [
    (20, "极度贪婪"),
    (40, "贪婪"),
    (60, "中性"),
    (80, "恐慌"),
    (101, "极度恐慌"),
]


def band(score):
    for upper, label in BANDS:
        if score < upper:
            return label
    return BANDS[-1][1]
