"""A股选股的全部参数。改这里的数字后需重跑回测，模型参数写在 astock/model.json。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "astock"
CACHE_DIR = DATA_DIR / "cache"  # 行情缓存，体积大，不进 git（Actions 用 actions/cache 保存）
MODEL_FILE = Path(__file__).resolve().parent / "model.json"
REPORT_DIR = ROOT / "reports" / "astock"
PAGE_DIR = ROOT / "docs" / "astock"
FEAR_INDEX_FILE = ROOT / "data" / "index.csv"

# 历史数据起点；2014 年的数据只用来给 2015 年初的因子预热
HISTORY_START = "2014-01-01"
TRAIN_START, TRAIN_END = "2015-01-01", "2020-12-31"  # 只在这段定因子权重和筛选门槛
TEST_START = "2021-01-01"  # 之后全部是样本外

# 股票池：全 A股，沪深主板、创业板、科创板（baostock 不含北交所），含已退市股票
BOARD_PREFIXES = ("sh.60", "sh.68", "sz.00", "sz.30")
FETCH_WORKERS = 4  # 并行拉取的 baostock 连接数

# 交易成本（单边比例）
COMMISSION = 0.00025
SLIPPAGE = 0.001
STAMP_DUTY_CHANGE = "2023-08-28"  # 卖出印花税 0.1% 下调到 0.05%
STAMP_DUTY_OLD, STAMP_DUTY_NEW = 0.001, 0.0005

# 可交易过滤
MIN_LISTED_DAYS = 180  # 上市满半年
MIN_AMOUNT_20D = 5e7  # 近 20 日日均成交额至少 5000 万元
MIN_TRADED_DAYS_60 = 50  # 近 60 个交易日至少交易 50 天
DELIST_EXCLUDE_DAYS = 30  # 退市日前 30 天（退市整理期）不选
MAX_HOLD_SCAN = 10  # 卖不出（停牌/跌停开盘）时最多顺延的交易日

# 选股数量、市场环境开关的候选，回测在训练期从中挑
TOP_N_GRID = (1, 3, 5, 10, 20)
FEAR_FLOOR_GRID = (None, 20, 30, 40, 50)  # A股恐慌贪婪指数低于此值的日子不选股
MIN_TRADE_DAY_SHARE = 0.3  # 挑参数时要求至少 30% 的交易日有选股，防止只挑出极少数幸运日
