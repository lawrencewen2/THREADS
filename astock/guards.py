"""开盘前清单的三道保险（2026-11-16 用户批准，量化小组第 1 轮提案）。

1. 清单冻结：北京时间 09:15（集合竞价开始）以后，同一个交易日已经有清单时不再覆盖，
   晚到的运行只写 picks_late.json 备查（数据工程发现 GitHub 晚到的定时运行会在盘中改写清单）。
2. 数据新鲜度：信号日的 K 线覆盖率太低、业绩预告太久没更新、恐慌指数日期落后时，清单标红，
   不买新股（继续持有和到期/止损卖出照常）。
3. −15% 红线：模型组合（docs/astock/ledger/nav.csv）从净值高点回撤到 −15% 时停止开新仓，
   状态写进 data/astock/redline_state.json；恢复必须由用户批准后把 halted 改成 false。
   回测 2015 年以来从未触发（reports/astock/redline.md），防的是“策略失效、回撤超出历史”。
"""

import json
import logging

import pandas as pd

from . import config, ledger

log = logging.getLogger(__name__)

FREEZE_HM = (9, 15)
MIN_COVERAGE = 0.9  # 信号日有 K 线的股票 / 近 10 天有 K 线的股票
MAX_FORECAST_AGE = 45  # 天：业绩预告最新公告日离信号日太久说明拉取坏了
MAX_FEAR_LAG = 4  # 天：恐慌指数日期落后信号日
RED_LINE = -0.15
REDLINE_FILE = config.DATA_DIR / "redline_state.json"
LATE_FILE = config.PAGE_DIR / "picks_late.json"


def frozen(trade_date, now, picks_file=None):
    """同一个交易日的清单已经在 09:15 前出过，现在又过了 09:15：不覆盖。"""
    picks_file = picks_file or config.PAGE_DIR / "picks.json"
    if (now.hour, now.minute) < FREEZE_HM or now.normalize() != pd.Timestamp(trade_date).normalize():
        return False
    try:
        old = json.loads(picks_file.read_text())
    except (FileNotFoundError, ValueError):
        return False
    return old.get("trade_date") == str(pd.Timestamp(trade_date).date())


def staleness(kline, forecast, signal_date, fear_date):
    """返回过期原因列表（空 = 正常）。"""
    reasons = []
    sd = pd.Timestamp(signal_date)
    recent = kline[kline["date"] >= sd - pd.Timedelta(days=14)]
    active = recent["code"].nunique()
    today = recent.loc[recent["date"] == sd, "code"].nunique()
    if active and today / active < MIN_COVERAGE:
        reasons.append(f"信号日 K 线覆盖率 {today / active:.0%}（低于 {MIN_COVERAGE:.0%}）")
    if forecast is not None and len(forecast):
        last_pub = pd.to_datetime(forecast["pub"]).max()
        if (sd - last_pub).days > MAX_FORECAST_AGE:
            reasons.append(f"业绩预告最新公告日 {last_pub.date()} 离信号日超过 {MAX_FORECAST_AGE} 天")
    if fear_date and (sd - pd.Timestamp(fear_date)).days > MAX_FEAR_LAG:
        reasons.append(f"恐慌指数日期 {fear_date} 落后信号日 {sd.date()}")
    return reasons


def redline(nav_rows=None, state_file=REDLINE_FILE):
    """读模型组合净值，触到 −15% 就写入 halted 状态。返回状态 dict。已经 halted 的不会自动恢复。"""
    try:
        state = json.loads(state_file.read_text())
    except (FileNotFoundError, ValueError):
        state = {"halted": False}
    rows = ledger.read("nav") if nav_rows is None else nav_rows
    if rows:
        nav = pd.Series([float(r["nav"]) for r in rows])
        dd = float(nav.iloc[-1] / nav.cummax().iloc[-1] - 1)
        state["drawdown"], state["as_of"] = round(dd, 4), rows[-1]["date"]
        if dd <= RED_LINE and not state.get("halted"):
            state.update(halted=True, since=rows[-1]["date"],
                         note="总回撤触到 −15% 红线：停止开新仓。恢复需用户批准后把 halted 改成 false。")
            log.warning("red line hit: drawdown %.3f", dd)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    return state


def block_buys(result, why):
    """不买新股：买入/候补改成观察，继续持有和卖出不变。"""
    result["gate_open"] = False
    result.setdefault("blocked", []).append(why)
    for r in result.get("picks", []):
        if r.get("action") in ("买入", "候补"):
            r["action"], r["selected"] = "观察", False
    result["n_selected"] = sum(1 for r in result.get("picks", []) if r.get("selected"))


def apply(result, kline, forecast, nav_rows=None, state_file=REDLINE_FILE):
    reasons = staleness(kline, forecast, result["signal_date"], result.get("fear_date"))
    result["stale"] = reasons
    if reasons:
        block_buys(result, "数据过期：" + "；".join(reasons))
    rl = redline(nav_rows, state_file)
    result["redline"] = rl
    if rl.get("halted"):
        block_buys(result, f"−15% 红线（{rl.get('since')} 触发，回撤 {rl.get('drawdown')}）")
    return result
