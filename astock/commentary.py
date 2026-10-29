"""把一段盘面点评里的方法拆开回测（2026-10-29 用户贴了一段点评：“分析下这段话所运用的方法，加进策略提高年化收益”）。

点评里能用现有数据检验的方法：
- 业绩不及预期就大跌：没有分析师一致预期的数据，用“财报公布后两天相对大盘的涨跌”代替盈利意外（公告反应），
  做成加分因子，或者直接剔除公告后大跌的股票；
- 外盘拖累（三星财报带崩韩股、再影响 A股科技股开盘）：没有韩国指数，用日经 225 当天和标普 500 前一晚的跌幅代替，
  大跌后不买新的；
- 大盘被 5 日线压制：沪深 300 或全市场等权收盘低于 5 日均线时不买新的；
- A股比日韩弱（日经回到布林中轨上方、A股还在下方）：这种时候不买新的；
- 跌时猛跌、涨时不涨：个股层面用“近 60 日大盘下跌那些天相对大盘的表现”做抗跌因子加分。

基准是线上规则（astock/slot_model.json：业绩预期排名 + 情绪开关，最多 20 只，止盈 2 倍 ATR、止损 3 倍 ATR、最长 10 天，
组合净值跌破 20 日均线空仓）。每种做法只改一处，其余不变；只在训练期（2015–2020）挑，2021 至今只检验。
"""

import json
import logging

import numpy as np
import pandas as pd

from . import config, model
from .alt_study import emotion_regime, iter_alt_factors
from .factors import Panel, eligible
from .live_slots import MODEL_FILE
from .slots import GOAL_WIN, simulate, stats
from .stability import overlay
from .winrate import adjusted_ohlc

log = logging.getLogger(__name__)

RAW = config.ROOT / "data" / "raw"
EVENT_JUMP = 0.01  # 由市盈率反推的每股收益（TTM）变化超过 1% 视为新财报
EAR_DAYS = 60  # 公告反应保留 60 个交易日
MISS = -0.05  # 公告后两天跑输大盘 5% 以上视为“不及预期”
DROP = -0.015  # 外盘跌超 1.5% 视为大跌

GATES = [
    ("沪深 300 收盘低于 5 日均线时不买新的", "csi_ma5"),
    ("全市场等权低于 5 日均线时不买新的", "ew_ma5"),
    ("A股弱于日本（沪深 300 低于布林中轨、日经在中轨上方）时不买新的", "rel_weak"),
    ("日经 225 当天跌超 1.5% 时不买新的", "n225_drop"),
    ("标普 500 前一晚跌超 1.5% 时不买新的", "spx_drop"),
]
FACTORS = [
    ("加分：财报公布后两天相对大盘的表现（权重 0.15）", "ear", 0.15),
    ("加分：财报公布后两天相对大盘的表现（权重 0.3）", "ear", 0.30),
    ("剔除：近 20 天财报公布后跑输大盘 5% 以上的", "miss", None),
    ("加分：近 60 日大盘下跌日的抗跌程度（权重 0.15）", "resist", 0.15),
    ("加分：近 60 日大盘下跌日的抗跌程度（权重 0.3）", "resist", 0.30),
]


def load_index(name, dates):
    path = RAW / f"{name}.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["value"].astype(float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(dates)).ffill().reindex(dates)


def stock_signals(p, elig):
    """个股的公告反应和抗跌程度，都只用 t 收盘前的数据。"""
    ret = (p.close / p.preclose - 1).where(p.traded)
    mkt = ret.where(elig).mean(axis=1)
    ex = ret.sub(mkt, axis=0)
    pe = p.valuation.get("peTTM") if p.valuation is not None else None
    ear = miss = None
    if pe is not None:
        eps = (p.close / pe.where(pe.abs() > 1e-6)).ffill(limit=5)
        jump = (eps / eps.shift(1) - 1).abs() > EVENT_JUMP
        two = ex.fillna(0) + ex.shift(-1).fillna(0)  # 公告日和下一天
        # 第 d 行只放到 d+1 收盘才知道的值：把事件日的两天反应挪到 d+1 那行
        ev = two.where(jump).shift(1)
        ear = ev.ffill(limit=EAR_DAYS)
        miss = (ev < MISS).astype(float).where(ev.notna()).ffill(limit=20).fillna(0) > 0
    down = mkt < 0
    d = ex.where(down.values[:, None] & ex.notna().values)
    resist = d.rolling(60, min_periods=10).sum() / d.notna().rolling(60, min_periods=10).sum().replace(0, np.nan)
    return ear, miss, resist, mkt


def gates(kind, p, mkt):
    """True = 允许买新的。按信号日 t 对齐：t 收盘后、T 开盘前已知。"""
    dates = p.dates
    if kind == "csi_ma5":
        c = load_index("csi300", dates)
        return None if c is None else ~(c < c.rolling(5).mean())
    if kind == "ew_ma5":
        m = (1 + mkt.fillna(0)).cumprod()
        return ~(m < m.rolling(5).mean())
    if kind == "rel_weak":
        c, n = load_index("csi300", dates), load_index("n225", dates)
        if c is None or n is None:
            return None
        return ~((c < c.rolling(20).mean()) & (n > n.rolling(20).mean()))
    if kind in ("n225_drop", "spx_drop"):
        s = load_index(kind[:-5], dates)
        if s is None:
            return None
        return ~(s.pct_change() < DROP)  # 日经 t 日 14:00（北京）收盘；标普 t 日收盘在北京时间 T 日凌晨
    raise ValueError(kind)


def run(kline, basic, valuation=None, forecast=None):
    kline = kline() if callable(kline) else kline
    valuation = valuation() if callable(valuation) else valuation
    forecast = forecast() if callable(forecast) else forecast
    sm = json.loads(MODEL_FILE.read_text())
    weights = {k: v for k, v in sm["weights"].items() if v}
    p = Panel(kline, valuation)
    del kline, valuation
    elig = eligible(p, basic)
    ranks = {}
    for k, v in iter_alt_factors(p, forecast):
        if k in weights:
            ranks[k] = model.xs_rank(v, elig)
        del v
    base = model.score(ranks, weights, elig)
    del ranks
    emotion = emotion_regime(p, elig)
    ohl = adjusted_ohlc(p)
    ear, miss, resist, mkt = stock_signals(p, elig)
    n, tp, sl, hold, share = sm["n"], sm["take_profit"], sm["stop_loss"], sm["max_hold"], sm.get("capital_share", 1.0)
    end = str(p.dates[-1].date())
    periods = dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, end))

    def evaluate(name, sc, gate, **extra):
        row = dict(variant=name, **extra)
        for k, (s, e) in periods.items():
            d, t = simulate(p, ohl, sc, gate, n, tp, sl, hold, s, e)
            d = overlay(d, sm.get("overlay", "trend20"))
            row[k] = stats(d, t)
            half = stats(d * share, t)
            row[k]["half_ann"], row[k]["half_dd"] = half.get("ann_return"), half.get("max_drawdown")
        log.info("%s: train %s test %s", name, row["train"].get("ann_return"), row["test"].get("ann_return"))
        return row

    rows = [evaluate("线上规则（对照）", base, emotion, kind="base")]
    gate_rows, factor_rows = [], []
    for name, kind in GATES:
        g = gates(kind, p, mkt)
        if g is None:
            log.warning("no index data for %s", kind)
            continue
        gate_rows.append(evaluate("线上规则 + " + name, base, emotion & g.reindex(p.dates).fillna(True).astype(bool),
                                  kind=kind, group="gate"))
    for name, kind, w in FACTORS:
        if kind in ("ear", "miss") and ear is None:
            continue
        if kind == "miss":
            sc = base.where(~miss)
        else:
            sc = base + w * model.xs_rank(ear if kind == "ear" else resist, elig).fillna(0)
            sc = sc.where(base.notna())
        factor_rows.append(evaluate("线上规则 + " + name, sc, emotion, kind=kind, weight=w, group="factor"))
    rows += gate_rows + factor_rows

    def ok(r):
        return (r["train"].get("trade_win") or 0) >= GOAL_WIN and r["train"].get("sharpe", -9) > rows[0]["train"].get("sharpe", -9)

    # 组合：训练期各挑一个比对照好的开关和因子，合在一起再测一次
    best_g = max((r for r in gate_rows if ok(r)), key=lambda r: r["train"]["sharpe"], default=None)
    best_f = max((r for r in factor_rows if ok(r)), key=lambda r: r["train"]["sharpe"], default=None)
    if best_g and best_f:
        g = gates(best_g["kind"], p, mkt).reindex(p.dates).fillna(True).astype(bool)
        if best_f["kind"] == "miss":
            sc = base.where(~miss)
        else:
            sc = (base + best_f["weight"] * model.xs_rank(ear if best_f["kind"] == "ear" else resist, elig).fillna(0)).where(base.notna())
        rows.append(evaluate(f"组合：{best_g['variant'][7:]}；{best_f['variant'][7:]}", sc, emotion & g, kind="combo", group="combo"))
    pool = [r for r in rows if ok(r)]
    chosen = max(pool, key=lambda r: r["train"]["sharpe"])["variant"] if pool else rows[0]["variant"]
    return dict(updated=end, rows=rows, chosen=chosen, capital_share=share,
                note="没有韩国指数和分析师一致预期的数据：外盘用日经 225 和标普 500 代替，盈利意外用财报公布后两天的股价反应代替。")


def _p(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def to_markdown(res):
    share = res["capital_share"]
    lines = ["# 盘面点评里的方法：回测", "",
             f"数据截至 {res['updated']}。基准是线上规则（业绩预期 + 情绪开关，最多 20 只，止盈 2 倍 ATR、止损 3 倍 ATR、最长 10 天，"
             "净值跌破 20 日均线空仓）。每种做法只改一处；只用 2015–2020 挑（每笔胜率 ≥ 50% 且夏普高于线上规则），2021 至今只检验。"
             "收益扣佣金、双边滑点和印花税。", "", res["note"], "",
             f"训练期挑出：**{res['chosen']}**" + ("（没有做法在训练期比线上规则好，维持不变）" if res["chosen"].startswith("线上规则（") else ""), "",
             f"“一半资金”两列按线上建议只用 {share * 100:.0f}% 资金折算到总仓位。", "",
             "| 做法 | 训练期年化 | 训练期夏普 | 训练期回撤 | 训练期胜率 | 样本外年化 | 样本外夏普 | 样本外回撤 | 样本外胜率 | 一半资金：样本外年化 | 一半资金：样本外回撤 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in res["rows"]:
        a, t = r["train"], r["test"]
        lines.append(f"| {r['variant']} | {_p(a.get('ann_return'))} | {a.get('sharpe', 0):.2f} | {_p(a.get('max_drawdown'))} | "
                     f"{_p(a.get('trade_win'), 0)} | {_p(t.get('ann_return'))} | {t.get('sharpe', 0):.2f} | {_p(t.get('max_drawdown'))} | "
                     f"{_p(t.get('trade_win'), 0)} | {_p(t.get('half_ann'))} | {_p(t.get('half_dd'))} |")
    years = sorted({y for r in res["rows"] for y in r["test"].get("yearly", {})})
    lines += ["", "## 样本外分年收益", "", "| 做法 | " + " | ".join(years) + " |", "| --- |" + " --- |" * len(years)]
    for r in res["rows"]:
        y = r["test"].get("yearly", {})
        lines.append(f"| {r['variant']} | " + " | ".join(_p(y.get(k)) for k in years) + " |")
    lines += ["", "仅供个人研究参考，不构成投资建议。"]
    return "\n".join(lines) + "\n"


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "commentary.md").write_text(to_markdown(res))
    (config.PAGE_DIR / "commentary.json").write_text(json.dumps(res, ensure_ascii=False, default=str))
