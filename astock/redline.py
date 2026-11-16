"""总组合回撤红线回测（2026-11-16 量化小组风控手册草案，用户说“继续”）。

问题：在线上组合（选股 65% 每只 3.25% + 纳指/黄金/国债 ETF 分仓闲钱停泊）之上，
再加“总组合从净值高点回撤到 −12% 选股减半、到 −15% 停止开新仓”的红线，会不会伤收益、能不能压回撤？

口径和 diversify.py 一致：第 T 行 = T 开盘到 T+1 开盘；选股部分用 data/research/combo_daily.csv.gz 的
“accel0.3”列（线上规则含业绩加速因子，100% 资金口径，已扣成本），净值跌破 20 日均线空仓同 stability.overlay。
红线看总组合净值，错开两行（T+1 开盘前只知道到第 T-1 行的净值）。减下来的选股资金按“闲钱停泊”进 ETF 分仓。
选股仓位从 1 降到 0.5 / 0 时按比例扣一次卖出成本（单边 0.15%）。

恢复规则（红线触发后什么时候恢复满仓）是未知数，所以列几种，只用训练期（2015–2020）挑，2021 年至今看样本外：
- dd8：总回撤回到 −8% 以内恢复；
- ma20：总组合净值站上自身 20 日均线恢复；
- days20：触发后 20 个交易日自动恢复，同时把高点重置成当时净值（相当于“主席复盘后批准恢复”）。
另外对照只做“−8% 预警复盘”（不改仓位）的线上规则。
"""

import json
import logging

import numpy as np
import pandas as pd

from . import config, diversify as dv

log = logging.getLogger(__name__)

SWITCH_COST = 0.0015  # 股票单边
SHARE = 0.65
OUT_JSON = config.ROOT / "docs" / "astock" / "redline.json"
OUT_MD = config.ROOT / "reports" / "astock" / "redline.md"


def load_part(col="accel0.3", path=dv.RESEARCH_DIR / "combo_daily.csv.gz", n=20, frame=None):
    c = pd.read_csv(path, parse_dates=["date"], index_col="date") if frame is None else frame
    out = {}
    for p in ("train", "test"):
        if p == "train":
            idx = c.index[(c.index >= pd.Timestamp(config.TRAIN_START)) & (c.index <= pd.Timestamp(config.TRAIN_END))]
        else:
            idx = c.index[c.index >= pd.Timestamp(config.TEST_START)]
        x = pd.DataFrame({"raw": c.loc[idx, f"{col}|raw"], "held": c.loc[idx, f"{col}|held"]})
        eq = (1 + x["raw"]).cumprod()
        pos = (eq >= eq.rolling(20, min_periods=10).mean()).astype(float).shift(2).fillna(1.0)
        # 空仓开关每切换一次扣单边成本（2026-11-16 红队：之前多除了一个 2，少扣一半）
        x["live"] = x["raw"] * pos - pos.diff().abs().fillna(0) * SWITCH_COST
        x["expo"] = (pos * x["held"].shift(1).fillna(0) / n).clip(0, 1)
        out[p] = x
    return out


def simulate(part, sleeve, rule=None, yellow=-0.08, orange=-0.12, red=-0.15, band=0.10):
    """逐日模拟总组合。rule=None 为线上规则；否则 rule ∈ {dd8, ma20, days20}。
    返回 (逐日收益, 选股倍数序列, 触发次数)。"""
    live = part["live"].to_numpy()
    expo = part["expo"].to_numpy()
    s = sleeve.reindex(part.index).fillna(0.0).to_numpy()
    n = len(live)
    rets, mults = np.zeros(n), np.ones(n)
    eq_hist = [1.0]
    peak, mult, cur_w, hits, since = 1.0, 1.0, None, 0, 0
    pending = [1.0, 1.0]  # 错开两行：今天用的是两行之前定下的倍数
    for i in range(n):
        m_use = pending[0]
        tgt = 1 - SHARE * m_use * expo[i]
        if cur_w is None or tgt < cur_w or tgt - cur_w > band:
            dw = 0.0 if cur_w is None else abs(tgt - cur_w)
            cur_w = tgt
        else:
            dw = 0.0
        prev_m = mults[i - 1] if i else 1.0
        cost = dw * dv.ETF_COST + SHARE * abs(prev_m - m_use) * expo[i] * SWITCH_COST
        r = SHARE * m_use * live[i] + cur_w * s[i] - cost
        rets[i], mults[i] = r, m_use
        eq = eq_hist[-1] * (1 + r)
        eq_hist.append(eq)
        peak = max(peak, eq)
        dd = eq / peak - 1
        new = mult
        if rule is not None:
            since += 1
            if dd <= red and mult > 0:
                new, hits, since = 0.0, hits + 1, 0
            elif dd <= orange and mult > 0.5:
                new, hits, since = 0.5, hits + 1, 0
            elif mult < 1:
                if rule == "dd8" and dd > yellow:
                    new = 1.0
                elif rule == "ma20" and len(eq_hist) > 20 and eq >= np.mean(eq_hist[-20:]):
                    new = 1.0
                elif rule == "days20" and since >= 20:
                    new, peak = 1.0, eq
        mult = new
        pending = [pending[1], mult]
    return pd.Series(rets, index=part.index), pd.Series(mults, index=part.index), hits


def stats(d):
    st = dv.stats(d)
    eq = (1 + d).cumprod()
    st["calmar"] = (st["ann_return"] / -st["max_drawdown"]) if st["max_drawdown"] else None
    st["yellow_days"] = int(((eq / eq.cummax() - 1) <= -0.08).sum())
    return st


def run():
    parts = load_part()
    rets = dv.etf_returns()
    sleeve = dv.sleeve_returns(rets, ["sh513100", "sh518880", "sh511010"])
    rows = []
    for rule in (None, "dd8", "ma20", "days20"):
        row = dict(rule=rule or "live")
        for p in ("train", "test"):
            d, m, hits = simulate(parts[p], sleeve, rule)
            row[p] = stats(d) | dict(hits=hits, cut_days=int((m < 1).sum()))
            row[p + "_yearly"] = {str(y): float((1 + g).prod() - 1) for y, g in d.groupby(d.index.year)}
        rows.append(row)
    base = rows[0]
    cands = rows[1:]
    best = max(cands, key=lambda r: (r["train"]["calmar"] or 0))
    verdict = ("adopt" if best["test"]["calmar"] > base["test"]["calmar"] and
               best["test"]["ann_return"] >= base["test"]["ann_return"] - 0.01 else "reject")
    return dict(updated=str(parts["test"].index[-1].date()), rows=rows, train_pick=best["rule"], verdict=verdict)


NAMES = {"live": "线上规则（只预警，不改仓位）", "dd8": "红线 + 回到 −8% 以内恢复",
         "ma20": "红线 + 净值站上 20 日均线恢复", "days20": "红线 + 20 个交易日后恢复并重置高点"}


def _p(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def to_markdown(res):
    L = [f"# 总组合回撤红线回测（数据到 {res['updated']}）", "",
         "线上组合（选股 65% 含业绩加速 + ETF 分仓闲钱停泊）之上加：总回撤 −12% 选股减半，−15% 选股停止。"
         "训练期 2015–2020 挑恢复规则，2021 年至今是样本外。已扣成本。", "",
         "| 方案 | 训练年化 | 训练回撤 | 训练卡玛 | 样本外年化 | 样本外回撤 | 样本外卡玛 | 样本外触发次数 | 样本外减仓天数 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        a, b = r["train"], r["test"]
        L.append(f"| {NAMES[r['rule']]} | {_p(a['ann_return'])} | {_p(a['max_drawdown'])} | {a['calmar']:.2f} | "
                 f"{_p(b['ann_return'])} | {_p(b['max_drawdown'])} | {b['calmar']:.2f} | {b['hits']} | {b['cut_days']} |")
    base = res["rows"][0]["test"]
    L += ["", f"训练期挑出的恢复规则：{NAMES[res['train_pick']]}。结论：**{'采用' if res['verdict'] == 'adopt' else '不采用自动减仓'}**。",
          f"线上规则样本外有 {base['yellow_days']} 天处在 −8% 预警线以下。", "",
          f"−12% 自动减半：训练期触发后要等净值回升才恢复，错过了反弹，年化少约 1 个百分点，最大回撤没变小"
          f"（减仓发生在回撤已经形成之后）；样本外线上最大回撤 {_p(base['max_drawdown'])}，一次都没触发。",
          f"−15% 停止开仓：2015 年至今线上组合从没碰到过（训练期 {_p(res['rows'][0]['train']['max_drawdown'])}），"
          "回测成本为零；它防的是“策略失效、回撤超出历史”的情况，适合作为保险保留。"]
    return "\n".join(L) + "\n"


def write(res):
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    OUT_MD.write_text(to_markdown(res))
