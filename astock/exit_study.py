"""出场规则研究（2026-11-16 量化小组第 2 轮，研究员）：在修正同日腾位后的口径上，比较几种出场/买入规则。

线上规则（对照）：业绩预期 + 业绩加速 0.3，最多 20 只，止盈 2×ATR、止损 3×ATR、最长 10 个交易日，情绪开关，
净值跌破 20 日均线空仓（_summ 里的 trend20），总组合 = 选股 65% + ETF 分仓闲钱停泊（redline）。
修正后逐笔诊断：到期卖出占 53%（样本外均值 −3.1%、胜率 30%），止盈占 36%（+9.9%），止损占 11%（−13.0%）。

变体（simulate_x 的 rule 字典，可以叠加；空字典 = 线上规则，与 slots.simulate 逐日一致，见测试）：
- trail     移动止盈：碰到 2×ATR 止盈价不卖，改为从持有期最高价回撤 k×ATR 卖（只用前一天及以前的最高价）。参数 k、最长持有。
- extend    续持：到期（scope=expiry）或到期/止盈（scope=both）时，若该股在前一天得分的前 m 名里，不卖，
            按当天价格重新定止盈止损并重新计时（等于“卖出再买回”但不付成本）。参数 m、scope。
- early     持仓不赚钱提前卖：持有满 k 个交易日、当天开盘价 ≤ 买入价就开盘卖（开盘卖的名额当天可以换新股）。参数 k。
- partial   分批止盈：碰到 a×ATR 卖一半，剩下一半按 trail 的回撤 k×ATR 卖。参数 a、k，最长持有 20。
- gap       高开过滤（研究员第 1 轮提案 F）：开盘较前收高开超过 x 当天不买。参数 x。
- breakeven 保本止损：最高价碰到买入价 + b×ATR 后，止损上移到买入价（从第二天起生效）。参数 b。
最后把各族训练期挑出的、训练期好过线上的规则叠在一起（combo），也只按训练期定。

参数只在训练期（2015–2020）挑：每笔胜率 ≥ 50% 里选股层（含 trend20）夏普最高；样本外（2021 至今）只看。
所有结论同时报告标准成本（滑点 0.1%/边）和保守成本（0.2%/边）。判定（预注册）：挑出的规则在两种成本下
样本外总组合年化 ≥ 线上、最大回撤不差于线上 1 个百分点以上，才算“建议采用”。
"""

import json
import logging
import os

import numpy as np
import pandas as pd

from . import config, diversify as dv, redline
from .alt_study import emotion_regime
from .factor_hunt import _summ
from .rerun import scores
from .slots import levels
from .winrate import adjusted_ohlc

log = logging.getLogger(__name__)

# EXIT_SCORE=base 时用不加业绩加速的打分（2026-11-16 红队第2轮建议对照），输出文件加 _noaccel 后缀
SCORE_KEY = os.environ.get("EXIT_SCORE", "accel")
_SFX = "" if SCORE_KEY == "accel" else "_noaccel"
OUT_DAILY = dv.RESEARCH_DIR / f"exit_daily{_SFX}.csv.gz"
OUT_JSON = config.PAGE_DIR / f"exit_study{_SFX}.json"
OUT_MD = config.REPORT_DIR / f"exit_study{_SFX}.md"
SLIP_STD, SLIP_CONS = 0.001, 0.002
GOAL_WIN = 0.50

# 族名 → [(参数说明, rule, 最长持有)]；每个参数 ≤ 4 个取值
GRID = {
    "base": [("线上规则", {}, 10)],
    "trail": [(f"移动止盈 {k}×ATR，最长 {h} 天", dict(trail=k), h) for k in (1.0, 1.5, 2.0, 3.0) for h in (10, 20)],
    "extend": [(f"续持（{'到期' if s == 'expiry' else '到期+止盈'}，前 {m} 名）", dict(extend=m, scope=s), 10)
               for m in (20, 40, 60) for s in ("expiry", "both")],
    "early": [(f"持有 {k} 天仍不赚钱开盘卖", dict(early=k), 10) for k in (3, 4, 5, 7)],
    "partial": [(f"分批：{a}×ATR 卖一半，余下回撤 {k}×ATR 卖，最长 20 天", dict(partial=a, trail=k), 20)
                for a in (1.0, 1.5, 2.0) for k in (1.5, 2.0)],
    "gap": [(f"高开超过 {x:.0%} 不买", dict(gap=x), 10) for x in (0.03, 0.05, 0.07)],
    "breakeven": [(f"浮盈 {b}×ATR 后止损移到买入价", dict(breakeven=b), 10) for b in (1.0, 1.5, 2.0)],
}
FAMILY = {"base": "线上规则", "trail": "移动止盈", "extend": "续持", "early": "持仓不赚钱提前卖",
          "partial": "分批止盈", "gap": "高开过滤", "breakeven": "保本止损", "combo": "组合（各族训练期最优叠加）"}


def simulate_x(p, ohl, sc, gate, n, tp, sl, hold, start, end, rule=None, gapv=None):
    """slots.simulate（修正后口径）的副本，加上 rule 里的出场/买入变体。

    返回 (逐日收益, 逐笔交易, 每天收盘持仓份数)。交易多了 why（tp/sl/expiry/trail/early/be）和 ext（续持次数）。
    rule 为空时与 slots.simulate(same_day_refill=False) 完全一致。"""
    rule = rule or {}
    if rule.get("partial") is not None and rule.get("trail") is None:
        raise ValueError("partial 需要同时给 trail（剩下一半的回撤倍数）")
    trail_k, ext_m, scope = rule.get("trail"), rule.get("extend"), rule.get("scope", "expiry")
    early_k, part_a, gap_max, be_b = rule.get("early"), rule.get("partial"), rule.get("gap"), rule.get("breakeven")
    o, h, l, atr, locked = ohl
    up, _ = p.limit_prices()
    up_v, raw_open, traded = up.to_numpy(), p.open.to_numpy(), p.traded.to_numpy()
    sc_v = sc.to_numpy()
    gate_v = gate.reindex(p.dates).fillna(False).to_numpy(dtype=bool)
    dates = p.dates
    i0 = max(dates.searchsorted(pd.Timestamp(start)), 15)
    i1 = min(dates.searchsorted(pd.Timestamp(end), side="right"), len(dates) - 1)
    buy_cost = config.SLIPPAGE + config.COMMISSION
    stamp_change = pd.Timestamp(config.STAMP_DUTY_CHANGE)
    held = {}  # 列号 -> 持仓状态 dict
    daily, trades, held_n = [], [], []

    def gross(px, entry, sell_cost):
        return px * (1 - sell_cost) / (entry * (1 + buy_cost)) - 1

    def first_target(price, a):
        if part_a is not None:
            return price * (1 + part_a * a)
        return levels(price, a, tp, sl)[0]

    for T in range(i0, i1):
        stamp = config.STAMP_DUTY_NEW if dates[T] >= stamp_change else config.STAMP_DUTY_OLD
        sell_cost = config.SLIPPAGE + config.COMMISSION + stamp
        r = 0.0
        intraday_freed, sold_today = 0, set()
        top = None
        if ext_m:
            row = sc_v[T - 1]
            ok = ~np.isnan(row)
            idx = np.argsort(-np.where(ok, row, -np.inf))[:ext_m]
            top = set(idx[ok[idx]].tolist())
        for j in list(held):
            s = held[j]
            px, at_open, why = None, False, None
            if not locked[T, j] and np.isfinite(o[T, j]):
                a = atr[T - 1, j] if np.isfinite(atr[T - 1, j]) else s["a"]
                if T - s["d0"] >= hold:
                    if top is not None and j in top:  # 续持：按今天开盘价重新定价位、重新计时
                        _, stp = levels(o[T, j], a, tp, sl)
                        s.update(basis=o[T, j], target=first_target(o[T, j], a) if s["w"] == 1.0 else np.inf,
                                 stop=stp, d0=T, a=a, trailing=s["w"] < 1.0, peak=o[T, j] if s["w"] < 1.0 else None,
                                 be=False, ext=s["ext"] + 1)
                    else:
                        px, at_open, why = o[T, j], True, "expiry"
                elif early_k and T - s["d0"] >= early_k and o[T, j] <= s["basis"] and not s["trailing"]:
                    px, at_open, why = o[T, j], True, "early"
                elif l[T, j] <= s["stop"]:
                    px = min(o[T, j], s["stop"])
                    at_open = o[T, j] <= s["stop"]
                    why = "trail" if s["trailing"] else ("be" if s["be"] else "sl")
                elif h[T, j] >= s["target"]:
                    tpx = max(o[T, j], s["target"])
                    if part_a is not None and s["w"] == 1.0:  # 分批：先卖一半，剩下转移动止盈
                        r += 0.5 * (tpx / o[T, j] - 1 - sell_cost) / n
                        s["realized"] += 0.5 * gross(tpx, s["entry"], sell_cost)
                        s.update(w=0.5, target=np.inf, trailing=True, peak=h[T, j])  # partial 必须配 trail
                    elif trail_k is not None and part_a is None:  # 移动止盈：不卖，开始跟踪最高价
                        s.update(target=np.inf, trailing=True, peak=h[T, j])
                    elif top is not None and scope == "both" and j in top:  # 止盈时续持：按止盈价重新定价位
                        tgt, stp = levels(tpx, a, tp, sl)
                        s.update(basis=tpx, target=tgt, stop=stp, d0=T, a=a, ext=s["ext"] + 1)
                    else:
                        px, at_open, why = tpx, o[T, j] >= s["target"], "tp"
            if px is None:
                # 收盘后更新（第二天起生效）：移动止盈价、保本止损
                if s["trailing"] and np.isfinite(h[T, j]):
                    s["peak"] = max(s["peak"], h[T, j])
                    s["stop"] = max(s["stop"], s["peak"] * (1 - trail_k * s["a"]))
                if be_b and not s["be"] and np.isfinite(h[T, j]) and h[T, j] >= s["basis"] * (1 + be_b * s["a"]):
                    s["stop"], s["be"] = max(s["stop"], s["basis"]), True
                continue
            if not at_open:
                intraday_freed += 1
            sold_today.add(j)
            r += s["w"] * (px / o[T, j] - 1 - sell_cost) / n
            trades.append(dict(entry=dates[s["d_in"]], exit=dates[T], code=p.codes[j], why=why, ext=s["ext"],
                               net=s["realized"] + s["w"] * gross(px, s["entry"], sell_cost)))
            del held[j]
        cap = n - intraday_freed
        if gate_v[T - 1] and len(held) < cap:
            row = sc_v[T - 1]
            ok = ~np.isnan(row)
            for j in np.argsort(-np.where(ok, row, -np.inf))[:n * 3]:
                if len(held) >= cap or not ok[j]:
                    break
                if j in held or not traded[T, j] or raw_open[T, j] >= up_v[T, j] - 0.005:
                    continue
                if j in sold_today:
                    continue
                if gap_max is not None and gapv is not None and np.isfinite(gapv[T, j]) and gapv[T, j] > gap_max:
                    continue
                a = atr[T - 1, j]
                if not np.isfinite(a) or not np.isfinite(o[T, j]):
                    continue
                _, stop = levels(o[T, j], a, tp, sl)
                held[j] = dict(entry=o[T, j], basis=o[T, j], target=first_target(o[T, j], a), stop=stop, d0=T, d_in=T,
                               a=a, w=1.0, realized=0.0, trailing=False, peak=None, be=False, ext=0)
                r -= buy_cost / n
        if T + 1 >= len(dates):
            break
        for j, s in held.items():
            if np.isfinite(o[T + 1, j]) and o[T, j] > 0:
                r += s["w"] * (o[T + 1, j] / o[T, j] - 1) / n
        daily.append((dates[T], r))
        held_n.append((dates[T], sum(s["w"] for s in held.values())))
    return (pd.Series(dict(daily), dtype="float64"), trades, pd.Series(dict(held_n), dtype="float64"))


def exit_mix(trades):
    """各出场原因的笔数占比、平均收益、胜率。"""
    out = {}
    if not trades:
        return out
    df = pd.DataFrame(trades)
    for w, g in df.groupby("why"):
        out[w] = dict(share=float(len(g) / len(df)), avg=float(g["net"].mean()), win=float((g["net"] > 0).mean()))
    out["_extended"] = int((df["ext"] > 0).sum())
    return out


def total(name, d, held, sleeve, per):
    """选股层逐日收益 + 持仓份数 → 总组合（redline 线上口径）统计。"""
    frame = pd.DataFrame({f"{name}|raw": d, f"{name}|held": held.reindex(d.index).fillna(0)})
    frame.index.name = "date"
    part = redline.load_part(name, frame=frame)[per]
    r, _, _ = redline.simulate(part, sleeve, None)
    return redline.stats(r), r


def pick(rows):
    """训练期挑参数：每笔胜率 ≥ 50% 里选股层夏普最高（与 slots.py 的规则相同）。"""
    pool = [r for r in rows if (r["train"]["std"]["stock"].get("trade_win") or 0) >= GOAL_WIN] or rows
    return max(pool, key=lambda r: r["train"]["std"]["stock"].get("sharpe") or -9)


def run(kline, basic, valuation=None, forecast=None, grid=None, families_test_all=True):
    kline = kline() if callable(kline) else kline
    valuation = valuation() if callable(valuation) else valuation
    forecast = forecast() if callable(forecast) else forecast
    p, elig, sm, sc = scores(kline, basic, valuation, forecast)
    del kline
    sc = sc[SCORE_KEY]
    gate = emotion_regime(p, elig)
    ohl = adjusted_ohlc(p)
    gapv = (p.open / p.preclose - 1).to_numpy()
    tp, sl, n = sm["take_profit"], sm["stop_loss"], sm["n"]
    end = str(p.dates[-1].date())
    periods = dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, end))
    rets = dv.etf_returns()
    sleeve = dv.sleeve_returns(rets, sm["sleeve"]["codes"], sm["sleeve"].get("rebalance", 20))
    grid = grid or GRID
    slip0 = config.SLIPPAGE
    daily_cols = {}

    def one(fam, label, rule, hold, per, cost):
        config.SLIPPAGE = SLIP_STD if cost == "std" else SLIP_CONS
        s, e = periods[per]
        d, t, held = simulate_x(p, ohl, sc, gate, n, tp, sl, hold, s, e, rule=rule, gapv=gapv)
        key = f"{fam}:{label}:{cost}"
        tot, r = total("x", d, held, sleeve, per)
        daily_cols.setdefault(key, []).append(d)
        return dict(stock=_summ(d, t), total=tot, exits=exit_mix(t))

    try:
        fams = {}
        for fam, items in grid.items():
            rows = []
            for label, rule, hold in items:
                row = dict(family=fam, label=label, rule=rule, hold=hold, train={}, test={})
                row["train"]["std"] = one(fam, label, rule, hold, "train", "std")
                if families_test_all:
                    row["test"]["std"] = one(fam, label, rule, hold, "test", "std")
                log.info("%s %s: train %s", fam, label, row["train"]["std"]["stock"])
                rows.append(row)
            fams[fam] = rows
        base = fams["base"][0]
        chosen = {fam: pick(rows) for fam, rows in fams.items()}
        # 组合：训练期夏普好过线上的各族最优叠加（gap/breakeven/early/extend 与 trail/partial 可叠；trail 与 partial 二选一）
        bs = base["train"]["std"]["stock"]["sharpe"]
        better = [f for f in ("trail", "partial", "extend", "early", "gap", "breakeven")
                  if f in chosen and (chosen[f]["train"]["std"]["stock"]["sharpe"] or -9) > bs]
        if "trail" in better and "partial" in better:
            worse = min(("trail", "partial"), key=lambda f: chosen[f]["train"]["std"]["stock"]["sharpe"])
            better.remove(worse)
        if len(better) >= 2:
            rule, hold = {}, 10
            for f in better:
                rule.update(chosen[f]["rule"])
                hold = max(hold, chosen[f]["hold"])
            label = " + ".join(chosen[f]["label"] for f in better)
            row = dict(family="combo", label=label, rule=rule, hold=hold, train={}, test={})
            row["train"]["std"] = one("combo", label, rule, hold, "train", "std")
            fams["combo"] = [row]
            chosen["combo"] = row
        for fam, row in chosen.items():
            for per, cost in (("train", "cons"), ("test", "std"), ("test", "cons")):
                if cost not in row[per]:
                    row[per][cost] = one(fam, row["label"], row["rule"], row["hold"], per, cost)
            row["chosen"] = True
    finally:
        config.SLIPPAGE = slip0

    def verdict(row):
        if row["family"] == "base":
            return "对照"
        ok = []
        for cost in ("std", "cons"):
            a, b = row["test"][cost]["total"], base["test"][cost]["total"]
            ok.append(a["ann_return"] >= b["ann_return"] and a["max_drawdown"] >= b["max_drawdown"] - 0.01)
        if row["train"]["std"]["stock"]["sharpe"] <= bs:
            return "不采用（训练期不优于线上）"
        return "建议采用" if all(ok) else "不采用（样本外未通过）"

    for fam, row in chosen.items():
        row["verdict"] = verdict(row)
    out = pd.DataFrame({k: pd.concat(v) for k, v in daily_cols.items()
                        if any(k.startswith(f"{f}:{r['label']}:") for f, r in chosen.items())})
    out.index.name = "date"
    OUT_DAILY.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DAILY, float_format="%.6g")
    return dict(updated=end, families=fams, chosen={f: r["label"] for f, r in chosen.items()})


def _p(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def _f(x):
    return "—" if x is None else f"{x:.2f}"


EXIT_NAMES = dict(tp="止盈", sl="止损", expiry="到期", trail="移动止盈", early="提前卖", be="保本")


def to_markdown(res):
    fams = res["families"]
    chosen = [r for rows in fams.values() for r in rows if r.get("chosen")]
    base = fams["base"][0]
    L = [f"# 出场规则研究（修正同日腾位后，数据到 {res['updated']}）", "",
         "参数只在训练期 2015–2020 挑（每笔胜率 ≥ 50% 里选股层夏普最高），2021 至今只看。选股层含 20 日均线空仓；"
         "总组合 = 选股 65% + ETF 分仓闲钱停泊。标准成本滑点 0.1%/边，保守 0.2%/边。", "",
         "判定：两种成本下样本外总组合年化 ≥ 线上、最大回撤不差于线上 1 个百分点以上，且训练期夏普好过线上，才建议采用。", "",
         "## 各族训练期挑出的规则：总组合", "",
         "| 族 | 规则 | 训练年化(标准/保守) | 训练回撤 | 样本外年化(标准) | 样本外回撤(标准) | 样本外年化(保守) | 样本外回撤(保守) | 判定 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in chosen:
        a, c = r["train"]["std"]["total"], r["train"]["cons"]["total"]
        b, d = r["test"]["std"]["total"], r["test"]["cons"]["total"]
        L.append(f"| {FAMILY[r['family']]} | {r['label']} | {_p(a['ann_return'])}/{_p(c['ann_return'])} | {_p(a['max_drawdown'])} | "
                 f"{_p(b['ann_return'])} | {_p(b['max_drawdown'])} | {_p(d['ann_return'])} | {_p(d['max_drawdown'])} | {r['verdict']} |")
    L += ["", "## 各族训练期挑出的规则：选股层", "",
          "| 族 | 训练年化 | 训练夏普 | 训练胜率 | 样本外年化(标准) | 样本外夏普 | 样本外胜率 | 样本外笔数 | 样本外年化(保守) | 样本外出场构成 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in chosen:
        a, b, d = r["train"]["std"]["stock"], r["test"]["std"]["stock"], r["test"]["cons"]["stock"]
        mix = r["test"]["std"]["exits"]
        m = "，".join(f"{EXIT_NAMES.get(k, k)} {v['share']:.0%}/{_p(v['avg'])}" for k, v in mix.items() if not k.startswith("_"))
        L.append(f"| {FAMILY[r['family']]} | {_p(a['ann_return'])} | {_f(a['sharpe'])} | {_p(a['trade_win'])} | {_p(b['ann_return'])} | "
                 f"{_f(b['sharpe'])} | {_p(b['trade_win'])} | {b['trades']} | {_p(d['ann_return'])} | {m} |")
    L += ["", "## 全部参数（标准成本；样本外一列只看，不用于挑选）", "",
          "| 族 | 参数 | 训练年化 | 训练夏普 | 训练胜率 | 训练总组合年化/回撤 | 样本外年化 | 样本外夏普 | 样本外总组合年化/回撤 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for fam, rows in fams.items():
        for r in rows:
            a, ta = r["train"]["std"]["stock"], r["train"]["std"]["total"]
            b = r["test"].get("std", {}).get("stock", {})
            tb = r["test"].get("std", {}).get("total", {})
            mark = " ★" if r.get("chosen") else ""
            L.append(f"| {FAMILY[fam]} | {r['label']}{mark} | {_p(a['ann_return'])} | {_f(a['sharpe'])} | {_p(a['trade_win'])} | "
                     f"{_p(ta['ann_return'])}/{_p(ta['max_drawdown'])} | {_p(b.get('ann_return'))} | {_f(b.get('sharpe'))} | "
                     f"{_p(tb.get('ann_return'))}/{_p(tb.get('max_drawdown'))} |")
    bt = base["test"]["std"]["total"]
    L += ["", f"线上规则样本外总组合（标准成本）{_p(bt['ann_return'])} / {_p(bt['max_drawdown'])}。仅供研究，不构成投资建议。"]
    return "\n".join(L) + "\n"


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    OUT_MD.write_text(to_markdown(res))
