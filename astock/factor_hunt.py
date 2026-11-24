"""新因子挖掘（2026-11-12 用户：发掘新的因子加入回测，从而提高胜率和年化）。

线上选股 = 业绩预期四个因子加权（slot_model.json 的 weights）+ 情绪开关，最多 20 只，止盈 2 倍 ATR、止损 3 倍 ATR、
最长 10 天，净值跌破 20 日均线空仓。这里每次只加一个新因子，两种加法：
- 加分：得分 = 线上得分 + w × 新因子排名（w = 0.15、0.30；线上四个权重合计 1）；
- 过滤：新因子最差的 30% 不买，得分不变。
方向事先按文献/常识定好（不在样本外调）。规则只在 2015–2020 挑，2021 至今检验；收益扣佣金、滑点、印花税。
另外用 8 个随机因子按同样方式加分当噪声参照：新因子在训练期要好过随机的 90 分位才算数。

新因子（之前的研究都没用过）：
- 业绩预告细化：上调（同一报告期第二次预告比第一次高）、加速（比上一期预告增速高）、
  预告区间窄（公司更有把握）、发布得早（离报告期末天数少）、PEG（预告增速 / 市盈率）；
- 行业景气：行业里近 60 个交易日发正面预告的比例；个股预告增速减去行业中位数（行业中性）；
- 风险与交易：近 20 日最大单日涨幅低（不买“彩票股”）、特质波动低、浮盈（CGO）高、成交额（流动性）。
"""

import json
import logging

import numpy as np
import pandas as pd

from . import config, model
from .alt_study import BAD_TYPES, FORECAST_DAYS, GOOD_TYPES, emotion_regime, iter_alt_factors
from .factors import Panel, eligible, iter_factors
from .live_slots import MODEL_FILE
from .slots import simulate, stats
from .stability import overlay
from .winrate import adjusted_ohlc

log = logging.getLogger(__name__)

WEIGHTS = (0.15, 0.30)
FILTER_Q = 0.30
N_RANDOM = 8
# 名称 → (说明, 方向)
NEW = {
    "fc_rev": ("业绩预告上调：同一报告期新预告中值减去上一次", 1),
    "fc_accel": ("业绩加速：本次预告中值减去该公司上一期预告中值", 1),
    "fc_narrow": ("预告区间窄：(上限−下限)/|中值| 越小越好", -1),
    "fc_early": ("预告发得早：公告日离报告期末的天数越少越好", -1),
    "fc_peg": ("PEG：预告增速 / 市盈率（盈利股）", 1),
    "ind_breadth": ("行业景气：行业里近 60 日发正面预告的股票比例", 1),
    "fc_ind_rel": ("行业中性：预告增速减去行业中位数", 1),
    "maxret20": ("彩票股：近 20 日最大单日涨幅（越小越好）", -1),
    "idiovol60": ("特质波动（越小越好）", -1),
    "cgo": ("浮盈：收盘价相对换手加权持仓成本（越高越好）", 1),
    "amount20": ("流动性：近 20 日成交额（越大越好）", 1),
}


def forecast_extras(p, forecast, industry=None):
    """新的业绩预告类因子，对齐到交易日（公告后第一个交易日起，最多 FORECAST_DAYS 个交易日）。"""
    out = {}
    empty = lambda: pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype="float32")  # noqa: E731
    if forecast is None or forecast.empty:
        return {k: empty() for k in ("fc_rev", "fc_accel", "fc_narrow", "fc_early", "fc_mid", "fc_good")}
    f = forecast[forecast["code"].isin(p.codes)].sort_values("pub").copy()
    f["mid"] = f[["up", "dn"]].mean(axis=1).clip(-500, 500)
    f["good"] = np.where(f["type"].isin(GOOD_TYPES), 1.0, np.where(f["type"].isin(BAD_TYPES), -1.0, 0.0))
    g = f.groupby(["code", "stat"])["mid"]
    f["rev"] = (f["mid"] - g.shift(1)).clip(-500, 500)  # 同一报告期上一次预告；没有就 NaN
    prev_period = f.drop_duplicates(["code", "stat"], keep="last").sort_values(["code", "stat"])
    prev_period["prev_mid"] = prev_period.groupby("code")["mid"].shift(1)
    f = f.merge(prev_period[["code", "stat", "prev_mid"]], on=["code", "stat"], how="left")
    f["accel"] = (f["mid"] - f["prev_mid"]).clip(-500, 500)
    width = (f["up"] - f["dn"]).abs()
    f["narrow"] = (width / f["mid"].abs().clip(lower=5)).clip(0, 10)
    f["early"] = (f["pub"] - f["stat"]).dt.days.clip(-120, 200).astype(float)
    pos = p.dates.searchsorted(f["pub"].values, side="right")
    keep = pos < len(p.dates)
    f, pos = f[keep], pos[keep]
    col = p.codes.get_indexer(f["code"])
    ok = col >= 0
    f, pos, col = f[ok], pos[ok], col[ok]

    def place(vals, limit=FORECAST_DAYS):
        a = np.full((len(p.dates), len(p.codes)), np.nan, dtype="float32")
        a[pos, col] = vals
        return pd.DataFrame(a, index=p.dates, columns=p.codes).ffill(limit=limit)

    for k in ("rev", "accel", "narrow", "early", "mid", "good"):
        out["fc_" + k] = place(f[k].to_numpy(dtype="float32"))
    out["fc_good60"] = place(f["good"].to_numpy(dtype="float32"), limit=60)
    return out


def industry_factors(p, ex, industry):
    """行业景气（行业里近 60 日正面预告比例）和行业中性的预告增速。industry: code → 行业。"""
    ind = pd.Series(industry).reindex(p.codes)
    out = {}
    good = (ex["fc_good60"] > 0).astype("float32").where(p.traded)
    mid = ex["fc_mid"]
    br = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype="float32")
    rel = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype="float32")
    for name, cols in ind.groupby(ind).groups.items():
        cols = list(cols)
        if len(cols) < 5:
            continue
        share = good[cols].mean(axis=1)
        br[cols] = np.repeat(share.to_numpy()[:, None], len(cols), axis=1)
        med = mid[cols].median(axis=1)
        rel[cols] = mid[cols].sub(med, axis=0)
    out["ind_breadth"] = br.where(p.traded)
    out["fc_ind_rel"] = rel.where(p.traded)
    return out


def _summ(d_raw, trades):
    st = stats(overlay(d_raw, "trend20"), trades)
    return {k: st.get(k) for k in ("ann_return", "max_drawdown", "sharpe", "trade_win", "trades", "avg_trade", "payoff")}


# 第二轮：第一轮训练期最好的几个组合起来（2026-11-12，只按第一轮的训练期结果挑）
COMBOS = {
    "accel0.3": dict(blend={"fc_accel": 0.30}),
    "accel0.45": dict(blend={"fc_accel": 0.45}),
    "accel0.3+liq": dict(blend={"fc_accel": 0.30}, filt=["amount20"]),
    "accel0.3+narrow": dict(blend={"fc_accel": 0.30}, filt=["fc_narrow"]),
    "accel0.3+liq+narrow": dict(blend={"fc_accel": 0.30}, filt=["amount20", "fc_narrow"]),
    "accel0.3+rev0.15": dict(blend={"fc_accel": 0.30, "fc_rev": 0.15}),
    "accel0.3+rev0.15+liq": dict(blend={"fc_accel": 0.30, "fc_rev": 0.15}, filt=["amount20"]),
    "liq+narrow": dict(filt=["amount20", "fc_narrow"]),
}


def combo_score(base, cand, spec):
    sc = base.copy()
    for k, w in spec.get("blend", {}).items():
        sc = sc + w * cand[k].fillna(0)
    sc = sc.where(base.notna())
    for k in spec.get("filt", []):
        r = cand[k]
        sc = sc.where(~r.lt(r.quantile(FILTER_Q, axis=1), axis=0))
    return sc


def run(kline, basic, valuation=None, forecast=None, industry=None, combos=False, dump=None):
    kline = kline() if callable(kline) else kline
    valuation = valuation() if callable(valuation) else valuation
    forecast = forecast() if callable(forecast) else forecast
    sm = json.loads(MODEL_FILE.read_text())
    weights = {k: v for k, v in sm["weights"].items() if v}
    tp, sl, hold, n = sm["take_profit"], sm["stop_loss"], sm["max_hold"], sm["n"]
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
    cand = {}
    for k, v in iter_factors(p):
        if k in NEW:
            cand[k] = model.xs_rank(v, elig) * NEW[k][1]
        del v
    ex = forecast_extras(p, forecast)
    pe = p.valuation.get("peTTM")
    if pe is not None:
        ex["fc_peg"] = (ex["fc_mid"] / pe.where(pe > 0)).clip(-50, 50)
    if industry is not None:
        ex.update(industry_factors(p, ex, industry))
    for k in NEW:
        if k in ex and k not in cand:
            cand[k] = model.xs_rank(ex[k], elig) * NEW[k][1]
    del ex
    gate = emotion_regime(p, elig)
    ohl = adjusted_ohlc(p)
    end = str(p.dates[-1].date())
    periods = dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, end))

    def sim(sc):
        res = {}
        for k, (s, e) in periods.items():
            d, t = simulate(p, ohl, sc, gate, n, tp, sl, hold, s, e)
            res[k] = _summ(d, t)
            res[k]["coverage"] = float(sc.loc[s:e].notna().sum(axis=1).mean())
        return res

    rows = [dict(name="线上规则（对照）", factor="live", mode="live", w=0.0, **sim(base))]
    log.info("live: %s", rows[0]["train"])
    if combos:
        cols = {}
        for name, spec in dict(live={}, **COMBOS).items():
            sc = base if name == "live" else combo_score(base, cand, spec)
            if name != "live":
                rows.append(dict(name=name, factor=name, mode="combo", w=0.0, **sim(sc)))
                log.info("%s: train %s test %s", name, rows[-1]["train"], rows[-1]["test"])
            if dump is not None:
                parts = []
                for k, (s, e) in periods.items():
                    d, t = simulate(p, ohl, sc, gate, n, tp, sl, hold, s, e)
                    held = pd.Series(0, index=d.index)
                    for x in t:
                        held[(held.index >= x["entry"]) & (held.index < x["exit"])] += 1
                    parts.append(pd.DataFrame({name + "|raw": d, name + "|held": held}))
                df = pd.concat(parts)
                for c in df:
                    cols[c] = df[c]
        if dump is not None:
            out = pd.DataFrame(cols)
            out.index.name = "date"
            out.to_csv(dump, float_format="%.6g")
        return dict(updated=end, rows=rows, new=NEW, combos=COMBOS, filter_q=FILTER_Q)
    for k, r in cand.items():
        cov = float(r.notna().mean().mean())
        for w in WEIGHTS:
            sc = (base + w * r.fillna(0)).where(base.notna())
            rows.append(dict(name=f"{NEW[k][0]} 加分 {w}", factor=k, mode="blend", w=w, cand_coverage=cov, **sim(sc)))
            log.info("%s blend %.2f: train %s test %s", k, w, rows[-1]["train"]["sharpe"], rows[-1]["test"]["sharpe"])
        # 过滤：排名最差的 30% 不买（没有这个因子的股票不过滤）
        cut = r.quantile(FILTER_Q, axis=1)
        bad = r.lt(cut, axis=0)
        sc = base.where(~bad)
        rows.append(dict(name=f"{NEW[k][0]} 过滤最差 30%", factor=k, mode="filter", w=0.0, cand_coverage=cov, **sim(sc)))
        log.info("%s filter: train %s test %s", k, rows[-1]["train"]["sharpe"], rows[-1]["test"]["sharpe"])
    rng = np.random.default_rng(7)
    for i in range(N_RANDOM):
        noise = pd.DataFrame(rng.random(base.shape, dtype=np.float32) - 0.5, index=base.index, columns=base.columns)
        for w in WEIGHTS:
            sc = (base + w * noise).where(base.notna())
            rows.append(dict(name=f"随机因子 {i + 1} 加分 {w}", factor=f"random{i + 1}", mode="random", w=w, **sim(sc)))
        del noise
    return dict(updated=end, rows=rows, new=NEW, weights=list(WEIGHTS), filter_q=FILTER_Q)


def _p(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def to_markdown(res):
    rows = res["rows"]
    live = rows[0]
    rnd = [r for r in rows if r["mode"] == "random"]
    rnd_tr = np.array([r["train"]["sharpe"] or 0 for r in rnd])
    p90 = float(np.quantile(rnd_tr, 0.9)) if len(rnd_tr) else float("nan")
    L = [f"# 新因子挖掘（数据到 {res['updated']}）", "",
         "线上选股（业绩预期 + 情绪开关，20 只，止盈止损，净值均线空仓）上每次加一个新因子。"
         "训练期 2015–2020，样本外 2021 至今，100% 资金口径，已扣成本。", "",
         f"随机因子加分的训练期夏普 90 分位：{p90:.2f}（线上 {live['train']['sharpe']:.2f}）。", "",
         "| 方案 | 训练年化 | 训练回撤 | 训练夏普 | 训练胜率 | 样本外年化 | 样本外回撤 | 样本外夏普 | 样本外胜率 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -(r["train"]["sharpe"] or -9)):
        if r["mode"] == "random":
            continue
        a, b = r["train"], r["test"]
        L.append(f"| {r['name']} | {_p(a['ann_return'])} | {_p(a['max_drawdown'])} | {a['sharpe'] or 0:.2f} | "
                 f"{_p(a['trade_win'])} | {_p(b['ann_return'])} | {_p(b['max_drawdown'])} | {b['sharpe'] or 0:.2f} | {_p(b['trade_win'])} |")
    L += ["", "随机因子（噪声参照）：", ""]
    for r in rnd:
        L.append(f"- {r['name']}：训练夏普 {r['train']['sharpe'] or 0:.2f}，样本外夏普 {r['test']['sharpe'] or 0:.2f}，"
                 f"样本外年化 {_p(r['test']['ann_return'])}")
    return "\n".join(L) + "\n"


def write(res, stem="factor_hunt"):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / f"{stem}.md").write_text(to_markdown(res))
    (config.PAGE_DIR / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
