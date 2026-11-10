"""分散化：选股 + 低相关 ETF 分仓（2026-11-10 用户：寻找扩大年化的同时减少回撤的策略）。

之前试过的“按市场/宏观/技术调整仓位”都没能在样本外超过线上规则，因为它们都还是押在同一个 A 股选股策略上。
提高“收益/回撤”比更可靠的办法是加一块和 A 股选股相关性很低、本身也赚钱的资产，让两者的回撤错开：
- 选股部分规则完全不变（业绩预期 + 情绪开关，最多 20 只，止盈 2 倍 ATR、止损 3 倍 ATR、最长 10 天，
  净值跌破 20 日均线空仓），只是每只的资金按 capital_share / 20 算；
- 其余资金放进 ETF 分仓，按资产大类各取一只：纳指ETF（美股）、黄金ETF、国债ETF，各 1/3，每 20 个交易日调回等权；
- “闲钱停泊”：选股没用上的钱（空位、空仓开关关着时）也放进 ETF 分仓；要买股票时先卖 ETF 腾出现金
  （这三只 ETF 都能 T+0，卖出当天资金可用）；闲钱攒到总资金的 10% 以上才买回 ETF，避免天天来回倒。

公平比较：组合方式本身也会改变回撤，所以每个方案的选股资金比例都按“训练期（2015–2020）最大回撤不超过 15%”
（用户上限）定，再看 2021 年至今的样本外；线上规则同样按训练期回撤 15% 折算资金比例作对照。
ETF 收益用后复权（含分红）的开盘价到下一天开盘价，信号用前一天收盘，和选股的逐日收益口径一致；
ETF 每次买卖扣 0.08%（佣金 + 滑点，ETF 不收印花税）。
另外把 11 只 2015 年前上市的 ETF 的 1–4 只等权组合全部跑一遍（穷举，约 290 组）当参照：
“随便分散一下”能好多少，按训练期挑出来的组合在样本外怎样。
"""

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

RESEARCH_DIR = config.ROOT / "data" / "research"
ETF_DIR = RESEARCH_DIR / "etf"
ETF_COST = 0.0008  # 每次买卖（单边）
TARGET_DD = -0.15  # 训练期最大回撤上限（用户要求 10–15%）
SHARES = np.round(np.arange(0.2, 1.0001, 0.05), 2)
NAMES = {
    "sh510300": "沪深300ETF", "sh510500": "中证500ETF", "sh512100": "中证1000ETF", "sz159915": "创业板ETF",
    "sh510880": "红利ETF", "sz159920": "恒生ETF", "sh513050": "中概互联ETF", "sh513100": "纳指ETF",
    "sh513500": "标普500ETF", "sh513520": "日经ETF", "sh518880": "黄金ETF", "sz159985": "豆粕ETF",
    "sh511010": "国债ETF", "sh511260": "十年国债ETF", "sh511220": "城投债ETF", "sh511880": "银华日利",
}


# ---------- 数据 ----------

def load_live(path=RESEARCH_DIR / "live_daily.csv", n=20):
    """线上选股规则 100% 资金的逐日收益（第 T 行 = T 开盘到 T+1 开盘）。返回 {train, test} → DataFrame。

    pos 是“净值跌破 20 日均线空仓”的开关（同 stability.overlay，错开两行），
    expo 是第 T 天开盘时股票占选股资金的比例（前一天收盘持有几只 / 20，乘开关）。"""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    out = {}
    for p in ("train", "test"):
        x = df[df["period"] == p].copy()
        eq = (1 + x["raw"]).cumprod()
        x["pos"] = (eq >= eq.rolling(20, min_periods=10).mean()).astype(float).shift(2).fillna(1.0)
        x["expo"] = (x["pos"] * x["held"].shift(1).fillna(0) / n).clip(0, 1)
        out[p] = x
    return out


def etf_returns(etf_dir=ETF_DIR):
    """每只 ETF 第 T 行 = T 开盘到 T+1 开盘的含分红收益（有后复权列用后复权，否则用前复权）。"""
    cols = {}
    for f in sorted(Path(etf_dir).glob("s*.csv")):
        d = pd.read_csv(f, parse_dates=["date"], index_col="date")
        o = d["h_open"] if "h_open" in d else d["open"]
        cols[f.stem] = o.pct_change(fill_method=None).shift(-1)
    return pd.DataFrame(cols).sort_index()


# ---------- 组合 ----------

def sleeve_returns(rets, codes, rebalance=20):
    """等权 ETF 分仓：权重随涨跌漂移，每 rebalance 个交易日调回等权（扣调仓成本）。没上市的那几天当现金。"""
    r = rets[codes].fillna(0.0).to_numpy()
    k = len(codes)
    w = np.full(k, 1.0 / k)
    out = np.empty(len(r))
    for i in range(len(r)):
        cost = 0.0
        if i % rebalance == 0:
            cost = np.abs(w - 1.0 / k).sum() * ETF_COST
            w = np.full(k, 1.0 / k)
        g = float((w * r[i]).sum())
        out[i] = g - cost
        w = w * (1 + r[i]) / (1 + g)
    return pd.Series(out, index=rets.index)


def sleeve_weight_path(expo, share, mode, band=0.10):
    """ETF 分仓占总资金的比例。fixed：固定 1 − share；park：选股闲着的钱也进 ETF，
    要用钱时立刻减（先卖 ETF 再买股），闲钱超过 band 才加回。"""
    if mode == "fixed":
        return pd.Series(1 - share, index=expo.index)
    tgt = (1 - share * expo).to_numpy()
    ws, cur = np.empty(len(tgt)), tgt[0] if len(tgt) else 0.0
    for i, t in enumerate(tgt):
        if t < cur or t - cur > band:
            cur = t
        ws[i] = cur
    return pd.Series(ws, index=expo.index)


def combine(part, sleeve, share, mode, band=0.10):
    """总资金的逐日收益：share × 选股（100% 口径）+ ETF 分仓 × 其比例 − ETF 进出成本。"""
    s = sleeve.reindex(part.index).fillna(0.0)
    ws = sleeve_weight_path(part["expo"], share, mode, band)
    return share * part["live"] + ws * s - ws.diff().abs().fillna(0) * ETF_COST, ws


def stats(d):
    d = pd.Series(d).dropna()
    if len(d) < 2:
        return dict(ann_return=None, max_drawdown=None, sharpe=None)
    eq = (1 + d).cumprod()
    sd = d.std()
    return dict(ann_return=float(eq.iloc[-1] ** (244 / len(d)) - 1), max_drawdown=float((eq / eq.cummax() - 1).min()),
                sharpe=float(d.mean() / sd * np.sqrt(244)) if sd > 0 else None)


def pick_share(fn, target=TARGET_DD):
    """训练期回撤不超过 target 的资金比例里，训练期年化最高的那个。fn(share) → 训练期逐日收益。"""
    best = None
    for c in SHARES:
        st = stats(fn(c))
        if st["max_drawdown"] is not None and st["max_drawdown"] >= target and (
                best is None or st["ann_return"] > best[1]["ann_return"]):
            best = (float(c), st)
    return best


def live_share_for(train, target=TARGET_DD):
    """只做选股时，训练期回撤正好到 target 的资金比例（二分）。"""
    def dd(c):
        return stats(train * c)["max_drawdown"]
    lo, hi = 0.01, 1.5
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if dd(mid) >= target else (lo, mid)
    return lo


# ---------- 研究 ----------

def run(model_file=config.ROOT / "astock" / "slot_model.json"):
    parts = load_live()
    rets = etf_returns()
    sm = json.loads(Path(model_file).read_text())
    cfg = sm.get("sleeve") or dict(codes=["sh513100", "sh518880", "sh511010"], rebalance=20, park_band=0.10)
    end = str(parts["test"].index[-1].date())

    def row(name, fn_train, fn_test, share, kind):
        return dict(name=name, kind=kind, share=share, train=stats(fn_train), test=stats(fn_test))

    rows = []
    c_eq = live_share_for(parts["train"]["live"])
    for name, c in (("只做选股（训练期回撤 15% 的资金比例）", c_eq), ("只做选股 40% 资金（现在线上）", 0.4),
                    ("只做选股 80% 资金（PR #17 提议）", 0.8)):
        rows.append(row(name, parts["train"]["live"] * c, parts["test"]["live"] * c, round(c, 3), "live"))

    sleeve = sleeve_returns(rets, cfg["codes"], cfg.get("rebalance", 20))
    label = " + ".join(NAMES.get(c, c) for c in cfg["codes"])
    rows.append(row(f"只买 ETF 分仓（{label} 等权）", sleeve[parts["train"].index], sleeve[parts["test"].index],
                    0.0, "sleeve"))
    curves = {}
    for mode, mname in (("fixed", "固定比例"), ("park", "闲钱停泊")):
        best = pick_share(lambda c, m=mode: combine(parts["train"], sleeve, c, m, cfg.get("park_band", 0.1))[0])
        c = best[0]
        tr, _ = combine(parts["train"], sleeve, c, mode, cfg.get("park_band", 0.1))
        te, ws = combine(parts["test"], sleeve, c, mode, cfg.get("park_band", 0.1))
        r = row(f"选股 {c:.0%} + ETF 分仓（{mname}）", tr, te, c, mode)
        r["avg_sleeve_weight"] = dict(test=float(ws.mean()))
        rows.append(r)
        curves[mode] = te

    chosen_share = float(sm.get("capital_share", rows[-1]["share"]))
    tr, _ = combine(parts["train"], sleeve, chosen_share, "park", cfg.get("park_band", 0.1))
    te, ws_te = combine(parts["test"], sleeve, chosen_share, "park", cfg.get("park_band", 0.1))
    chosen = row(f"上线方案：选股 {chosen_share:.0%} + ETF 分仓闲钱停泊", tr, te, chosen_share, "chosen")
    yearly = {}
    for nm, d in (("chosen", pd.concat([tr, te])),
                  ("live40", pd.concat([parts["train"]["live"], parts["test"]["live"]]) * 0.4),
                  ("live80", pd.concat([parts["train"]["live"], parts["test"]["live"]]) * 0.8),
                  ("sleeve", sleeve[pd.concat([parts["train"], parts["test"]]).index])):
        y = (1 + d).groupby(d.index.year).prod() - 1
        yearly[nm] = {str(k): round(float(v), 4) for k, v in y.items()}

    # 穷举参照：2015 年前上市的 ETF，1–4 只等权
    full = [c for c in rets.columns if rets[c].loc["2015-01-05":"2015-03-01"].notna().all()]
    scan = []
    for k in (1, 2, 3, 4):
        for combo in itertools.combinations(full, k):
            sl = sleeve_returns(rets, list(combo), 20)
            for mode in ("fixed", "park"):
                best = pick_share(lambda c, m=mode: combine(parts["train"], sl, c, m)[0])
                if not best:
                    continue
                te_s = stats(combine(parts["test"], sl, best[0], mode)[0])
                scan.append(dict(set="+".join(NAMES.get(x, x) for x in combo), k=k, mode=mode, share=best[0],
                                 train=best[1]["ann_return"], train_dd=best[1]["max_drawdown"],
                                 test=te_s["ann_return"], test_dd=te_s["max_drawdown"], test_sharpe=te_s["sharpe"]))
    scan = pd.DataFrame(scan)
    base = rows[0]["test"]
    summary = {}
    for mode in ("fixed", "park"):
        x = scan[scan["mode"] == mode]
        top = x.sort_values("train", ascending=False).head(5)
        summary[mode] = dict(n=int(len(x)), test_p10=float(x["test"].quantile(0.1)), test_p50=float(x["test"].median()),
                             test_p90=float(x["test"].quantile(0.9)),
                             beat_live=float((x["test"] > base["ann_return"]).mean()),
                             train_test_corr=float(x[["train", "test"]].corr().iloc[0, 1]),
                             top_by_train=top.to_dict("records"))

    def weekly(d):
        eq = (1 + d).cumprod()
        return [round(float(v), 4) for v in eq.resample("W-FRI").last().ffill()]

    idx = parts["test"]["live"].resample("W-FRI").last().index
    curve = dict(dates=[str(d.date()) for d in idx], chosen=weekly(te), live40=weekly(parts["test"]["live"] * 0.4),
                 live80=weekly(parts["test"]["live"] * 0.8), live_eq=weekly(parts["test"]["live"] * c_eq),
                 sleeve=weekly(sleeve[parts["test"].index]))
    corr = float(np.corrcoef(parts["test"]["live"], sleeve[parts["test"].index].fillna(0))[0, 1])
    return dict(updated=end, sleeve=cfg, sleeve_label=label, target_dd=TARGET_DD, rows=rows, chosen=chosen,
                avg_sleeve_weight_test=float(ws_te.mean()), yearly=yearly, scan=summary, curve=curve,
                corr_live_sleeve_test=corr, scan_rows=scan)


def _p(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def to_markdown(res):
    L = [f"# 选股 + ETF 分仓（分散化）回测（数据到 {res['updated']}）", "",
         "选股规则不变；其余资金和选股闲着的钱放进 " + res["sleeve_label"] + " 等权分仓。"
         "训练期 2015–2020 定资金比例（训练期最大回撤不超过 15%），2021 年至今是样本外。收益已扣成本。", "",
         "| 方案 | 选股资金 | 训练期年化 | 训练期回撤 | 样本外年化 | 样本外回撤 | 样本外夏普 |", "|---|---|---|---|---|---|---|"]
    for r in res["rows"] + [res["chosen"]]:
        L.append(f"| {r['name']} | {_p(r['share'], 0)} | {_p(r['train']['ann_return'])} | {_p(r['train']['max_drawdown'])} | "
                 f"{_p(r['test']['ann_return'])} | {_p(r['test']['max_drawdown'])} | {r['test']['sharpe'] or 0:.2f} |")
    L += ["", f"样本外选股和 ETF 分仓的日收益相关系数 {res['corr_live_sleeve_test']:.2f}；"
          f"上线方案样本外平均 {_p(res['avg_sleeve_weight_test'], 0)} 的资金在 ETF 分仓里。", "",
          "## 逐年收益", "", "| 年份 | 上线方案 | 只做选股 40% | 只做选股 80% | 只买 ETF 分仓 |", "|---|---|---|---|---|"]
    y = res["yearly"]
    for yr in y["chosen"]:
        L.append(f"| {yr} | {_p(y['chosen'].get(yr))} | {_p(y['live40'].get(yr))} | {_p(y['live80'].get(yr))} | {_p(y['sleeve'].get(yr))} |")
    L += ["", "## 穷举参照（11 只 2015 年前上市的 ETF，1–4 只等权，资金比例同样按训练期回撤 15% 定）", ""]
    for mode, s in res["scan"].items():
        nm = "固定比例" if mode == "fixed" else "闲钱停泊"
        L.append(f"- {nm}：{s['n']} 组，样本外年化 10%/50%/90% 分位 {_p(s['test_p10'])} / {_p(s['test_p50'])} / "
                 f"{_p(s['test_p90'])}，{_p(s['beat_live'], 0)} 的组合好过只做选股（同回撤）；"
                 f"训练期和样本外年化的相关 {s['train_test_corr']:.2f}。训练期最好的 5 组：")
        for t in s["top_by_train"]:
            L.append(f"  - {t['set']}（选股 {_p(t['share'], 0)}）：训练 {_p(t['train'])}/{_p(t['train_dd'])} → "
                     f"样本外 {_p(t['test'])}/{_p(t['test_dd'])}")
    L += ["", "## 注意", "",
          "- 2021 年以来黄金和纳指涨得特别好，样本外的超额有一部分是这段行情给的；训练期（2015–2020）同样有效，但以后未必这么多。",
          "- 纳指ETF 是 QDII，场内价格常有溢价（2023–2024 年一度超过 10%），溢价很高时少买或换标普500ETF。",
          "- ETF 分仓本身最大回撤约 11%（2022 年美股大跌时），和选股的回撤大多不在同一段，所以合起来回撤没有变大。"]
    return "\n".join(L) + "\n"


def chart(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei"):
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [f]
            break
    plt.rcParams["axes.unicode_minus"] = False
    c = res["curve"]
    d = pd.to_datetime(c["dates"])
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    ch = res["chosen"]["test"]
    lines = [("chosen", f"上线方案：选股 {res['chosen']['share']:.0%} + ETF 分仓（年化 {_p(ch['ann_return'])}，回撤 {_p(ch['max_drawdown'])}）", "#c0392b", 2.4)]
    for k, nm, col in (("live80", "只做选股 80%", "#e67e22"), ("live40", "只做选股 40%（现在线上）", "#7f8c8d"),
                       ("sleeve", "只买 ETF 分仓", "#2980b9")):
        r = next((x for x in res["rows"] if x["kind"] == ("sleeve" if k == "sleeve" else "live") and
                  (k == "sleeve" or x["share"] == (0.8 if k == "live80" else 0.4))), None)
        lab = nm if r is None else f"{nm}（年化 {_p(r['test']['ann_return'])}，回撤 {_p(r['test']['max_drawdown'])}）"
        lines.append((k, lab, col, 1.6))
    for k, lab, col, lw in lines:
        ax.plot(d, c[k], label=lab, color=col, lw=lw)
    ax.set_title("样本外净值（2021 年至今，扣成本）")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "diversify.md").write_text(to_markdown(res))
    out = {k: v for k, v in res.items() if k != "scan_rows"}
    (config.PAGE_DIR / "diversify.json").write_text(json.dumps(out, ensure_ascii=False, default=float))
    res["scan_rows"].to_csv(config.REPORT_DIR / "diversify_scan.csv", index=False, float_format="%.4f")
