"""试验：A股、港股对急跌反应偏慢的几种改法，只在回测里比较，不改线上配置。

用法：python scripts/experiment_cn.py   （需先有 data/raw 缓存）
结果写入 reports/experiments/cn_hk_variants.md。方案只依据 2018 年以前的结果挑选，2019 年以后用来检验。
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fearindex import backtest, compute, config, fetch  # noqa: E402

SPLIT = pd.Timestamp("2019-01-01")
OUT = Path(__file__).resolve().parent.parent / "reports" / "experiments" / "cn_hk_variants.md"
CHECK_DAYS = ["2026-03-20", "2026-03-23", "2026-03-24", "2026-04-07", "2026-10-23"]

# ---------- 试验用的新变换 ----------
# 输出序列的 name 决定标准化方式：默认滚动 5 年分位；__fixed30__ 为固定刻度（回撤 30% = 满分恐慌）；
# __expanding__ 为与全部历史比较的分位（不受最近 5 年行情影响）。


def _dd(s):
    return 1 - s / s.rolling(252, min_periods=20).max()


def _down_volume(v, p):
    v, p = compute._align(v, p)
    surge = np.log(v / v.rolling(20, min_periods=15).mean())
    # 下跌日放量才算恐慌；上涨日记 0
    return surge.where(p.pct_change() < 0, 0.0).clip(lower=0)


NEW_TRANSFORMS = {
    "fall_5d": lambda s: -compute._ret(s, 5),
    "fall_10d": lambda s: -compute._ret(s, 10),
    "down_volume": _down_volume,
    "dd_fixed30": lambda s: _dd(s).rename("__fixed30__"),
    "dd_expanding": lambda s: _dd(s).rename("__expanding__"),
}

_orig_rank = compute.rolling_pct_rank


def _rank(s, window=None, min_periods=None):
    if s.name == "__fixed30__":
        return (s / 0.30 * 100).clip(0, 100)
    if s.name == "__expanding__":
        return s.expanding(min_periods=config.RANK_MIN_PERIODS).rank(pct=True) * 100
    return _orig_rank(s, window, min_periods)


compute.TRANSFORMS.update(NEW_TRANSFORMS)
compute.rolling_pct_rank = _rank

# ---------- 方案 ----------


def comp(region, transform, inputs, weight, name):
    return dict(region=region, transform=transform, inputs=inputs, weight=weight, name=name)


VARIANTS = {
    "cn": {
        "现行": {},
        "A 加5日急跌": {"cn_crash": comp("cn", "fall_5d", ["csi300"], 1.0, "5日急跌")},
        "A' 加10日急跌": {"cn_crash": comp("cn", "fall_10d", ["csi300"], 1.0, "10日急跌")},
        "B 放量下跌算恐慌": {"cn_panicvol": comp("cn", "down_volume", ["a_turnover", "csi300"], 0.5, "放量下跌")},
        "C1 回撤固定刻度": {"cn_dd": comp("cn", "dd_fixed30", ["csi300"], 1.0, "沪深300回撤")},
        "C2 回撤比全历史": {"cn_dd": comp("cn", "dd_expanding", ["csi300"], 1.0, "沪深300回撤")},
        "A+B+C1": {"cn_crash": comp("cn", "fall_5d", ["csi300"], 1.0, "5日急跌"),
                   "cn_panicvol": comp("cn", "down_volume", ["a_turnover", "csi300"], 0.5, "放量下跌"),
                   "cn_dd": comp("cn", "dd_fixed30", ["csi300"], 1.0, "沪深300回撤")},
        "A+B+C2": {"cn_crash": comp("cn", "fall_5d", ["csi300"], 1.0, "5日急跌"),
                   "cn_panicvol": comp("cn", "down_volume", ["a_turnover", "csi300"], 0.5, "放量下跌"),
                   "cn_dd": comp("cn", "dd_expanding", ["csi300"], 1.0, "沪深300回撤")},
    },
    "hk": {
        "现行": {},
        "A 加5日急跌": {"hk_crash": comp("hk", "fall_5d", ["hsi"], 1.0, "5日急跌")},
        "A' 加10日急跌": {"hk_crash": comp("hk", "fall_10d", ["hsi"], 1.0, "10日急跌")},
        "C1 回撤固定刻度": {"hk_dd": comp("hk", "dd_fixed30", ["hsi"], 1.0, "恒指回撤")},
        "C2 回撤比全历史": {"hk_dd": comp("hk", "dd_expanding", ["hsi"], 1.0, "恒指回撤")},
        "A+C1": {"hk_crash": comp("hk", "fall_5d", ["hsi"], 1.0, "5日急跌"),
                 "hk_dd": comp("hk", "dd_fixed30", ["hsi"], 1.0, "恒指回撤")},
        "A+C2": {"hk_crash": comp("hk", "fall_5d", ["hsi"], 1.0, "5日急跌"),
                 "hk_dd": comp("hk", "dd_expanding", ["hsi"], 1.0, "恒指回撤")},
    },
}


def _perf(ret):
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    vol = ret.std() * np.sqrt(252)
    return (eq.iloc[-1] ** (252 / len(ret)) - 1) / vol, (1 - eq / eq.cummax()).max()


def evaluate(score, price):
    start = score.first_valid_index()
    score, price = score.loc[start:], price.loc[start:]
    events = backtest.event_capture(score, price)
    alarms = backtest.false_alarms(score, price)
    ret = price.pct_change()
    crash = ret <= -0.03  # 单日跌 3% 以上
    resp = score.diff()[crash]
    expo = (1 - 0.5 * config.risk_off(score).astype(float)).shift(1)
    row = {}
    for per, in_per in (("内", lambda d: d < SPLIT), ("外", lambda d: d >= SPLIT)):
        ev = [e for e in events if in_per(pd.Timestamp(e["peak"]))]
        al = [a for a in alarms if in_per(pd.Timestamp(a["date"]))]
        mask = in_per(score.index)
        row[f"识别率{per}"] = f"{np.mean([e['captured'] for e in ev]) * 100:.0f}%（{len(ev)}次）" if ev else "—"
        hits = [e["dd_at_hit"] for e in ev if e["captured"] and e.get("dd_at_hit") is not None]
        row[f"报警时回撤{per}"] = f"{np.median(hits):.1f}%" if hits else "—"
        row[f"误报{per}"] = f"{sum(a['false_alarm'] for a in al)}/{len(al)}"
        row[f"恐慌天数占比{per}"] = f"{(score[mask] <= 30).mean() * 100:.0f}%"
        r = resp[in_per(resp.index)]
        row[f"急跌日降分{per}"] = f"{-r.mean():.1f}" if len(r) else "—"
        sh, mdd = _perf((expo * ret)[mask])
        bsh, bmdd = _perf(ret[mask])
        row[f"减仓夏普增量{per}"] = f"{sh - bsh:+.2f}"
        row[f"减仓回撤改善{per}"] = f"{(bmdd - mdd) * 100:+.1f}"
    for d in CHECK_DAYS:
        v = score.loc[:d].dropna()
        row[d[5:]] = f"{v.iloc[-1]:.0f}" if len(v) else "—"
    return row


def main():
    series = fetch.load_all()
    # “现行”指 2026-10-25 加入急跌成分之前的配置
    base = {c: s for c, s in copy.deepcopy(config.COMPONENTS).items() if not c.endswith("_crash")}
    tables = {}
    for region, variants in VARIANTS.items():
        rows = {}
        for name, change in variants.items():
            config.COMPONENTS = copy.deepcopy(base)
            config.COMPONENTS.update(copy.deepcopy(change))
            idx, _, _ = compute.build(series)
            px = backtest.price_panel(series, idx.index)
            rows[name] = evaluate(idx[region], px[region])
            print(region, name, rows[name])
        tables[region] = pd.DataFrame(rows).T
    config.COMPONENTS = base
    write(tables)


def write(tables):
    names = {"cn": "A股", "hk": "港股"}
    lines = ["# A股、港股改法回测对比", "",
             "只在回测里比较，线上指数没有改动。“内”= 样本内（2018 年以前，用来挑方案），“外”= 样本外（2019 年以后，只检验）。",
             "分数 0 = 极度恐慌，100 = 极度贪婪。", "",
             "- 识别率：回撤超过 15% 的下跌中，分数降到 30 以下的比例。",
             "- 报警时回撤：分数第一次降到 30 时，市场已从高点跌了多少（越小越早）。",
             "- 误报：分数跌破 20 但市场前后都没跌超过 5% 的次数 / 跌破 20 的总次数。",
             "- 恐慌天数占比：分数 ≤30 的天数比例，太高说明指数整天喊恐慌。",
             "- 急跌日降分：单日跌 3% 以上时，分数当天平均下降多少。",
             "- 减仓夏普增量、减仓回撤改善：按“≤40 且比 20 日前低就半仓”操作，相对一直满仓的夏普比率变化和最大回撤减少（百分点）。",
             "- 最后几列是具体日子的分数。", ""]
    for region, df in tables.items():
        lines += [f"## {names[region]}", ""]
        groups = [("识别与报警", ["识别率内", "识别率外", "报警时回撤内", "报警时回撤外", "误报内", "误报外"]),
                  ("灵敏度", ["恐慌天数占比内", "恐慌天数占比外", "急跌日降分内", "急跌日降分外"]),
                  ("仓位规则", ["减仓夏普增量内", "减仓夏普增量外", "减仓回撤改善内", "减仓回撤改善外"]),
                  ("具体日子", [d[5:] for d in CHECK_DAYS])]
        for title, cols in groups:
            lines += [f"### {title}", "", "| 方案 | " + " | ".join(cols) + " |", "| --- " * (len(cols) + 1) + "|"]
            for name, row in df.iterrows():
                lines.append(f"| {name} | " + " | ".join(str(row[c]) for c in cols) + " |")
            lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
