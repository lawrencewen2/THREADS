"""早盘放量实时监测（9:25–10:30）：集合竞价 + 早盘量比，10:30 按回测选出的规则定当天的名单。

- 每天早上 astock.yml 跑完选股后由 scripts/astock_insight.py 调 write_ref()，写 docs/astock/early_ref.json：
  每只可交易股票过去 5 日平均成交额、涨跌停幅度、名称
- 盘中循环（scripts/run_intraday.py）每轮把全市场腾讯行情交给 update()，写 intraday 分支的 early.json
- 9:25–9:30 记下集合竞价成交额（这时的累计成交额就是竞价成交额），10:30 之后第一轮把名单固定下来，
  买入参考价 = 那一刻的价格，止损止盈按回测选出的比例
- 每天 10:30 的全市场快照存进 early_log/，以后用来检验竞价成交量（历史上没有这项数据）

只依赖 numpy / pandas，盘中工作流不用装 baostock。
"""

import gzip
import json
import logging
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

REF_FILE = config.PAGE_DIR / "early_ref.json"
MODEL_FILE = Path(__file__).resolve().parent / "early_volume_model.json"
FREEZE = time(10, 30)
AUCTION_START, AUCTION_END = time(9, 25), time(9, 30)
SHOW = 30  # 页面上列出量比最高的多少只


def write_ref(kline, basic):
    """最近一个交易日收盘后的参考数据：只放能交易的股票（非 ST、上市满半年、近 20 日日均成交额 ≥ 5000 万）。"""
    last = kline["date"].max()
    k = kline[kline["date"] > last - pd.Timedelta(days=45)].sort_values(["code", "date"])
    k = k[k["code"].str[:5].isin(config.BOARD_PREFIXES)]
    g = k.groupby("code")
    tail5 = g.tail(5).groupby("code")["amount"].agg(["mean", "size"])
    tail20 = g.tail(20).groupby("code")["amount"].mean()
    lastrow = g.tail(1).set_index("code")
    info = basic.set_index("code")
    ipo = pd.to_datetime(info["ipoDate"], errors="coerce")
    codes = {}
    for code, row in lastrow.iterrows():
        if row["date"] != last or row["isST"] == 1 or tail5.loc[code, "size"] < 5:
            continue
        if tail20.get(code, 0) < config.MIN_AMOUNT_20D:
            continue
        if code not in ipo.index or (last - ipo[code]).days < config.MIN_LISTED_DAYS:
            continue
        pct = 0.2 if code.startswith(("sh.68", "sz.30")) else 0.1
        codes[code] = [round(float(tail5.loc[code, "mean"]), 0), pct, str(info.loc[code, "code_name"])]
    ref = dict(signal_date=str(last.date()), codes=codes)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    REF_FILE.write_text(json.dumps(ref, ensure_ascii=False, separators=(",", ":")))
    return ref


def _minutes(now_cn):
    t = now_cn.hour * 60 + now_cn.minute + now_cn.second / 60
    return min(max(t - (9 * 60 + 30), 0), 120) + min(max(t - 13 * 60, 0), 120)


def features(ref, quotes, now_cn):
    """quotes 来自 live_market.fetch_quotes（含 open/high/low，amount 单位元）。返回当天每只股票的早盘特征。"""
    codes = ref["codes"]
    q = quotes[quotes["code"].isin(codes.keys()) & (quotes["time"].dt.date == now_cn.date())].copy()
    q = q[(q["preclose"] > 0) & (q["open"] > 0) & (q["close"] > 0)]
    if q.empty:
        return q
    q["name"] = q["code"].map(lambda c: codes[c][2])
    amt5 = q["code"].map(lambda c: codes[c][0])
    pct = q["code"].map(lambda c: codes[c][1])
    mins = max(_minutes(now_cn), 1.0)
    q["lb"] = (q["amount"] / mins) / (amt5 / 240)
    q["gap"] = q["open"] / q["preclose"] - 1
    q["r1"] = q["close"] / q["open"] - 1
    q["rp"] = q["close"] / q["preclose"] - 1
    rng = (q["high"] - q["low"]).replace(0, np.nan)
    q["pos"] = ((q["close"] - q["low"]) / rng).fillna(0.5)
    q["up_px"] = (q["preclose"] * (1 + pct) + 1e-6).round(2)
    q["at_limit"] = q["close"] >= q["up_px"] - 0.005
    q["amt5"] = amt5
    return q


def _in(v, rng):
    lo, hi = rng
    return (v >= lo) & (v < hi)


def match(q, model, breadth):
    r = model["rule"]
    lbmax = r.get("lbmax") or np.inf
    ok = (q["lb"] >= r["lbmin"]) & (q["lb"] < lbmax) & _in(q["gap"], model["gap_range"]) & _in(q["r1"], model["r1_range"])
    ok &= _in(q["pos"], model["pos_range"]) & ~q["at_limit"]
    lo, hi = model["gate_range"]
    if not (lo <= breadth < hi):
        ok &= False
    return q[ok].sort_values("lb", ascending=False)


def _row(r, auction):
    a = auction.get(r["code"])
    return dict(code=r["code"], name=r["name"], price=round(float(r["close"]), 2), pct=round(float(r["rp"]) * 100, 2),
                gap=round(float(r["gap"]) * 100, 2), lb=round(float(r["lb"]), 2), r1=round(float(r["r1"]) * 100, 2),
                pos=round(float(r["pos"]), 2),
                auction_ratio=None if a is None else round(a / (float(r["amt5"]) / 240), 2))


def update(out, ref, quotes, now_cn, model=None):
    """写 out/early.json；返回新出的提醒文字列表（名单固定、触及止损/止盈）。"""
    out = Path(out)
    path = out / "early.json"
    day = str(now_cn.date())
    old = json.loads(path.read_text()) if path.exists() else {}
    if old.get("date") != day:
        old = dict(date=day, auction={}, picks=[], frozen=False)
    if ref.get("signal_date", "") >= day or now_cn.time() < AUCTION_START:
        return []
    q = features(ref, quotes, now_cn)
    if q.empty:
        return []
    notes = []
    st = old
    # 集合竞价：9:25–9:30 之间第一次看到的累计成交额就是竞价成交额
    if AUCTION_START <= now_cn.time() < AUCTION_END and not st["auction"]:
        st["auction"] = {c: float(a) for c, a in zip(q["code"], q["amount"]) if a > 0}
    auction = st["auction"]
    breadth = float((q["rp"] > 0).mean())
    phase = "集合竞价" if now_cn.time() < AUCTION_END else ("监测中" if now_cn.time() < FREEZE else "已定名单")
    if now_cn.time() >= time(15, 0):
        phase = "已收盘"
    cand = match(q, model, breadth) if model else q.iloc[0:0]
    top_n = (model or {}).get("top_n", 5)
    if not st.get("snapshot") and FREEZE <= now_cn.time() < time(11, 30):
        _snapshot(out, q, auction, now_cn)  # 不管有没有规则都存 10:30 快照，以后检验竞价成交量
        st["snapshot"] = now_cn.strftime("%H:%M")
    if model and not st["frozen"] and FREEZE <= now_cn.time() < time(11, 30):
        r = model["rule"]
        st["picks"] = [dict(_row(x, auction), entry=round(float(x["close"]), 2),
                            stop=round(float(x["close"]) * (1 - r["sl"]), 2), take=round(float(x["close"]) * (1 + r["tp"]), 2))
                       for _, x in cand.head(top_n).iterrows()]
        st["frozen"] = True
        st["frozen_at"] = now_cn.strftime("%H:%M")
        st["breadth_at_freeze"] = round(breadth, 3)
        names = "、".join(f"{p['name']}（{p['code']}）" for p in st["picks"]) or "无（今天没有股票符合条件）"
        notes.append((f"early:{day}:freeze", f"早盘放量名单 {st['frozen_at']}：{names}"))
    # 名单固定后跟踪现价、止损止盈（T+1，当天只能看不能卖）
    px = dict(zip(q["code"], q["close"]))
    for p in st["picks"]:
        if p["code"] in px:
            p["now"] = round(float(px[p["code"]]), 2)
            p["from_entry"] = round((p["now"] / p["entry"] - 1) * 100, 2)
            p["status"] = ("触及止损" if p["now"] <= p["stop"] else "触及止盈" if p["now"] >= p["take"] else "持有")
    show = q.sort_values("lb", ascending=False)
    show = show[~show["at_limit"]].head(SHOW)
    auc = sorted(((a / (q.set_index("code").loc[c, "amt5"] / 240), c) for c, a in auction.items()
                  if c in set(q["code"])), reverse=True)[:SHOW]
    qi = q.set_index("code")
    st.update(
        time=now_cn.strftime("%m-%d %H:%M"), phase=phase, breadth=round(breadth, 3), stocks=int(len(q)),
        top_volume=[_row(r, auction) for _, r in show.iterrows()],
        top_auction=[dict(code=c, name=qi.loc[c, "name"], auction_ratio=round(float(v), 2),
                          gap=round(float(qi.loc[c, "gap"]) * 100, 2), pct=round(float(qi.loc[c, "rp"]) * 100, 2))
                     for v, c in auc],
        matching=[_row(r, auction) for _, r in cand.head(SHOW).iterrows()],
        rule=None if not model else dict(text=model.get("rule_text"), sl=model["rule"]["sl"], tp=model["rule"]["tp"],
                                          oos=model.get("oos"), oos_port=model.get("oos_port")),
    )
    path.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    return notes


def _snapshot(out, q, auction, now_cn):
    """10:30 全市场快照，供以后检验竞价成交量等历史上没有的数据。"""
    d = out / "early_log"
    d.mkdir(parents=True, exist_ok=True)
    s = q[["code", "close", "preclose", "open", "high", "low", "amount", "amt5", "lb", "gap", "r1", "pos"]].copy()
    s["auction_amount"] = s["code"].map(auction)
    s["time"] = now_cn.strftime("%H:%M:%S")
    with gzip.open(d / f"{now_cn:%Y-%m-%d}.csv.gz", "wt") as f:
        s.to_csv(f, index=False, float_format="%.4f")


def issue_lines(st):
    if not st or not st.get("phase"):
        return []
    lines = ["", f"**早盘放量监测**（{st['time']}，{st['phase']}；上涨家数占比 {st['breadth'] * 100:.0f}%）", ""]
    if st.get("frozen"):
        lines += [f"10:30 名单（{st.get('frozen_at')} 固定，T+1 次日卖出）：", "",
                  "| 股票 | 买入参考 | 现价 | 相对买入 | 止损 | 止盈 | 量比 | 竞价 | 状态 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for p in st["picks"]:
            lines.append(f"| {p['name']} {p['code']} | {p['entry']} | {p.get('now', '—')} | {p.get('from_entry', 0):+.2f}% | "
                         f"{p['stop']} | {p['take']} | {p['lb']} | {p['gap']:+.2f}% | {p.get('status', '持有')} |")
        if not st["picks"]:
            lines.append("| 今天没有符合条件的股票 | | | | | | | | |")
    else:
        if not st.get("rule"):
            lines += ["回测（2021 年起）没找到扣成本后能赚钱的量比/竞价规则，这里只观察、不出买入名单；每天记竞价成交量，攒够数据再测。", ""]
        lines += ["量比最高（随时变化，仅供观察）：", "", "| 股票 | 现价 | 今日 | 量比 | 竞价 | 符合规则 |", "| --- | --- | --- | --- | --- | --- |"]
        hits = {r["code"] for r in st.get("matching", [])}
        for r in st.get("top_volume", [])[:10]:
            lines.append(f"| {r['name']} {r['code']} | {r['price']} | {r['pct']:+.2f}% | {r['lb']} | {r['gap']:+.2f}% | "
                         f"{'是' if r['code'] in hits else ''} |")
    return lines
