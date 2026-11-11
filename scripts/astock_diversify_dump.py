"""分散化研究的输入数据（2026-11-10 用户：寻找扩大年化同时减少回撤的策略）。

1. 线上选股规则（slot_model.json）在 100% 资金下的每日收益（训练期 2015–2020、样本外 2021 至今各跑一次），
   叠加前、叠加“净值跌破 20 日均线空仓”后都存，另存每天持有几只（算闲置资金）。
2. 场内 ETF 的前复权日线（腾讯行情，失败时用东方财富），用于 ETF 轮动 / 趋势跟踪分仓。
写 data/research/live_daily.csv、data/research/etf/*.csv。只读行情缓存，不拉 baostock 行情。
"""

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent.parent / "data" / "research"
log = logging.getLogger("dump")


def forecast_rows_before(data, year=2021):
    fc = data.load_forecast()
    if fc is None or not len(fc):
        return 0
    return int((fc["pub"].dt.year < year).sum())


def dump_live():
    from astock import config, data, model
    from astock.alt_study import emotion_regime, iter_alt_factors
    from astock.factors import Panel, eligible
    from astock.live_slots import MODEL_FILE
    from astock.slots import simulate
    from astock.stability import overlay
    from astock.winrate import adjusted_ohlc

    basic = data.load_basic()
    log.info("forecast rows before 2021: %d", forecast_rows_before(data))
    sm = json.loads(MODEL_FILE.read_text())
    weights = {k: v for k, v in sm["weights"].items() if v}
    kline, valuation, forecast = data.load_kline(), data.load_valuation(), data.load_forecast()
    p = Panel(kline, valuation)
    del kline, valuation
    elig = eligible(p, basic)
    ranks = {}
    for k, v in iter_alt_factors(p, forecast):
        if k in weights:
            ranks[k] = model.xs_rank(v, elig)
        del v
    sc = model.score(ranks, weights, elig)
    del ranks
    gate = emotion_regime(p, elig)
    ohl = adjusted_ohlc(p)
    end = str(p.dates[-1].date())
    parts, trades = [], []
    for name, (s, e) in dict(train=(config.TRAIN_START, config.TRAIN_END), test=(config.TEST_START, end)).items():
        d, t = simulate(p, ohl, sc, gate, sm["n"], sm["take_profit"], sm["stop_loss"], sm["max_hold"], s, e)
        df = pd.DataFrame({"raw": d, "live": overlay(d, sm["overlay"])})
        df["period"] = name
        held = pd.Series(0, index=d.index)
        for x in t:
            held[(held.index >= x["entry"]) & (held.index < x["exit"])] += 1
        df["held"] = held
        parts.append(df)
        trades += [dict(x, period=name) for x in t]
        log.info("%s: %d days, %d trades", name, len(d), len(t))
    out = pd.concat(parts)
    out.index.name = "date"
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "live_daily.csv", float_format="%.6g")
    pd.DataFrame(trades).to_csv(OUT / "live_trades.csv.gz", index=False, float_format="%.6g")


ALT = {  # 第二个选股策略候选：和业绩预期选股低相关的风格
    "lowvol_value": {"ep": 1, "bp": 1, "vol20": -1, "idiovol60": -1},
    "small_rev": {"size": -1, "rev20": -1, "turn20": -1, "idiovol60": -1},
    "momentum": {"mom60": 1, "high250": 1},
    "value": {"ep": 1, "bp": 1, "sp": 1},
}
ALT_RULES = [(None, "atr3", 20), ("atr2", "atr3", 10), ("atr3", 0.08, 20)]


def dump_alt():
    """几种别的选股风格，同一套 20 只 + 止盈止损 + 净值均线空仓的模拟，存逐日收益。"""
    import numpy as np
    from astock import config, data, model
    from astock.alt_study import emotion_regime
    from astock.factors import Panel, eligible, iter_factors
    from astock.slots import simulate
    from astock.stability import overlay
    from astock.winrate import adjusted_ohlc

    basic = data.load_basic()
    p = Panel(data.load_kline(), data.load_valuation())
    elig = eligible(p, basic)
    need = {k for w in ALT.values() for k in w}
    ranks = {}
    for k, v in iter_factors(p):
        if k in need:
            ranks[k] = model.xs_rank(v, elig)
        del v
    log.info("factors: %s", sorted(ranks))
    gate_em = emotion_regime(p, elig)
    gate_all = pd.Series(True, index=p.dates)
    ohl = adjusted_ohlc(p)
    end = str(p.dates[-1].date())
    cols = {}
    for name, w in ALT.items():
        sc = model.score(ranks, w, elig)
        for tp, sl, hold in ALT_RULES:
            for gname, gate in (("em", gate_em), ("all", gate_all)):
                key = f"{name}|{tp}-{sl}-{hold}|{gname}"
                parts = []
                for s, e in ((config.TRAIN_START, config.TRAIN_END), (config.TEST_START, end)):
                    d, t = simulate(p, ohl, sc, gate, 20, tp, sl, hold, s, e)
                    held = pd.Series(0, index=d.index)
                    for x in t:
                        held[(held.index >= x["entry"]) & (held.index < x["exit"])] += 1
                    parts.append(pd.DataFrame({key + "|live": overlay(d, "trend20"), key + "|held": held}))
                df = pd.concat(parts)
                for c in df:
                    cols[c] = df[c]
                log.info("%s: train/test days %d", key, len(df))
        del sc
    out = pd.DataFrame(cols)
    out.index.name = "date"
    out.to_csv(OUT / "alt_daily.csv.gz", float_format="%.6g")


# 代码、名称、类别。全部是普通股票账户能买的场内基金。
ETFS = [
    ("sh510300", "沪深300ETF", "A股大盘"), ("sh510500", "中证500ETF", "A股中盘"), ("sh512100", "中证1000ETF", "A股小盘"),
    ("sz159915", "创业板ETF", "A股成长"), ("sh510880", "红利ETF", "A股红利"), ("sz159920", "恒生ETF", "港股"),
    ("sh513050", "中概互联ETF", "港美中概"), ("sh513100", "纳指ETF", "美股"), ("sh513500", "标普500ETF", "美股"),
    ("sh513520", "日经ETF", "日股"), ("sh518880", "黄金ETF", "黄金"), ("sz159985", "豆粕ETF", "商品"),
    ("sh511010", "国债ETF", "债券"), ("sh511260", "十年国债ETF", "债券"), ("sh511220", "城投债ETF", "债券"),
    ("sh511880", "银华日利", "货币"),
]
UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}


def tencent(session, sym, start=2012):
    rows = []
    for y in range(start, pd.Timestamp.today().year + 1, 2):
        url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,{y}-01-01,{y + 1}-12-31,640,qfq")
        r = session.get(url, headers=UA, timeout=20)
        r.raise_for_status()
        body = r.json()["data"][sym]
        k = body.get("qfqday") or body.get("day") or []
        rows += [x[:6] for x in k]
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"]).drop_duplicates("date")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date")


def eastmoney(session, sym):
    secid = ("1." if sym.startswith("sh") else "0.") + sym[2:]
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?fields1=f1&fields2=f51,f52,f53,f54,f55,f56"
           f"&klt=101&fqt=1&beg=20100101&end=20500101&secid={secid}")
    r = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    k = r.json()["data"]["klines"]
    df = pd.DataFrame([x.split(",") for x in k], columns=["date", "open", "close", "high", "low", "volume"])
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


class _Resp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return json.loads(self.body)


class _Session:
    """标准库 urllib，astock 的依赖里没有 requests。"""

    def get(self, url, headers=None, timeout=20):
        import urllib.request
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _Resp(r.read().decode("utf-8", errors="replace"))


def dump_etf():
    s = _Session()
    d = OUT / "etf"
    d.mkdir(parents=True, exist_ok=True)
    meta = []
    for sym, name, cat in ETFS:
        df, src = None, None
        for fn in (tencent, eastmoney):
            try:
                df, src = fn(s, sym), fn.__name__
                if len(df) > 100:
                    break
            except Exception as exc:
                log.warning("%s %s failed: %s", sym, fn.__name__, exc)
        if df is None or len(df) < 100:
            log.warning("%s: no data", sym)
            continue
        df.to_csv(d / f"{sym}.csv", index=False)
        meta.append(dict(code=sym, name=name, cat=cat, src=src, first=df["date"].iloc[0], last=df["date"].iloc[-1], n=len(df)))
        log.info("%s %s: %d rows from %s (%s→%s)", sym, name, len(df), src, df["date"].iloc[0], df["date"].iloc[-1])
    (d / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "--ensure-forecast" in sys.argv:
        from astock import data
        old = forecast_rows_before(data)
        log.info("forecast rows before 2021: %d", old)
        if old < 2000:
            df, failed = data.update_forecast(data.stock_codes(data.load_basic()), workers=2, merge=True)
            log.info("forecast refetched: %d rows, %d failed, before 2021 now %d", len(df), len(failed),
                     forecast_rows_before(data))
        return
    if "--etf" in sys.argv:
        dump_etf()
    if "--live" in sys.argv:
        dump_live()
    if "--alt" in sys.argv:
        dump_alt()


if __name__ == "__main__":
    main()
