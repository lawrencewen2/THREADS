"""盘中做T研究（2026-11-10 用户：80% 资金按选股规则操作，剩下 20% 用来实时做T，降低持仓成本、减小回撤）。

A股 T+1：当天买的股票当天不能卖，所以做T只能在“已经持有、今天可卖”的股票上做，日内买卖数量相同，收盘后持股数不变：
- 正T（先买后卖）：盘中跌到 参考价×(1−k×ATR) 时用备用资金买进 q 份，之后涨到 买价×(1+g×ATR) 卖出同样数量的老股；
- 倒T（先卖后买）：盘中涨到 参考价×(1+k×ATR) 时卖出 q 份老股，之后跌到 卖价×(1−g×ATR) 买回；
- 双向：两个单子同时挂，先成交的那边算数，一天最多做一次；
- 每笔都有止损：反方向走 s×ATR 就按止损价了结；14:55 还没了结的按 14:55 收盘价了结（保证当天股数不变）；
- 14:00 以后不再开新的T。参考价 = 昨收或今天开盘价；ATR = 买入前一天的 14 日相对 ATR（和选股规则同一个）。

成交按 5 分钟K线模拟：限价单在某根K线最低价（最高价）碰到挂单价时成交，开盘就越过挂单价按开盘价成交；
同一根K线里先后顺序不知道，所以成交那根不检查了结，之后每根先看止损再看止盈（偏保守）。
每边扣佣金 + 滑点，卖出那边再扣印花税（config）。另做一档滑点减半的敏感性检验。

样本：
- 训练（只在这里挑规则）：随机抽 RANDOM_N 只当时满足选股可交易条件的股票，2021–2024 年每个交易日都当作持有；
- 样本外 1：同一批随机股票 2025 年至今；
- 样本外 2：线上选股规则（业绩预期 + 情绪开关、止盈 2 倍 ATR、止损 3 倍 ATR、最长 10 天、净值跌破 20 日均线空仓）
  2025 年至今真实会持有的股票，只在买入后第二天到卖出前一天之间做T（买入当天不能卖、卖出那天老股已经卖掉）。
资金：选股 80%（每只 4%），做T每次 q = 半份（总资金 2%），正T同时最多 RESERVE/每次金额 笔（备用 20% 资金封顶）。
"""

import json
import logging
from itertools import product

import numpy as np
import pandas as pd

from . import config, data, model, portfolio

log = logging.getLogger(__name__)

M5_DIR = config.DATA_DIR / "m5"  # 5 分钟线缓存，单独一个 actions/cache
M5_FILE = M5_DIR / "m5.pkl.gz"
M5_FIELDS = "date,time,code,open,high,low,close,volume,amount"
M5_START = "2021-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
RANDOM_N = 150
SEED = 7
BARS = 48  # 9:35 … 11:30、13:05 … 15:00
LAST_ENTRY = 35  # 第 36 根 = 14:00 那根，之后不开新的T
EXIT_BAR = 46  # 14:55 那根收盘了结
SHARE = 0.8  # 选股用的资金
RESERVE = 0.2
N_SLOTS = 20
Q = 0.5  # 每次做T的数量 = 半份持仓
REPORT_FILE = config.REPORT_DIR / "t0.md"
PAGE_FILE = config.PAGE_DIR / "t0.json"

SIDES = ("正T", "倒T", "双向")
REFS = ("昨收", "开盘")
K_GRID = (0.3, 0.5, 0.75, 1.0)
G_GRID = (0.25, 0.5, 0.75, 1.0)
S_GRID = (0.5, 1.0, None)  # None = 不设止损，只在 14:55 了结


# ---------------------------------------------------------------- 线上规则的持仓

def live_holdings(kline, basic, valuation, forecast):
    """按线上 slot_model.json 跑一遍影子组合：返回 (每日收益, 空仓开关, 交易列表, 面板, ATR)。"""
    from .alt_study import emotion_regime, iter_alt_factors
    from .factors import Panel, eligible
    from .live_slots import load_model
    from .slots import simulate
    from .winrate import adjusted_ohlc

    sm = load_model()
    kline = kline[kline["date"] >= pd.Timestamp("2019-06-01")]
    if valuation is not None:
        valuation = valuation[valuation["date"] >= pd.Timestamp("2019-06-01")]
    p = Panel(kline, valuation)
    elig = eligible(p, basic)
    weights = {k: v for k, v in sm["weights"].items() if v}
    ranks = {k: model.xs_rank(v, elig) for k, v in iter_alt_factors(p, forecast) if k in weights}
    sc = model.score(ranks, weights, elig)
    gate = emotion_regime(p, elig)
    ohl = adjusted_ohlc(p)
    st = {}
    daily, trades = simulate(p, ohl, sc, gate, sm["n"], sm["take_profit"], sm["stop_loss"], sm["max_hold"],
                             config.TEST_START, str(p.dates[-1].date()), state=st)
    for j, (_, _, _, d0) in st["held"].items():  # 还没卖的：做T做到最后一天
        trades.append(dict(entry=p.dates[d0], exit=None, code=p.codes[j], net=np.nan))
    eq = (1 + daily).cumprod()
    pos = (eq >= eq.rolling(20, min_periods=10).mean()).astype(float).shift(2).fillna(1.0)
    atr = pd.DataFrame(ohl[3], index=p.dates, columns=p.codes)
    return daily, pos, trades, p, atr, elig


def hold_days(trades, dates):
    """(code, date) 可以做T的日子：买入后第二天起，到卖出前一天。"""
    rows = []
    last = dates[-1]
    for t in trades:
        i0 = dates.searchsorted(t["entry"]) + 1
        i1 = dates.searchsorted(t["exit"]) if t["exit"] is not None else len(dates)
        for d in dates[i0:i1]:
            rows.append((t["code"], d))
    out = pd.DataFrame(rows, columns=["code", "date"]).drop_duplicates()
    return out[out["date"] <= last]


def random_codes(elig, n=RANDOM_N, seed=SEED):
    """2021 年至今至少一半交易日满足可交易条件的股票里随机抽 n 只。"""
    e = elig[elig.index >= pd.Timestamp(M5_START)]
    ok = e.mean() >= 0.5
    pool = sorted(ok[ok].index)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(pool, size=min(n, len(pool)), replace=False))


# ---------------------------------------------------------------- 5 分钟线

def _fetch_m5_chunk(job):
    items = job
    parts, failed = [], []
    try:
        sess = data._session()
    except Exception as exc:
        log.warning("baostock unavailable: %s", exc)
        return parts, [c for c, _, _ in items]
    for n, (code, start, end) in enumerate(items):
        try:
            df = sess.query("query_history_k_data_plus", code, M5_FIELDS, start_date=start, end_date=end,
                            frequency="5", adjustflag="3")
        except data.BaostockBlocked as exc:
            log.warning("baostock blocked: %s", exc)
            return parts, failed + [c for c, _, _ in items[n:]]
        except Exception as exc:
            log.warning("m5 %s failed: %s", code, exc)
            failed.append(code)
            continue
        if len(df):
            parts.append(clean_m5(df))
    return parts, failed


def clean_m5(df):
    out = pd.DataFrame({"date": pd.to_datetime(df["date"]), "code": df["code"].values,
                        "hm": df["time"].str[8:12].values})
    for c in ("open", "high", "low", "close", "amount"):
        out[c] = pd.to_numeric(df[c].values, errors="coerce").astype("float32")
    return out


def load_m5():
    return pd.read_pickle(M5_FILE) if M5_FILE.exists() else None


def update_m5(items, workers=config.FETCH_WORKERS):
    """items: [(code, start, end)]。缓存已经覆盖的区间跳过；每 20 批落盘一次，超时也不丢已下载的。"""
    from multiprocessing import Pool

    old = load_m5()
    if old is not None and len(old):
        rng = old.groupby("code")["date"].agg(["min", "max"])
        keep = []
        for c, a, b in items:
            if c in rng.index and rng.at[c, "min"] <= pd.Timestamp(a) + pd.Timedelta(days=7) and rng.at[c, "max"] >= min(
                    pd.Timestamp(b), rng["max"].max()):
                continue
            if c in rng.index and rng.at[c, "min"] <= pd.Timestamp(a) + pd.Timedelta(days=7):
                a = (rng.at[c, "max"] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # 只往后补
            keep.append((c, a, b))
        items = [(c, a, b) for c, a, b in keep if a <= b]
    log.info("m5: %d queries", len(items))
    jobs = [items[i:i + 10] for i in range(0, len(items), 10)]
    parts, failed = [], []
    M5_DIR.mkdir(parents=True, exist_ok=True)
    if not jobs:
        return old, failed
    with Pool(max(1, workers), initializer=data._init_worker) as pool:
        for k, (got, bad) in enumerate(pool.imap_unordered(_fetch_m5_chunk, jobs), 1):
            parts += got
            failed += bad
            log.info("m5 chunks %d/%d", k, len(jobs))
            if k % 20 == 0 or k == len(jobs):
                old = pd.concat(([old] if old is not None else []) + parts, ignore_index=True)
                old = old.drop_duplicates(["code", "date", "hm"], keep="last")
                old.to_pickle(M5_FILE)
                parts = []
    if len(failed) > 0.3 * max(len(items), 1):
        raise data.BaostockBlocked(f"m5: {len(failed)} of {len(items)} failed")
    return old, failed


def fetch_items(ctx, end):
    """要拉的 5 分钟线：随机股票 2021 年至今一段；线上持仓每笔从买入到卖出一段（重叠的合并）。"""
    daily, pos, trades, p, atr_df, elig = ctx
    rc = random_codes(elig)
    items = [(c, M5_START, end) for c in rc]
    spans = {}
    for t in trades:
        if t["entry"] < pd.Timestamp(TEST_START) or t["code"] in rc:
            continue
        a = t["entry"]
        b = t["exit"] if t["exit"] is not None else pd.Timestamp(end)
        spans.setdefault(t["code"], []).append((a, b))
    for c, ss in spans.items():
        ss.sort()
        cur = list(ss[0])
        for a, b in ss[1:]:
            if a <= cur[1] + pd.Timedelta(days=10):
                cur[1] = max(cur[1], b)
            else:
                items.append((c, cur[0].strftime("%Y-%m-%d"), cur[1].strftime("%Y-%m-%d")))
                cur = [a, b]
        items.append((c, cur[0].strftime("%Y-%m-%d"), cur[1].strftime("%Y-%m-%d")))
    return items


def day_arrays(m5, keys):
    """把 5 分钟线排成 (天数, 48) 的数组；keys = DataFrame[code, date]，K线不满 48 根的天丢掉。"""
    m = m5.merge(keys[["code", "date"]], on=["code", "date"])
    m = m.sort_values(["code", "date", "hm"])
    cnt = m.groupby(["code", "date"])["hm"].transform("size")
    m = m[cnt == BARS]
    idx = m[["code", "date"]].iloc[::BARS].reset_index(drop=True)
    arr = {c: m[c].to_numpy(dtype="float64").reshape(-1, BARS) for c in ("open", "high", "low", "close")}
    return idx, arr


# ---------------------------------------------------------------- 模拟

def _first(mask, start=None):
    """每行第一个 True 的列号（没有给 BARS）；start 给每行起始列（不含之前的列）。"""
    if start is not None:
        cols = np.arange(mask.shape[1])
        mask = mask & (cols[None, :] >= start[:, None])
    hit = mask.any(1)
    return np.where(hit, mask.argmax(1), BARS)


def simulate_t(arr, ref, atr, side, k, g, s, slip=config.SLIPPAGE, stamp=config.STAMP_DUTY_NEW):
    """每天最多一次T。返回 (成交了吗, 每笔净收益（按做T金额）, 正T吗)。"""
    o, h, l, c = arr["open"], arr["high"], arr["low"], arr["close"]
    n = len(ref)
    cols = np.arange(BARS)
    can = cols[None, :] <= LAST_ENTRY
    buy_lim = ref * (1 - k * atr)
    sell_lim = ref * (1 + k * atr)
    ib = _first((l <= buy_lim[:, None]) & can) if side in ("正T", "双向") else np.full(n, BARS)
    is_ = _first((h >= sell_lim[:, None]) & can) if side in ("倒T", "双向") else np.full(n, BARS)
    longside = ib <= is_  # 同一根都碰到：按正T算
    fill_i = np.where(longside, ib, is_)
    done = fill_i < BARS
    fi = np.minimum(fill_i, BARS - 1)
    rows = np.arange(n)
    op = o[rows, fi]
    px_in = np.where(longside, np.minimum(op, buy_lim), np.maximum(op, sell_lim))
    # 正T：止盈在上、止损在下；倒T 相反
    tgt = np.where(longside, px_in * (1 + g * atr), px_in * (1 - g * atr))
    stp = None if s is None else np.where(longside, px_in * (1 - s * atr), px_in * (1 + s * atr))
    nxt = fi + 1
    win = cols[None, :] <= EXIT_BAR
    t_hit = np.where(longside[:, None], h >= tgt[:, None], l <= tgt[:, None]) & win
    it = _first(t_hit, nxt)
    if stp is not None:
        s_hit = np.where(longside[:, None], l <= stp[:, None], h >= stp[:, None]) & win
        isl = _first(s_hit, nxt)
    else:
        isl = np.full(n, BARS)
    stop_first = isl <= it
    ex_i = np.minimum(np.where(stop_first, isl, it), EXIT_BAR)
    forced = np.minimum(isl, it) >= BARS
    oe = o[rows, ex_i]
    px_t = np.where(longside, np.maximum(oe, tgt), np.minimum(oe, tgt))
    px_s = np.where(longside, np.minimum(oe, stp), np.maximum(oe, stp)) if stp is not None else px_t
    px_out = np.where(forced, c[rows, EXIT_BAR], np.where(stop_first, px_s, px_t))
    buy_px = np.where(longside, px_in, px_out)
    sell_px = np.where(longside, px_out, px_in)
    cb = slip + config.COMMISSION
    cs = slip + config.COMMISSION + stamp
    net = sell_px * (1 - cs) / (buy_px * (1 + cb)) - 1
    return done, np.where(done, net, 0.0), longside


def rule_name(side, ref, k, g, s):
    stop = "不止损（14:55 了结）" if s is None else f"止损 {s:g}×ATR"
    return f"{side}｜参考{ref}｜挂单 {k:g}×ATR｜止盈 {g:g}×ATR｜{stop}"


def evaluate(idx, arr, refs, atr, slip=config.SLIPPAGE):
    """全部规则在一个样本上的结果：每个持有日的平均净收益（按做T金额）、成交率、胜率。"""
    stamp = np.where(idx["date"] >= pd.Timestamp(config.STAMP_DUTY_CHANGE), config.STAMP_DUTY_NEW, config.STAMP_DUTY_OLD)
    rows = []
    for side, ref, k, g, s in product(SIDES, REFS, K_GRID, G_GRID, S_GRID):
        done, net, _ = simulate_t(arr, refs[ref], atr, side, k, g, s, slip, stamp)
        f = done.sum()
        rows.append(dict(side=side, ref=ref, k=k, g=g, s=s, days=len(done), fills=int(f),
                         fill_rate=float(f / max(len(done), 1)),
                         per_day=float(net.mean()) if len(net) else 0.0,
                         per_trade=float(net[done].mean()) if f else 0.0,
                         win=float((net[done] > 0).mean()) if f else 0.0))
    return pd.DataFrame(rows)


def prep(idx, arr, p, atr_df):
    """参考价（昨收、开盘）和前一天 ATR；跳过一字板、开盘涨跌停和没有 ATR 的天。"""
    di = p.dates.get_indexer(idx["date"])
    ci = p.codes.get_indexer(idx["code"])
    ok = (di > 0) & (ci >= 0)
    pre = np.full(len(idx), np.nan)
    a = np.full(len(idx), np.nan)
    up, dn = p.limit_prices()
    upv, dnv = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
    pre[ok] = p.preclose.to_numpy()[di[ok], ci[ok]]
    a[ok] = atr_df.to_numpy()[di[ok] - 1, ci[ok]]
    upv[ok] = up.to_numpy()[di[ok], ci[ok]]
    dnv[ok] = dn.to_numpy()[di[ok], ci[ok]]
    op = arr["open"][:, 0]
    hi, lo = arr["high"].max(1), arr["low"].min(1)
    good = ok & np.isfinite(pre) & np.isfinite(a) & (a > 0) & (hi > lo) & (op < upv - 0.005) & (op > dnv + 0.005)
    good &= np.abs(op / pre - 1) < 0.25  # 除权日昨收和 5 分钟线对不上，丢掉
    sel = np.flatnonzero(good)
    idx = idx.iloc[sel].reset_index(drop=True)
    arr = {kk: v[sel] for kk, v in arr.items()}
    return idx, arr, {"昨收": pre[sel], "开盘": arr["open"][:, 0]}, a[sel]


def daily_layer(idx, arr, refs, atr, rule, dates, slip=config.SLIPPAGE):
    """选定规则下做T对总资金的每日贡献，正T同时最多 RESERVE / 每次金额 笔。"""
    stamp = np.where(idx["date"] >= pd.Timestamp(config.STAMP_DUTY_CHANGE), config.STAMP_DUTY_NEW, config.STAMP_DUTY_OLD)
    done, net, longside = simulate_t(arr, refs[rule["ref"]], atr, rule["side"], rule["k"], rule["g"], rule["s"], slip, stamp)
    size = SHARE / N_SLOTS * Q
    df = pd.DataFrame(dict(date=idx["date"], code=idx["code"], done=done, net=net, long=longside))
    df = df[df["done"]].copy()
    cap = int(RESERVE / size)
    df["n_long"] = df[df["long"]].groupby("date").cumcount()
    df = df[~(df["long"] & (df["n_long"] >= cap))]
    layer = (df.groupby("date")["net"].sum() * size).reindex(dates).fillna(0.0)
    per_code = df.groupby("code")["net"].sum() * Q  # 每只持仓做T累计：占这只持仓金额的比例
    return layer, df, per_code


def _s(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def run(ctx, m5):
    daily, pos, trades, p, atr_df, elig = ctx
    dates = daily.index
    hd = hold_days(trades, p.dates)
    hd = hd[hd["date"] >= pd.Timestamp(TEST_START)]
    rc = random_codes(elig)
    ek = elig.stack()
    ek = ek[ek].reset_index()
    ek.columns = ["date", "code", "_"]
    rk = ek[ek["code"].isin(rc) & (ek["date"] >= pd.Timestamp(M5_START))][["code", "date"]]

    samples = {}
    for name, keys in (("训练：随机股票 2021–2024", rk[rk["date"] <= pd.Timestamp(TRAIN_END)]),
                       ("样本外：随机股票 2025 至今", rk[rk["date"] >= pd.Timestamp(TEST_START)]),
                       ("样本外：线上规则真实持仓 2025 至今", hd)):
        idx, arr = day_arrays(m5, keys)
        samples[name] = prep(idx, arr, p, atr_df)
        log.info("%s: %d stock-days (of %d)", name, len(samples[name][0]), len(keys))
    names = list(samples)
    grid = {n: evaluate(*samples[n]) for n in names}
    train = grid[names[0]]
    # 训练期挑：成交率 ≥ 10% 里每个持有日平均净收益最高的
    pool = train[train["fill_rate"] >= 0.10]
    bi = pool["per_day"].idxmax()
    best = pool.loc[bi].to_dict()
    best["s"] = _s(best["s"])
    table = train.copy()
    for n in names[1:]:
        g = grid[n]
        table[n + "|per_day"] = g["per_day"].values
        table[n + "|per_trade"] = g["per_trade"].values
        table[n + "|win"] = g["win"].values
        table[n + "|fill_rate"] = g["fill_rate"].values
    # 滑点减半
    half = {n: evaluate(*samples[n], slip=config.SLIPPAGE / 2) for n in names}

    # 组合层面：线上持仓 2025 至今，80% 选股 + 20% 做T
    hidx, harr, hrefs, hatr = samples[names[2]]
    d25 = dates[dates >= pd.Timestamp(TEST_START)]
    base = (daily * pos).reindex(d25).fillna(0) * SHARE
    layers = {}
    for label, sl in (("默认滑点", config.SLIPPAGE), ("滑点减半", config.SLIPPAGE / 2)):
        layer, fills, per_code = daily_layer(hidx, harr, hrefs, hatr, best, d25, sl)
        # 空仓开关关时实际账户没有持仓，也就不做T
        layers[label] = (layer * pos.reindex(d25).fillna(1.0), fills, per_code)
    # 对照：每类方向在训练期最好的那条
    side_best = {}
    for side in SIDES:
        sp = pool[pool["side"] == side]
        if len(sp):
            r = sp.loc[sp["per_day"].idxmax()].to_dict()
            r["s"] = _s(r["s"])
            lay, _, _ = daily_layer(hidx, harr, hrefs, hatr, r, d25)
            side_best[side] = (r, lay * pos.reindex(d25).fillna(1.0))

    def pst(s):
        eq = (1 + s).cumprod()
        dd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
        yrs = max(len(s) / 244, 1e-9)
        ann = float(eq.iloc[-1] ** (1 / yrs) - 1) if len(eq) else 0.0
        vol = float(s.std() * np.sqrt(244))
        return dict(total=float(eq.iloc[-1] - 1) if len(eq) else 0.0, ann=ann, max_dd=dd,
                    sharpe=float(ann / vol) if vol > 0 else 0.0, worst_day=float(s.min()) if len(s) else 0.0)

    combo = {"只持有（80% 选股，20% 现金）": pst(base)}
    curves = {"只持有": (1 + base).cumprod()}
    for label, (lay, fills, per_code) in layers.items():
        combo[f"持有 + 做T（{label}）"] = pst(base + lay)
        curves[f"持有+做T（{label}）"] = (1 + base + lay).cumprod()
    for side, (r, lay) in side_best.items():
        combo[f"持有 + 只做{side}（训练期最好的一条）" if side != "双向" else "持有 + 双向（训练期最好的一条）"] = pst(base + lay)
    lay, fills, per_code = layers["默认滑点"]
    # 下跌日做T贡献：看做T是不是在亏钱的日子帮上忙
    down = base < -0.005
    res = dict(
        updated=str(p.dates[-1].date()), best=best, best_name=rule_name(best["side"], best["ref"], best["k"], best["g"], best["s"]),
        samples={n: int(len(samples[n][0])) for n in names},
        best_by_sample={n: grid[n].loc[bi].to_dict() for n in names},
        best_half_slip={n: half[n].loc[bi].to_dict() for n in names},
        positive_rules={n: int((grid[n]["per_day"] > 0).sum()) for n in names},
        positive_rules_half={n: int((half[n]["per_day"] > 0).sum()) for n in names},
        n_rules=int(len(train)),
        top_train=table.sort_values("per_day", ascending=False).head(15).to_dict("records"),
        side_best={s: dict(rule=rule_name(r["side"], r["ref"], r["k"], r["g"], r["s"]), per_day_train=r["per_day"])
                   for s, (r, _) in side_best.items()},
        combo=combo,
        t_fills=int(len(fills)), t_win=float((fills["net"] > 0).mean()) if len(fills) else 0.0,
        t_avg=float(fills["net"].mean()) if len(fills) else 0.0,
        cost_cut_per_holding=float(per_code.mean()) if len(per_code) else 0.0,
        down_days=int(down.sum()), t_on_down_days=float(lay[down].mean()) if down.any() else 0.0,
        base_on_down_days=float(base[down].mean()) if down.any() else 0.0,
        trades_live=int(sum(1 for t in trades if t["entry"] >= pd.Timestamp(TEST_START))),
        curves={k: [[str(d.date()), round(float(v), 5)] for d, v in s.items()] for k, s in curves.items()},
        t_cum=[[str(d.date()), round(float(v), 6)] for d, v in lay.cumsum().items()],
    )
    return res


def _p(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def to_markdown(res):
    b = res["best"]
    names = list(res["samples"])
    L = ["# 盘中做T回测：80% 资金按选股规则，20% 备用资金做T", "",
         f"数据截至 {res['updated']}。5 分钟K线模拟限价单成交，扣佣金、滑点（每边 {config.SLIPPAGE * 100:.2f}%）和卖出印花税。"
         f"规则 {res['n_rules']} 条，只在训练样本（随机股票 2021–2024）里挑，之后只检验。", "",
         "样本（股票×交易日）：" + "；".join(f"{n} {v}" for n, v in res["samples"].items()), "",
         f"**训练期挑出：{res['best_name']}**", "",
         "| 样本 | 成交率 | 每笔平均净收益 | 每笔胜率 | 每个持有日平均（按做T金额） | 滑点减半：每个持有日平均 | 净收益为正的规则数 |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for n in names:
        r, h = res["best_by_sample"][n], res["best_half_slip"][n]
        L.append(f"| {n} | {_p(r['fill_rate'], 1)} | {_p(r['per_trade'], 3)} | {_p(r['win'], 0)} | {_p(r['per_day'], 3)} | "
                 f"{_p(h['per_day'], 3)} | {res['positive_rules'][n]}/{res['n_rules']}（滑点减半 {res['positive_rules_half'][n]}） |")
    L += ["", "## 组合层面（线上规则真实持仓，2025 年至今）", "",
          "| 方案 | 累计收益 | 年化 | 最大回撤 | 夏普 | 最差一天 |", "| --- | --- | --- | --- | --- | --- |"]
    for k, v in res["combo"].items():
        L.append(f"| {k} | {_p(v['total'], 1)} | {_p(v['ann'], 1)} | {_p(v['max_dd'], 1)} | {v['sharpe']:.2f} | {_p(v['worst_day'], 2)} |")
    L += ["", f"做T共成交 {res['t_fills']} 笔，胜率 {_p(res['t_win'], 0)}，每笔平均 {_p(res['t_avg'], 3)}；"
          f"每只持仓做T累计降低成本 {_p(res['cost_cut_per_holding'], 2)}（占这只持仓金额）。",
          f"组合当天跌超 0.5% 的 {res['down_days']} 天里：选股部分平均 {_p(res['base_on_down_days'], 2)}，做T平均 {_p(res['t_on_down_days'], 3)}。", "",
          "## 训练期前 15 条规则", "",
          "| 规则 | 训练：每日平均 | 训练：成交率 | 训练：胜率 | 随机 2025+：每日平均 | 真实持仓 2025+：每日平均 | 真实持仓：胜率 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in res["top_train"]:
        L.append(f"| {rule_name(r['side'], r['ref'], r['k'], r['g'], _s(r['s']))} | {_p(r['per_day'], 3)} | {_p(r['fill_rate'], 0)} | "
                 f"{_p(r['win'], 0)} | {_p(r[names[1] + '|per_day'], 3)} | {_p(r[names[2] + '|per_day'], 3)} | {_p(r[names[2] + '|win'], 0)} |")
    L += ["", "仅供个人研究参考，不构成投资建议。"]
    return "\n".join(L) + "\n"


def write(res):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(to_markdown(res))
    PAGE_FILE.write_text(json.dumps(res, ensure_ascii=False, default=lambda o: None if o is None else float(o)))
