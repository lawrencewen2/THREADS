from datetime import datetime

import pandas as pd

from astock import insight, live_market
from astock.factors import Panel
from tests.test_astock_insight import _setup


def _quotes_for(k, day):
    rows = k[k["date"] == day]
    return pd.DataFrame({
        "code": rows["code"].values,
        "time": pd.Timestamp(day) + pd.Timedelta(hours=15),
        "close": rows["close"].values,
        "preclose": rows["preclose"].values,
        "volume": rows["volume"].values,
        "amount": rows["amount"].values,
    })


def test_live_matches_daily_temperature():
    """收盘后用当天报价算的盘中温度，应和用日线算的正式温度一致。"""
    k, *_ = _setup()
    days = sorted(k["date"].unique())
    today, prev = days[-1], days[-2]
    full = insight.market_temperature(Panel(k))
    p_prev = Panel(k[k["date"] <= prev])
    ref = insight.live_ref(p_prev, insight.market_temperature(p_prev))
    assert ref["signal_date"] == str(pd.Timestamp(prev).date())
    now = datetime.combine(pd.Timestamp(today).date(), datetime.min.time()).replace(hour=15, minute=5,
                                                                                    tzinfo=live_market.CN_TZ)
    res = live_market.build(ref, _quotes_for(k, today), now)
    row = full.iloc[-1]
    assert res["session"] == "已收盘"
    assert res["up"] == row["up"] and res["down"] == row["down"]
    assert res["limit_up"] == row["limit_up"] and res["limit_down"] == row["limit_down"]
    assert abs(res["above_ma20"] - row["above_ma20"]) < 0.03
    assert abs(res["temperature"] - row["temperature"]) < 3


def test_amount_projection_and_session():
    tz = live_market.CN_TZ
    assert live_market.minutes_traded(datetime(2026, 11, 9, 10, 30, tzinfo=tz)) == 60
    assert live_market.minutes_traded(datetime(2026, 11, 9, 12, 0, tzinfo=tz)) == 120
    assert live_market.minutes_traded(datetime(2026, 11, 9, 14, 0, tzinfo=tz)) == 180
    assert live_market.minutes_traded(datetime(2026, 11, 9, 16, 0, tzinfo=tz)) == 240


def test_stale_quotes_ignored():
    k, *_ = _setup()
    days = sorted(k["date"].unique())
    p_prev = Panel(k[k["date"] <= days[-2]])
    ref = insight.live_ref(p_prev, insight.market_temperature(p_prev))
    q = _quotes_for(k, days[-2])  # 昨天的报价（停牌或还没开盘）不算
    now = datetime.combine(pd.Timestamp(days[-1]).date(), datetime.min.time()).replace(hour=10, tzinfo=live_market.CN_TZ)
    assert live_market.build(ref, q, now) is None
