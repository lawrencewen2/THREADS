import json

import numpy as np
import pandas as pd

from astock import insight
from astock.factors import Panel
from tests.test_astock import synthetic_kline


def _setup():
    k, basic = synthetic_kline(n_codes=40, start="2015-01-01", end="2016-06-30")
    val = k[["date", "code"]].copy()
    rng = np.random.default_rng(1)
    val["peTTM"] = rng.normal(20, 5, len(val)).astype("float32")
    val.loc[val["code"] == k["code"].iloc[0], "peTTM"] = -5.0
    val["pbMRQ"] = rng.normal(2, 0.3, len(val)).astype("float32")
    val["psTTM"] = 1.0
    codes = sorted(k["code"].unique())
    industry = pd.DataFrame({"code": codes, "industry": ["甲" if i % 2 else "乙" for i in range(len(codes))]})
    ranking = pd.DataFrame({"rank": range(1, len(codes) + 1), "code": codes, "name": "测试",
                            "score": np.linspace(1, 0, len(codes))})
    ranking["pct"] = 1 - (ranking["rank"] - 1) / len(codes)
    return k, basic, val, industry, ranking


def test_clean_industry():
    df = pd.DataFrame({"code": ["a", "b", "c"], "industry": ["C39计算机、通信和其他电子设备制造业", "J66货币金融服务", ""]})
    out = insight.clean_industry(df)
    assert list(out["industry"]) == ["计算机、通信和其他电子设备制造业", "货币金融服务"]


def test_temperature_counts_limits():
    k, basic, *_ = _setup()
    k = k.copy()
    last = k["date"].max()
    i = k.index[(k["date"] == last)][:3]
    k.loc[i, "close"] = (k.loc[i, "preclose"] * 1.1 + 1e-6).round(2)
    t = insight.market_temperature(Panel(k))
    row = t.iloc[-1]
    assert row["limit_up"] >= 3
    assert row["up"] + row["down"] <= k["code"].nunique()
    assert 0 <= row["temperature"] <= 100


def test_temperature_has_no_lookahead():
    k, *_ = _setup()
    cut = pd.Timestamp("2016-01-29")
    a = insight.market_temperature(Panel(k)).loc[:cut]
    k2 = k.copy()
    k2.loc[k2["date"] > cut, ["open", "high", "low", "close"]] *= 2
    b = insight.market_temperature(Panel(k2)).loc[:cut]
    pd.testing.assert_frame_equal(a, b)


def test_build_and_json(tmp_path, monkeypatch):
    k, basic, val, industry, ranking = _setup()
    insight.MIN_INDUSTRY_SIZE = 5
    picks = dict(fear_cn=18.0, top_n=3, picks=[dict(code=c) for c in ranking["code"][:3]])
    bt = dict(test_start="2015-06-01",
              chosen=dict(test=dict(avg_next_open_net=-0.002)),
              equity=[dict(date=str(d.date()), strategy=1 + 0.001 * i, baseline=1 - 0.0005 * i)
                      for i, d in enumerate(pd.bdate_range("2015-06-01", periods=60, freq="W-MON"))],
              contrarian=dict(chosen=dict(buy=20, sell=70), equity=dict(
                  strategy={str(d.date()): 1 + 0.002 * i for i, d in enumerate(pd.bdate_range("2015-06-01", periods=60, freq="W-MON"))},
                  always={str(d.date()): 1 + 0.001 * i for i, d in enumerate(pd.bdate_range("2015-06-01", periods=60, freq="W-MON"))})))
    out = insight.build(k, basic, val, industry, ranking, picks, bt)
    s = json.dumps(out, ensure_ascii=False, default=insight._clean)
    assert "NaN" not in s
    assert out["sectors"] and out["checkups"] and out["confidence"]
    first = next(r for r in out["checkups"] if r["code"] == k["code"].iloc[0])
    assert "亏损" in first["flags"]
    items = {i["item"]: i for i in out["checklist"]}
    assert items["市场情绪"]["status"] == "好"
    assert items["隔夜选股能否赚钱"]["status"] == "差"
    c = next(c for c in out["confidence"] if c["strategy"].startswith("恐慌买贪婪卖（选股）"))
    assert c["p_positive"] > 0.9 and c["p_beat_hold"] > 0.9


def test_bootstrap_interval_contains_actual():
    rng = np.random.default_rng(0)
    r = rng.normal(0.002, 0.02, 200)
    b = insight.bootstrap(r)
    assert b["ci_low"] < b["ann"] < b["ci_high"]
    assert 0 <= b["p_positive"] <= 1


def test_news_helpers():
    from astock import news
    assert news._title_and_body("【央行降准】央行宣布降准0.5个百分点") == ("央行降准", "央行宣布降准0.5个百分点")
    t, b = news._title_and_body("没有标题的快讯")
    assert t == "没有标题的快讯" and b == t
    assert news.short_industry("计算机、通信和其他电子设备制造业") == "计算机"
    assert news.short_industry("汽车制造业") == "汽车"
    rows = news.tag_market([dict(title="众兴菌业公告", body="汽车板块走强")], ["众兴菌业", "鑫磊股份"], ["汽车"])
    assert rows[0]["related"] == "众兴菌业、汽车"


def test_news_build_survives_failures(monkeypatch):
    from astock import news

    def boom(*a, **k):
        raise OSError("blocked")
    monkeypatch.setattr(news, "_get", boom)
    monkeypatch.setattr(news.time, "sleep", lambda s: None)
    out = news.build(dict(picks=[dict(code="sz.002772", name="众兴菌业")]), [])
    assert out["market"] == [] and out["stocks"] == []
    assert all(not s["ok"] for s in out["status"])


def test_week_events_merge_and_rank():
    from astock import news
    rows = [dict(time="2026-10-20 09:00", title="央行宣布降准0.5个百分点", body="释放长期资金约1万亿元", tags=["焦点"]),
            dict(time="2026-10-20 09:05", title="央行宣布降准0.5个百分点 释放资金1万亿", body="", tags=[]),
            dict(time="2026-10-21 10:00", title="某公司召开股东大会", body="审议议案", tags=[]),
            dict(time="2026-10-22 15:00", title="美国宣布对华加征关税", body="涉及汽车", tags=[]),
            dict(time="2026-10-23 15:00", title="9月CPI同比上涨0.3%", body="", tags=[])]
    ev = news.week_events(rows, ["众兴菌业"], ["汽车"])
    assert [e["title"] for e in ev] == ["央行宣布降准0.5个百分点", "美国宣布对华加征关税", "9月CPI同比上涨0.3%"]
    assert ev[0]["reports"] == 2 and ev[0]["focus"] and ev[0]["category"] == "货币政策"
    assert ev[1]["related"] == "汽车"


def test_week_events_merge_split_document():
    from astock import news
    rows = [dict(time="2026-10-23 19:13", title="李强主持召开国务院常务会议", body="", tags=["焦点"]),
            dict(time="2026-10-23 19:15", title="李强主持召开国务院常务会议，会议指出，要进一步发挥服务业对经济发展的支撑作用", body="", tags=[]),
            dict(time="2026-10-21 18:04", title="中办、国办：全面实施“人工智能+”行动", body="", tags=[]),
            dict(time="2026-10-21 18:04", title="中办、国办：推动集成电路等关键核心技术攻关", body="", tags=[]),
            dict(time="2026-10-25 04:00", title="美国务院更新中东多国旅行警告", body="", tags=[]),
            dict(time="2026-10-26 09:54", title="ETF两市成交额超5000亿元", body="", tags=["焦点"])]
    ev = news.week_events(rows)
    assert [e["reports"] for e in ev] == [2, 2]
    assert ev[0]["category"].startswith("重大会议")


def test_event_targets():
    from astock import news
    t = news.event_targets([dict(stocks=[])], "也门胡塞武装称，他们用无人机袭击了沙特阿拉伯的东西向原油泵站")
    assert [x["why"] for x in t][:2] == ["原油", "原油"] and len(t) == 3 and t[2]["why"] != "原油"
    t = news.event_targets([dict(stocks=[("sz.300750", "宁德时代")]), dict(stocks=[("sz.300750", "宁德时代")])], "电池")
    assert t[0] == dict(code="sz.300750", name="宁德时代", why="快讯点名")
    assert news.event_targets([dict(stocks=[])], "党的二十届五中全会召开")[0]["why"] == "大盘"
