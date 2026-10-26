import numpy as np
import pandas as pd

from astock import themes
from astock.factors import Panel
from tests.test_astock import synthetic_kline


def _panel_with_hot_theme():
    k, basic = synthetic_kline(n_codes=60, start="2015-01-01", end="2015-12-31")
    k = k.copy()
    codes = sorted(k["code"].unique())
    hot = codes[:12]
    dates = sorted(k["date"].unique())
    # 最后 20 个交易日，热门题材每天多涨 3%，成交额翻倍
    for d in dates[-20:]:
        m = (k["date"] == d) & k["code"].isin(hot)
        k.loc[m, "close"] = (k.loc[m, "preclose"] * 1.03).round(2)
        k.loc[m, "amount"] *= 3
    # 收盘价要连贯：用每日涨跌重算
    k = k.sort_values(["code", "date"])
    r = (k["close"] / k["preclose"]).fillna(1)
    k["close"] = (k.groupby("code")["close"].transform("first") * r.groupby(k["code"]).cumprod()).astype("float32")
    k["preclose"] = k.groupby("code")["close"].shift(1).fillna(k["close"])
    members = pd.DataFrame({"concept": ["热门"] * 12 + ["冷门"] * 12 + ["融资融券"] * 30,
                            "code": hot + codes[12:24] + codes[:30]})
    names = basic.set_index("code")["code_name"]
    return Panel(k), themes.load_members(members), names


def test_hot_theme_ranks_first_and_filters_non_themes():
    p, members, names = _panel_with_hot_theme()
    assert "融资融券" not in members
    out = themes.build(p, members, names)
    assert out["themes"][0]["theme"] == "热门"
    assert out["themes"][0]["stage"] in ("加速", "启动")
    assert out["themes"][0]["excess20"] > 0.2
    assert {x["theme"] for x in out["chart"]} == {"热门", "冷门"}


def test_history_check_runs():
    p, members, names = _panel_with_hot_theme()
    t = p.dates[-30]
    h = themes.history_check(p, members, names, [t])
    assert set(h["theme"]) <= {"热门", "冷门"} and h["fwd"].notna().all()


def test_short_name_and_heat():
    assert themes.short_name("商业航天概念") == "商业航天"
    now = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M")
    heat = themes.news_heat([dict(time=now, title="商业航天板块走强", body="")], ["商业航天概念", "光伏"])
    assert heat["商业航天概念"] > 0 and heat["光伏"] == 0


def test_update_members_falls_back_and_keeps_old_file(tmp_path, monkeypatch):
    monkeypatch.setattr(themes, "CONCEPT_FILE", tmp_path / "concepts.csv")
    few = lambda: pd.DataFrame({"concept": ["甲"] * 3, "code": ["sh.600000"] * 3, "source": "x"})
    many = lambda: pd.DataFrame({"concept": [f"题材{i}" for i in range(60)], "code": "sz.000001", "source": "y"})
    assert themes.update_members([("a", few)]) is None
    assert not themes.CONCEPT_FILE.exists()
    out = themes.update_members([("a", few), ("b", many)])
    assert out["source"].iloc[0] == "y" and themes.CONCEPT_FILE.exists()
    assert themes._bs_code("688001") == "sh.688001" and themes._bs_code("sz300750") == "sz.300750"
    assert themes._bs_code("830001") is None
