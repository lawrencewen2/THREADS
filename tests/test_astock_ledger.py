import pandas as pd

from astock import ledger

SM = dict(weights={"a": 1}, n=20, capital_share=0.65, sleeve={"codes": ["x", "y"]})


def result(td, picks):
    return dict(trade_date=td, invest=True, gate_open=True, n_selected=len(picks), picks=picks, sells=[])


def test_append_only_and_dedupe(tmp_path):
    p = [dict(code="sh.600000", name="A", action="买入", weight=0.0325, close=10.0,
              stop_price_ref=9.0, take_profit_price_ref=11.0)]
    v = ledger.rule_version(SM)
    assert ledger.log_signals(result("2026-11-17", p), v, tmp_path) == 1
    # 同一天重跑不重复，也不改第一次的
    p2 = [dict(p[0], close=99.0)]
    assert ledger.log_signals(result("2026-11-17", p2), v, tmp_path) == 0
    rows = ledger.read("signals", tmp_path)
    assert len(rows) == 1 and rows[0]["ref_close"] == "10.0" and rows[0]["stop_price"] == "9.0"


def test_nav_freezes_and_waits_for_etf(tmp_path):
    v = ledger.rule_version(SM)
    ledger.log_plan(result("2026-11-17", []), dict(stock_used=0.3, sleeve_weight=0.7), v, tmp_path)
    ledger.log_plan(result("2026-11-18", []), dict(stock_used=0.3, sleeve_weight=0.7), v, tmp_path)
    live = dict(summary=dict(daily=[dict(trade_date="2026-11-17", net=0.01), dict(trade_date="2026-11-18", net=-0.02)]))
    idx = pd.to_datetime(["2026-11-16", "2026-11-17"])
    closes = {"x": pd.Series([1.0, 1.1], index=idx), "y": pd.Series([1.0, 0.9], index=idx)}
    # 11-18 的 ETF 收益还没有：只记 11-17
    assert ledger.log_nav(live, closes, v, tmp_path) == 1
    r = ledger.read("nav", tmp_path)[0]
    assert abs(float(r["total_net"]) - 0.01) < 1e-9  # 两只 ETF +10% / −10% 等权为 0
    # live.json 的历史被改了也不影响已记的数
    live["summary"]["daily"][0]["net"] = 0.5
    idx2 = pd.to_datetime(["2026-11-16", "2026-11-17", "2026-11-18"])
    closes2 = {"x": pd.Series([1.0, 1.1, 1.21], index=idx2), "y": pd.Series([1.0, 0.9, 0.99], index=idx2)}
    assert ledger.log_nav(live, closes2, v, tmp_path) == 1
    rows = ledger.read("nav", tmp_path)
    assert rows[0]["total_net"] == r["total_net"]
    assert abs(float(rows[1]["total_net"]) - (-0.02 + 0.7 * 0.1)) < 1e-9
    assert abs(float(rows[1]["nav"]) - 1.01 * (1 - 0.02 + 0.07)) < 1e-6


def test_trades_once(tmp_path):
    live = dict(rows=[dict(code="a", name="A", entry_date="2026-11-03", exit_date="2026-11-05", net=0.05)])
    assert ledger.log_trades(live, tmp_path) == 1
    assert ledger.log_trades(live, tmp_path) == 0


def test_rule_version_changes_with_rules():
    assert ledger.rule_version(SM) != ledger.rule_version(dict(SM, capital_share=0.5))


def _plan(tmp_path, days, w=0.7):
    v = ledger.rule_version(SM)
    for td in days:
        ledger.log_plan(result(td, []), dict(stock_used=0.3, sleeve_weight=w), v, tmp_path)
    return v


def test_missing_etf_does_not_average_the_rest(tmp_path):
    v = _plan(tmp_path, ["2026-11-17"])
    live = dict(summary=dict(daily=[dict(trade_date="2026-11-17", net=0.0)]))
    idx = pd.to_datetime(["2026-11-16", "2026-11-17"])
    closes = {"x": pd.Series([1.0, 1.1], index=idx)}  # y 没拉到
    assert ledger.log_nav(live, closes, v, tmp_path, codes=["x", "y"]) == 0


def test_one_etf_gap_does_not_freeze_nav(tmp_path):
    v = _plan(tmp_path, ["2026-11-17", "2026-11-18"])
    live = dict(summary=dict(daily=[dict(trade_date="2026-11-17", net=0.0), dict(trade_date="2026-11-18", net=0.0)]))
    closes = {"x": pd.Series([1.0, 1.1, 1.1], index=pd.to_datetime(["2026-11-16", "2026-11-17", "2026-11-18"])),
              "y": pd.Series([1.0, 1.0], index=pd.to_datetime(["2026-11-16", "2026-11-18"]))}  # y 停牌一天
    assert ledger.log_nav(live, closes, v, tmp_path, codes=["x", "y"]) == 2


def test_failed_sleeve_not_logged_as_zero(tmp_path):
    v = ledger.rule_version(SM)
    assert ledger.log_plan(result("2026-11-17", []), None, v, tmp_path, need_sleeve=True) == 0
    _plan(tmp_path, ["2026-11-16"], w=0.5)
    live = dict(summary=dict(daily=[dict(trade_date="2026-11-17", net=0.0)]))
    idx = pd.to_datetime(["2026-11-16", "2026-11-17"])
    closes = {"x": pd.Series([1.0, 1.1], index=idx), "y": pd.Series([1.0, 1.1], index=idx)}
    ledger.log_nav(live, closes, v, tmp_path, ts="2026-11-18T01:00:00Z", codes=["x", "y"])
    r = ledger.read("nav", tmp_path)[-1]
    assert float(r["sleeve_weight"]) == 0.5 and "沿用" in r["source"]


def test_note_change_keeps_version():
    a = dict(SM, sleeve={"codes": ["x", "y"], "note": "旧"})
    b = dict(SM, sleeve={"codes": ["x", "y"], "note": "新"})
    assert ledger.rule_version(a) == ledger.rule_version(b)
