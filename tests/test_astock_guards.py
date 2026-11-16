import json

import pandas as pd

from astock import guards


def kl(dates_codes):
    return pd.DataFrame([dict(date=pd.Timestamp(d), code=c) for d, c in dates_codes])


def res():
    return dict(signal_date="2026-11-13", fear_date="2026-11-13", gate_open=True, n_selected=2,
                picks=[dict(code="a", action="继续持有", selected=True), dict(code="b", action="买入", selected=True),
                       dict(code="c", action="候补", selected=False)])


def test_fresh_data_passes(tmp_path):
    k = kl([("2026-11-12", "a"), ("2026-11-13", "a"), ("2026-11-13", "b")])
    fc = pd.DataFrame(dict(pub=["2026-11-10"]))
    r = guards.apply(res(), k, fc, nav_rows=[], state_file=tmp_path / "s.json")
    assert r["gate_open"] and r["n_selected"] == 2 and not r["stale"]


def test_stale_blocks_new_buys_only(tmp_path):
    k = kl([("2026-11-12", c) for c in "abcdefghij"] + [("2026-11-13", "a")])  # 覆盖率 10%
    r = guards.apply(res(), k, None, nav_rows=[], state_file=tmp_path / "s.json")
    assert not r["gate_open"] and r["stale"]
    acts = [p["action"] for p in r["picks"]]
    assert acts == ["继续持有", "观察", "观察"] and r["n_selected"] == 1


def test_old_forecast_and_fear_flagged():
    k = kl([("2026-11-13", "a")])
    fc = pd.DataFrame(dict(pub=["2026-08-01"]))
    reasons = guards.staleness(k, fc, "2026-11-13", "2026-11-05")
    assert len(reasons) == 2


def test_redline_latches(tmp_path):
    f = tmp_path / "s.json"
    rows = [dict(date="d1", nav="1.0"), dict(date="d2", nav="0.84")]
    st = guards.redline(rows, f)
    assert st["halted"] and st["since"] == "d2"
    # 净值回升也不会自动恢复
    st = guards.redline(rows + [dict(date="d3", nav="1.1")], f)
    assert st["halted"]
    f.write_text(json.dumps(dict(halted=False)))
    assert not guards.redline(rows[:1], f)["halted"]


def test_frozen(tmp_path):
    pf = tmp_path / "picks.json"
    pf.write_text(json.dumps(dict(trade_date="2026-11-17")))
    assert guards.frozen("2026-11-17", pd.Timestamp("2026-11-17 11:55"), pf)
    assert not guards.frozen("2026-11-17", pd.Timestamp("2026-11-17 08:50"), pf)
    assert not guards.frozen("2026-11-18", pd.Timestamp("2026-11-18 11:55"), pf)
