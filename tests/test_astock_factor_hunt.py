import numpy as np
import pandas as pd

from astock import factor_hunt
from astock.factors import Panel
from test_astock import synthetic_kline


def _fc(rows):
    return pd.DataFrame(rows, columns=["code", "pub", "stat", "type", "up", "dn"]).assign(
        pub=lambda d: pd.to_datetime(d["pub"]), stat=lambda d: pd.to_datetime(d["stat"]))


def test_forecast_extras_revision_and_no_lookahead():
    k, basic = synthetic_kline(n_codes=6, start="2015-01-01", end="2015-12-31", seed=1)
    p = Panel(k)
    c = p.codes[0]
    fc = _fc([(c, "2015-03-02", "2014-12-31", "预增", 60.0, 40.0),
              (c, "2015-04-01", "2015-03-31", "预增", 30.0, 10.0),
              (c, "2015-04-20", "2015-03-31", "预增", 50.0, 30.0)])
    ex = factor_hunt.forecast_extras(p, fc)
    d1 = p.dates[p.dates.searchsorted(pd.Timestamp("2015-04-20"), side="right")]
    d0 = p.dates[p.dates.searchsorted(pd.Timestamp("2015-04-20"), side="right") - 1]
    assert ex["fc_rev"].loc[d1, c] == 20.0           # 40 → 20 upgrade, same period
    assert np.isnan(ex["fc_rev"].loc[d0, c])          # not known before the next trading day
    a = ex["fc_accel"].loc[d1, c]
    assert a == 40.0 - 50.0                           # this period mid 40 vs previous period mid 50
    assert ex["fc_narrow"].loc[d1, c] == 0.5          # width 20 / |mid| 40


def test_industry_breadth():
    k, basic = synthetic_kline(n_codes=10, start="2015-01-01", end="2015-06-30", seed=2)
    p = Panel(k)
    codes = list(p.codes)
    fc = _fc([(codes[0], "2015-03-02", "2014-12-31", "预增", 60.0, 40.0)])
    ex = factor_hunt.forecast_extras(p, fc)
    ind = {c: ("A" if i < 5 else "B") for i, c in enumerate(codes)}
    out = factor_hunt.industry_factors(p, ex, ind)
    d = p.dates[p.dates.searchsorted(pd.Timestamp("2015-03-10"))]
    traded = p.traded.loc[d]
    a = [c for c in codes[:5] if traded[c]]
    if codes[0] in a:
        assert abs(out["ind_breadth"].loc[d, a[0]] - 1 / 5) < 1e-6
    assert all(np.isnan(out["ind_breadth"].loc[d, c]) or out["ind_breadth"].loc[d, c] == 0 for c in codes[5:])


def test_live_blend_matches_backtest_combo():
    from astock import live_slots, model
    from astock.factors import eligible

    k, basic = synthetic_kline(n_codes=12, start="2015-01-01", end="2015-12-31", seed=3)
    p = Panel(k)
    codes = list(p.codes)
    fc = _fc([(c, "2015-03-02", "2014-12-31", "预增", 40.0 + i, 20.0) for i, c in enumerate(codes)]
             + [(c, "2015-04-10", "2015-03-31", "预增", 80.0 - 5 * i, 50.0) for i, c in enumerate(codes)])
    elig = eligible(p, basic)
    base = pd.DataFrame(np.random.default_rng(0).random(elig.shape), index=p.dates, columns=p.codes).where(elig)
    live = live_slots.apply_blend(base, p, fc, elig, {"fc_accel": 0.3})
    cand = {"fc_accel": model.xs_rank(factor_hunt.forecast_extras(p, fc)["fc_accel"], elig)}
    bt = factor_hunt.combo_score(base, cand, factor_hunt.COMBOS["accel0.3"])
    pd.testing.assert_frame_equal(live, bt)
    assert live_slots.apply_blend(base, p, fc, elig, None) is base
    d = p.dates[p.dates.searchsorted(pd.Timestamp("2015-04-20"))]
    assert (live.loc[d] != base.loc[d]).any()
