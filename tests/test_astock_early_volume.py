import numpy as np
import pandas as pd

from astock import early_volume as ev


def _fake(n_codes=40, n_days=900, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-11-01", periods=n_days)
    rows, hrows = [], []
    for i in range(n_codes):
        code = f"sz.00{i:04d}"
        px = 10.0
        for d in dates:
            pre = px
            o = pre * (1 + rng.normal(0, 0.01))
            c = pre * (1 + rng.normal(0, 0.02))
            h, l = max(o, c) * 1.01, min(o, c) * 0.99
            v = rng.uniform(1e6, 3e6)
            rows.append(dict(date=d, code=code, open=o, high=h, low=l, close=c, preclose=pre, volume=v,
                             amount=v * c * 10, turn=1.0, tradestatus=1, isST=0))
            hc = (o + c) / 2
            hrows.append(dict(date=d, code=code, h1_open=o, h1_high=max(o, hc) * 1.002, h1_low=min(o, hc) * 0.998,
                              h1_close=hc, h1_vol=v * rng.uniform(0.2, 1.5), h1_amount=v * hc))
            px = c
    kl = pd.DataFrame(rows)
    basic = pd.DataFrame(dict(code=kl["code"].unique(), ipoDate="2010-01-01"))
    return kl, pd.DataFrame(hrows), basic


def test_panel_and_run(monkeypatch):
    kl, hr, basic = _fake()
    p = ev.build_panel(kl, hr, basic)
    assert len(p) > 1000
    assert (p["lb"] > 0).all()
    monkeypatch.setattr(ev, "MIN_TRADES", 20)
    monkeypatch.setattr(ev, "MIN_DAY_SHARE", 0.01)
    monkeypatch.setattr(ev, "LB_MIN", (1.5, 2.0))
    res = ev.run(p)
    md = ev.to_markdown(res)
    assert "样本外" in md
    ev.page_data(res)


def test_tp_sl_order():
    p = pd.DataFrame(dict(h1_close=[10.0] * 4, entry=[10.0] * 4, sell_cost=[1.0] * 4,
                          n_open=[9.0, 10.0, 10.0, 11.5], n_high=[10.5, 11.0, 10.2, 12.0],
                          n_low=[8.8, 9.4, 9.8, 11.0], n_close=[9.5, 10.5, 10.1, 11.8]))
    r = ev.tp_sl_return(p, 0.05, 0.08).round(4).tolist()
    assert r == [-0.1, -0.05, 0.01, 0.15]
