import numpy as np
import pandas as pd

from astock import early_volume as ev


def _fake(n_codes=40, n_days=900, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-11-01", periods=n_days)
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
                              h1_close=hc, h1_vol=v * rng.uniform(0.2, 1.5), h1_amount=v * hc * 10 * rng.uniform(0.2, 1.5)))
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


def test_live_monitor(tmp_path, monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from astock import early_live

    tz = ZoneInfo("Asia/Shanghai")
    codes = [f"sz.00{i:04d}" for i in range(20)]
    assert early_live.update(tmp_path / "small", dict(signal_date="2026-11-09", codes={}), pd.DataFrame(),
                             datetime(2026, 11, 10, 10, 0, tzinfo=tz)) == []  # 参考数据太少就不记
    monkeypatch.setattr(early_live, "MIN_REF_CODES", 10)
    ref = dict(signal_date="2026-11-09", codes={c: [2.4e8, 0.1, f"股{i}"] for i, c in enumerate(codes)})
    model = dict(rule=dict(lbmin=2.0, lbmax=None, sl=0.05, tp=0.08), gap_range=[-1, 1], r1_range=[0, 1],
                 pos_range=[-0.01, 1.01], gate_range=[-0.01, 1.01], top_n=3, rule_text="test")

    def quotes(now, amt_scale, px_scale):
        rows = []
        for i, c in enumerate(codes):
            rows.append(dict(code=c, time=now.replace(tzinfo=None), close=10 * px_scale * (1 + i / 400), preclose=10.0,
                             volume=1.0, amount=1e6 * amt_scale * (i + 1), open=10.1, high=11.0, low=9.9))
        return pd.DataFrame(rows)

    t1 = datetime(2026, 11, 10, 9, 26, tzinfo=tz)
    assert early_live.update(tmp_path, ref, quotes(t1, 1, 1.01), t1, model) == []
    t2 = datetime(2026, 11, 10, 10, 31, tzinfo=tz)
    notes = early_live.update(tmp_path, ref, quotes(t2, 40, 1.02), t2, model)
    st = __import__("json").loads((tmp_path / "early.json").read_text())
    assert st["frozen"] and len(st["picks"]) == 3 and notes
    assert st["picks"][0]["code"] == codes[-1]
    assert st["auction"]
    assert (tmp_path / "early_log" / "2026-11-10.csv.gz").exists()
    assert early_live.issue_lines(st)
    t3 = datetime(2026, 11, 10, 14, 0, tzinfo=tz)
    early_live.update(tmp_path, ref, quotes(t3, 60, 0.9), t3, model)
    st = __import__("json").loads((tmp_path / "early.json").read_text())
    assert st["picks"][0]["status"] == "触及止损"

    # 没有规则文件：只观察，照样存 10:30 快照和竞价
    obs = tmp_path / "obs"
    obs.mkdir()
    early_live.update(obs, ref, quotes(t1, 1, 1.01), t1, None)
    assert early_live.update(obs, ref, quotes(t2, 40, 1.02), t2, None) == []
    st = __import__("json").loads((obs / "early.json").read_text())
    assert not st["picks"] and st["auction"] and st["top_volume"]
    assert (obs / "early_log" / "2026-11-10.csv.gz").exists()
    assert any("只观察" in x for x in early_live.issue_lines(st))


def test_write_ref(monkeypatch, tmp_path):
    from astock import early_live

    monkeypatch.setattr(early_live.config, "PAGE_DIR", tmp_path)
    monkeypatch.setattr(early_live, "REF_FILE", tmp_path / "early_ref.json")

    kl, _, basic = _fake(n_codes=5, n_days=60)
    basic["code_name"] = "x"
    ref = early_live.write_ref(kl, basic)
    assert len(ref["codes"]) == 5
