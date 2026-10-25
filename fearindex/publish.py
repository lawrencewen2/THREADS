"""输出：CSV/JSON 数据、SVG 走势图、README 中的今日读数、网页仪表盘。"""

import json
import re
from pathlib import Path

import pandas as pd

from . import config

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"

COLORS = {"global": "#111827", "us": "#2563eb", "eu": "#16a34a", "jp": "#dc2626", "cn": "#d97706", "hk": "#7c3aed"}
NAMES = {"global": "全球", **config.REGIONS}


def write_data(index_df, comp_df, raw_df, status):
    DATA.mkdir(exist_ok=True)
    index_df.rename_axis("date").to_csv(DATA / "index.csv", float_format="%.1f")
    comp_df.rename_axis("date").to_csv(DATA / "components.csv", float_format="%.1f")
    raw_df.rename_axis("date").to_csv(DATA / "components_raw.csv", float_format="%.6g")
    latest = latest_snapshot(index_df, comp_df, status)
    (DATA / "latest.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2))
    return latest


def latest_snapshot(index_df, comp_df, status):
    last = index_df.index[-1]
    row = index_df.loc[last]
    week = index_df.loc[:last].iloc[-6] if len(index_df) > 5 else row
    regions = {}
    for key in ["global"] + list(config.REGIONS):
        v = row.get(key)
        regions[key] = dict(
            name=NAMES[key],
            score=None if pd.isna(v) else float(v),
            band=None if pd.isna(v) else config.band(v),
            change_5d=None if pd.isna(v) or pd.isna(week.get(key)) else round(float(v - week[key]), 1),
            risk_off=bool(config.risk_off(index_df[key]).iloc[-1]),
        )
    comps = {}
    for cid, spec in config.COMPONENTS.items():
        v = comp_df[cid].iloc[-1] if cid in comp_df else float("nan")
        comps[cid] = dict(name=spec["name"], region=spec["region"], score=None if pd.isna(v) else float(v))
    return dict(date=str(last.date()), method=config.fingerprint(), regions=regions, components=comps, status=status)


def svg_chart(index_df, days=520, width=900, height=360):
    df = index_df[["global"] + list(config.REGIONS)].iloc[-days:]
    pad_l, pad_r, pad_t, pad_b = 40, 90, 16, 28
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(df)

    def x(i):
        return pad_l + w * i / max(n - 1, 1)

    def y(v):
        return pad_t + h * (1 - v / 100)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="11">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    for lo, hi, fill in ((0, 20, "#fee2e2"), (80, 100, "#dcfce7")):
        parts.append(f'<rect x="{pad_l}" y="{y(hi):.1f}" width="{w}" height="{y(lo) - y(hi):.1f}" fill="{fill}"/>')
    for v in (0, 20, 40, 60, 80, 100):
        parts.append(f'<line x1="{pad_l}" x2="{pad_l + w}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y(v) + 4:.1f}" text-anchor="end" fill="#6b7280">{v}</text>')
    # 每季度一个日期刻度
    for i, d in enumerate(df.index):
        if d.month in (1, 4, 7, 10) and (i == 0 or df.index[i - 1].month != d.month):
            parts.append(f'<text x="{x(i):.1f}" y="{height - 8}" text-anchor="middle" fill="#6b7280">{d:%Y-%m}</text>')
    for key in reversed(list(df.columns)):
        s = df[key]
        pts, segs = [], []
        for i, v in enumerate(s):
            if pd.isna(v):
                if pts:
                    segs.append(pts)
                pts = []
            else:
                pts.append(f"{x(i):.1f},{y(v):.1f}")
        if pts:
            segs.append(pts)
        sw, op = (2.6, 1) if key == "global" else (1.0, 0.45)
        for seg in segs:
            parts.append(f'<polyline fill="none" stroke="{COLORS[key]}" stroke-width="{sw}" stroke-opacity="{op}" '
                         f'points="{" ".join(seg)}"/>')
    # 右侧标签按数值排序并错开，避免重叠
    labels = sorted(((y(s.dropna().iloc[-1]), k, s.dropna().iloc[-1]) for k, s in df.items() if s.notna().any()))
    prev = -1e9
    for ly, key, v in labels:
        ly = max(ly, prev + 13)
        prev = ly
        weight = "bold" if key == "global" else "normal"
        parts.append(f'<text x="{pad_l + w + 6}" y="{ly + 4:.1f}" fill="{COLORS[key]}" font-weight="{weight}">'
                     f'{NAMES[key]} {v:.0f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def readme_block(latest):
    lines = [f"**最新读数（{latest['date']} 收盘，0 = 极度恐慌，100 = 极度贪婪）**", "",
             "| 区域 | 恐慌贪婪指数 | 状态 | 5日变化 | 仓位信号 |", "| --- | --- | --- | --- | --- |"]
    for r in latest["regions"].values():
        score = "—" if r["score"] is None else f"{r['score']:.0f}"
        chg = "—" if r["change_5d"] is None else f"{r['change_5d']:+.1f}"
        signal = "恐慌加深，减仓" if r.get("risk_off") else "正常"
        lines.append(f"| {r['name']} | {score} | {r['band'] or '—'} | {chg} | {signal} |")
    st = latest["status"]
    if st.get("warnings"):
        lines += ["", "⚠️ " + "；".join(st["warnings"])]
    lines += ["", "![近两年走势](docs/chart.svg)"]
    return "\n".join(lines)


def _fmt(v):
    return "—" if pd.isna(v) else f"{v:.0f}"


def region_tables(index_df, comp_df, latest):
    """每个区域一张表：区域分与各成分在今天、5日前、20日前的得分，下附近21个交易日折线图。"""
    lines = ["## 分区域明细", ""]
    for key, name in config.REGIONS.items():
        r = latest["regions"][key]
        score = index_df[key]
        head = f"### {name}：{_fmt(r['score'])}（{r['band'] or '数据缺失'}）"
        chg = "—" if r["change_5d"] is None else f"{r['change_5d']:+.1f}"
        signal = "恐慌加深，减仓" if r.get("risk_off") else "正常"
        lines += [head, "", f"5日变化 {chg} · 仓位信号：{signal}", "",
                  "| 成分 | 今日 | 5日前 | 20日前 | 权重 |", "| --- | --- | --- | --- | --- |",
                  f"| **{name}恐慌贪婪指数** | **{_fmt(score.iloc[-1])}** | {_fmt(score.iloc[-6])} | "
                  f"{_fmt(score.iloc[-21])} | — |"]
        for cid, spec in config.COMPONENTS.items():
            if spec["region"] != key:
                continue
            if cid in comp_df:
                c = comp_df[cid]
                lines.append(f"| {spec['name']} | {_fmt(c.iloc[-1])} | {_fmt(c.iloc[-6])} | {_fmt(c.iloc[-21])} | "
                             f"{spec['weight']:g} |")
            else:
                lines.append(f"| {spec['name']} | 缺失 | — | — | {spec['weight']:g} |")
        lines += ["", f"![{name}近21个交易日](docs/regions/{key}.svg)", ""]
    return "\n".join(lines).rstrip()


def svg_recent(series, color, days=21, width=640, height=220):
    """单个区域近 days 个交易日的折线图：0–20 极度恐慌区标红，80–100 极度贪婪区标绿，每个点标数值。"""
    s = series.dropna().iloc[-days:]
    pad_l, pad_r, pad_t, pad_b = 34, 16, 14, 26
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(s)

    def x(i):
        return pad_l + w * i / max(n - 1, 1)

    def y(v):
        return pad_t + h * (1 - v / 100)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="10">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
             f'<rect x="{pad_l}" y="{y(20):.1f}" width="{w}" height="{y(0) - y(20):.1f}" fill="#fee2e2"/>',
             f'<rect x="{pad_l}" y="{y(100):.1f}" width="{w}" height="{y(80) - y(100):.1f}" fill="#dcfce7"/>']
    for v, label in ((0, "0"), (20, "20"), (40, "40"), (60, "60"), (80, "80"), (100, "100")):
        parts.append(f'<line x1="{pad_l}" x2="{pad_l + w}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{pad_l - 5}" y="{y(v) + 3:.1f}" text-anchor="end" fill="#6b7280">{label}</text>')
    parts.append(f'<text x="{pad_l + 4}" y="{y(2):.1f}" fill="#b91c1c">恐慌</text>')
    parts.append(f'<text x="{pad_l + 4}" y="{y(92):.1f}" fill="#15803d">贪婪</text>')
    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(s))
    parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
    for i, (d, v) in enumerate(s.items()):
        parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="2.5" fill="{color}"/>')
        parts.append(f'<text x="{x(i):.1f}" y="{y(v) - 6:.1f}" text-anchor="middle" fill="#374151">{v:.0f}</text>')
        if i % 5 == 0 or i == n - 1:
            parts.append(f'<text x="{x(i):.1f}" y="{height - 8}" text-anchor="middle" fill="#6b7280">{d:%m/%d}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def update_readme(latest, index_df=None, comp_df=None):
    path = ROOT / "README.md"
    text = path.read_text()
    body = readme_block(latest)
    if index_df is not None and comp_df is not None:
        body += "\n\n" + region_tables(index_df, comp_df, latest)
    block = f"<!-- latest:start -->\n{body}\n<!-- latest:end -->"
    text = re.sub(r"<!-- latest:start -->.*<!-- latest:end -->", lambda _: block, text, flags=re.S)
    path.write_text(text)


def write_dashboard(index_df, latest):
    DOCS.mkdir(exist_ok=True)
    (DOCS / "chart.svg").write_text(svg_chart(index_df))
    (DOCS / "regions").mkdir(exist_ok=True)
    for key in config.REGIONS:
        (DOCS / "regions" / f"{key}.svg").write_text(svg_recent(index_df[key], COLORS[key]))
    hist = index_df[["global"] + list(config.REGIONS)].iloc[-1500:]
    payload = dict(
        dates=[str(d.date()) for d in hist.index],
        series={k: [None if pd.isna(v) else float(v) for v in hist[k]] for k in hist.columns},
        names=NAMES, colors=COLORS, latest=latest,
    )
    template = (ROOT / "fearindex" / "dashboard.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False))
    (DOCS / "index.html").write_text(html)
