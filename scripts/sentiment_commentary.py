"""每日市场情绪解读：读每日更新已经产出的数据，调用 Claude 写一段中文解读。

不重新抓行情，只用仓库里已有的文件：
  docs/page/latest.json、docs/page/history.json   恐慌贪婪指数读数和历史
  data/components.csv                              各分项得分（0–100，越高越恐慌，发布时会反过来）
  data/raw/*.csv                                   原始行情（算涨跌幅）
  docs/astock/insight.json、docs/astock/picks.json A股市场温度、行业强弱、选股开关

输出：
  reports/sentiment/YYYY-MM-DD.md   当天解读（YYYY-MM-DD = 数据日期）
  docs/page/commentary.json         最新一期（给页面/早读用）
  README.md                         <!-- commentary:start --> ... <!-- commentary:end --> 之间

用法：
  python scripts/sentiment_commentary.py            # 已有当天解读就跳过（每日任务会跑两次）
  python scripts/sentiment_commentary.py --force    # 强制重写
  python scripts/sentiment_commentary.py --dry-run  # 只打印发给 Claude 的数据，不调用 API

需要环境变量 ANTHROPIC_API_KEY（在 GitHub 仓库 Secrets 里设置，绝不写进代码）。没有就跳过，不报错。
模型默认 claude-opus-5-5，可用环境变量 COMMENTARY_MODEL 改。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fearindex import config  # noqa: E402

MODEL = os.environ.get("COMMENTARY_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("COMMENTARY_EFFORT", "medium")
OUT_DIR = ROOT / "reports" / "sentiment"
REGION_KEYS = ["global", "us", "eu", "jp", "cn", "hk"]
REGION_NAMES = {"global": "全球", "us": "美国", "eu": "欧洲", "jp": "日本", "cn": "A股", "hk": "港股"}

# 原始行情：文件名 → (中文名, 怎么显示变化)  pct = 涨跌幅，diff = 差值（利率、波动率指数）
MARKETS = {
    "spx": ("标普500", "pct"),
    "vix": ("VIX", "diff"),
    "us10y": ("美债10年收益率", "diff"),
    "move": ("MOVE 美债波动率", "diff"),
    "sx5e": ("欧洲斯托克50", "pct"),
    "eurchf": ("欧元/瑞郎", "pct"),
    "n225": ("日经225", "pct"),
    "topix": ("TOPIX", "pct"),
    "usdjpy": ("美元/日元", "pct"),
    "csi300": ("沪深300", "pct"),
    "csi1000": ("中证1000", "pct"),
    "qvix50": ("50ETF 期权波动率", "diff"),
    "a_turnover": ("两市成交额（元）", "pct"),
    "sh_margin": ("沪市融资余额", "pct"),
    "hsi": ("恒生指数", "pct"),
    "hstech": ("恒生科技（3033.HK）", "pct"),
    "gold": ("黄金", "pct"),
}

SYSTEM_PROMPT = """你是一位谨慎、讲证据的跨市场策略分析师，为一位个人投资者写每日“市场情绪解读”。
读者自己维护一个 0–100 的恐慌贪婪指数（0 = 极度恐慌，100 = 极度贪婪），覆盖美国、欧洲、日本、A股、港股，并用它来控制仓位。

写作要求：
- 全文用简体中文，Markdown 格式，600–900 字。
- 只用用户消息里给出的数字，不要编造任何数据、新闻或事件；数据里没有的就不说。不要猜测涨跌原因背后的新闻。
- 引用数字时保留一位小数以内，说清是哪一天的。
- 不要写“37→39→41”这种箭头串来描述走势；趋势用文字概括（例如“过去 10 个交易日从 30 出头回升到 40 附近”）。
- 分项得分在数据里是“越高越恐慌”（0–100 历史分位），解读时换成读者能懂的说法，例如“VIX 处在过去五年偏高的位置”。
- 读者已有的规则：区域读数 ≤20 视为恐慌买入区、≥70 视为贪婪卖出区；“恐慌加深，减仓”是指数自带的风险信号。可以说明当前离这些线有多远，但不要给出具体个股买卖建议，也不要替读者改规则。
- 语气平实，不渲染情绪，不用“暴涨”“崩盘”这类词，除非数字确实如此。

结构（用这些二级标题）：
## 一句话结论
## 各区域情绪
（每个区域 1–3 句：读数、所处区间、近 5/20 日变化、主要是哪些分项在推动）
## 跨市场联动
（区域之间是同步还是分化，避险资产——美债、黄金、日元、瑞郎——在说什么）
## A股观察
（结合市场温度、涨跌家数、成交额、融资余额、选股开关状态）
## 需要留意的信号
（2–4 条要点，写清楚触发条件，例如“若 A股读数跌破 20 则进入恐慌买入区”）

最后一行固定写：*以上由 Claude 根据当日数据自动生成，仅供参考，不构成投资建议。*"""


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _series(name: str) -> pd.Series | None:
    path = ROOT / "data" / "raw" / f"{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).dropna()
    return df.set_index("date")["value"].sort_index()


def _change(s: pd.Series, n: int, kind: str) -> float | None:
    if len(s) <= n:
        return None
    now, then = float(s.iloc[-1]), float(s.iloc[-1 - n])
    if kind == "pct":
        return round((now / then - 1) * 100, 2) if then else None
    return round(now - then, 2)


def market_table(asof: pd.Timestamp) -> list[dict]:
    rows = []
    for key, (name, kind) in MARKETS.items():
        s = _series(key)
        if s is None or s.empty:
            continue
        s = s[s.index <= asof + pd.Timedelta(days=1)]  # 亚洲市场日期可能比美股晚一天
        if s.empty:
            continue
        unit = "%" if kind == "pct" else "（差值）"
        rows.append(dict(
            name=name, date=str(s.index[-1].date()), last=round(float(s.iloc[-1]), 4),
            **{f"chg_1d{unit}": _change(s, 1, kind), f"chg_5d{unit}": _change(s, 5, kind),
               f"chg_20d{unit}": _change(s, 20, kind)},
        ))
    return rows


def component_table(asof: pd.Timestamp) -> dict:
    path = ROOT / "data" / "components.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df[df.index <= asof]
    if df.empty:
        return {}
    last, prev5 = df.iloc[-1], df.iloc[-6] if len(df) > 5 else None
    out: dict[str, list] = {}
    for col, spec in config.COMPONENTS.items():
        if col not in df.columns or pd.isna(last[col]):
            continue
        region = REGION_NAMES.get(spec["region"], spec["region"])
        item = dict(name=spec["name"], fear_pct=round(float(last[col]), 1))
        if prev5 is not None and not pd.isna(prev5[col]):
            item["chg_5d"] = round(float(last[col] - prev5[col]), 1)
        out.setdefault(region, []).append(item)
    return out


def index_history(history: list[dict], days: int = 30) -> dict:
    rows = [r for r in history if r.get("global") is not None][-days:]
    return {
        "dates": [r["date"] for r in rows],
        **{REGION_NAMES[k]: [None if r.get(k) is None else round(r[k], 1) for r in rows] for k in REGION_KEYS},
    }


def long_context(history: list[dict]) -> dict:
    """近一年各区域的最高/最低/均值，给“处在什么位置”做参照。"""
    rows = [r for r in history if r.get("global") is not None][-250:]
    out = {}
    for k in REGION_KEYS:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[REGION_NAMES[k]] = dict(min=round(min(vals), 1), max=round(max(vals), 1),
                                       mean=round(sum(vals) / len(vals), 1))
    return out


def astock_context() -> dict:
    out = {}
    insight = _read_json(ROOT / "docs" / "astock" / "insight.json")
    if insight:
        out["signal_date"] = insight.get("signal_date")
        out["market_temperature"] = insight.get("temperature")
        out["breadth_note"] = ("沪深全市场：up/down 上涨/下跌家数，limit_up/limit_down 涨停/跌停家数，amount 成交额（亿元），"
                               "above_ma20 站上 20 日均线的股票占比，median_ret 个股涨跌幅中位数，temperature 市场温度 0–100")
        market = insight.get("market") or []
        if market:
            keep = ["date", "up", "down", "limit_up", "limit_down", "amount", "above_ma20", "median_ret", "temperature"]
            out["market_breadth_last5"] = [{k: m.get(k) for k in keep} for m in market[-5:]]
        sectors = [s for s in insight.get("sectors") or [] if (s.get("stocks") or 0) >= 5]
        if sectors:
            sectors.sort(key=lambda s: s.get("strength") or 0, reverse=True)
            brief = lambda s: dict(industry=s.get("industry"), ret5_pct=round((s.get("ret5") or 0) * 100, 1),
                                   ret20_pct=round((s.get("ret20") or 0) * 100, 1))
            out["sectors_strongest"] = [brief(s) for s in sectors[:5]]
            out["sectors_weakest"] = [brief(s) for s in sectors[-5:]]
    picks = _read_json(ROOT / "docs" / "astock" / "picks.json")
    if picks:
        out["screen"] = {k: picks.get(k) for k in
                         ("signal_date", "trade_date", "gate_open", "emotion_on", "invest", "equity_vs_ma20")}
        out["screen"]["n_buy"] = sum(1 for p in picks.get("picks") or [] if p.get("action") == "买入")
        out["screen"]["n_sell"] = len(picks.get("sells") or [])
    return out


def build_payload() -> dict:
    latest = _read_json(ROOT / "docs" / "page" / "latest.json")
    history = _read_json(ROOT / "docs" / "page" / "history.json") or []
    if not latest:
        raise SystemExit("docs/page/latest.json 不存在，先跑每日更新")
    data_date = max(r["date"] for r in latest)
    asof = pd.Timestamp(data_date)
    return dict(
        data_date=data_date,
        scale="0 = 极度恐慌，100 = 极度贪婪",
        latest=latest,
        last_30_days=index_history(history),
        past_year_range=long_context(history),
        components_note="分项 fear_pct 是该指标在过去约五年里的分位（0–100），越高越恐慌；chg_5d 为 5 个交易日的变化",
        components=component_table(asof),
        markets=market_table(asof),
        astock=astock_context(),
    )


def call_claude(payload: dict) -> tuple[str, dict]:
    import anthropic

    client = anthropic.Anthropic(max_retries=4)
    user = ("下面是今天的数据（JSON）。请按系统提示写市场情绪解读。\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=1))
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        # 若模型出于安全分类拒答，服务端自动换模型重试
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": user}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"模型拒答：{resp.stop_details}")
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if not text:
        raise RuntimeError(f"没有返回正文，stop_reason={resp.stop_reason}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("输出被截断（max_tokens）")
    usage = dict(model=resp.model, input_tokens=resp.usage.input_tokens,
                 output_tokens=resp.usage.output_tokens, request_id=resp._request_id)
    return text, usage


def write_outputs(data_date: str, text: str, usage: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = f"# 市场情绪解读（{data_date} 收盘）\n\n{text}\n"
    (OUT_DIR / f"{data_date}.md").write_text(md)
    (ROOT / "docs" / "page" / "commentary.json").write_text(json.dumps(
        dict(date=data_date, generated_at=now, model=usage.get("model"), text=text),
        ensure_ascii=False, indent=1))

    readme = ROOT / "README.md"
    body = re.sub(r"^## ", "#### ", text, flags=re.M)
    block = (f"<!-- commentary:start -->\n### 今日情绪解读（{data_date} 收盘，Claude 自动生成）\n\n"
             f"{body}\n\n[往期解读](reports/sentiment/)\n<!-- commentary:end -->")
    content = readme.read_text()
    if "<!-- commentary:start -->" in content:
        content = re.sub(r"<!-- commentary:start -->.*<!-- commentary:end -->", lambda _: block, content, flags=re.S)
    else:  # 第一次：放在最新读数表格后面
        content = content.replace("<!-- latest:end -->", "<!-- latest:end -->\n\n" + block, 1)
    readme.write_text(content)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload = build_payload()
    data_date = payload["data_date"]
    if args.dry_run:
        print(SYSTEM_PROMPT, "\n\n", json.dumps(payload, ensure_ascii=False, indent=1))
        return 0
    if (OUT_DIR / f"{data_date}.md").exists() and not args.force:
        print(f"{data_date} 的解读已存在，跳过（--force 可重写）")
        return 0
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("没有设置 ANTHROPIC_API_KEY，跳过情绪解读")
        return 0
    text, usage = call_claude(payload)
    write_outputs(data_date, text, usage)
    print(f"已生成 {data_date} 情绪解读：{usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
