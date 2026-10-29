"""盘中任务：拉最新行情 → 估算 → 写入盘中数据目录 → 刷新 GitHub“盘中监控”Issue，有新情况时在 Issue 下评论提醒。

用法：python scripts/run_intraday.py --out <盘中数据目录>   （GitHub Actions 里是 intraday 分支的工作区）
      python scripts/run_intraday.py --any-open           （有市场在交易时段内时退出码为 0，供循环判断）
没有 GITHUB_TOKEN 时只写文件、打印提醒。

同时给 A股选股名单（docs/astock/picks.json）配上实时行情，写到 picks.json；当天走势写到 today.json。
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fearindex import fetch, intraday  # noqa: E402

ISSUE_TITLE = "盘中监控：恐慌贪婪指数"
PICKS_PATH = Path(__file__).resolve().parent.parent / "docs" / "astock" / "picks.json"
TODAY_HOURS = 24  # today.json 保留最近这么多小时的读数
ORDER = ["global", "us", "eu", "jp", "cn", "hk"]


def picks_table(rows):
    if not rows:
        return []
    lines = ["", f"**A股选股名单盘中**（交易日 {rows[0]['trade_date']}，止损/止盈带 * 的是参考价）", "",
             "| 股票 | 操作 | 现价 | 今日 | 相对买入价 | 止损 | 止盈 | 状态 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    pct = lambda v: "—" if v is None else f"{v:+.2f}%"
    star = lambda r, v: "—" if v is None else f"{v:g}" + ("*" if r["ref"] else "")
    for r in rows:
        lines.append(f"| {r['name']} {r['code']} | {r['action']} | {r['price'] if r['price'] is not None else '—'} | "
                     f"{pct(r['pct'])} | {pct(r['from_entry'])} | {star(r, r['stop'])} | {star(r, r['take'])} | {r['status']} |")
    return lines


def issue_body(est, errors, picks=None):
    t = datetime.fromisoformat(est["as_of"]).astimezone(intraday.CN_TZ)
    lines = [f"**更新于北京时间 {t:%Y-%m-%d %H:%M}**（交易时段每 2 分钟左右自动刷新）", "",
             "分数 0 = 极度恐慌，100 = 极度贪婪。盘中读数 = 如果现在收盘，指数是多少；正式读数以收盘后为准。", "",
             "| 区域 | 盘中 | 比上一交易日 | 如果现在收盘 | 说明 |", "| --- | --- | --- | --- | --- |"]
    for key in ORDER:
        r = est["regions"][key]
        if r["fresh"]:
            ch = f"{r['change']:+.1f}"
            note = f"{r['date']} {r['session']}" + ("（收盘读数，次日早上正式发布）" if r["session"] == "已收盘" else "")
        else:
            ch = "—"
            note = f"{r['session']}，显示 {r['official_date']} 收盘"
        lines.append(f"| {r['name']} | {r['score']:.1f}（{r['band']}） | {ch} | {r['signal_text']} | {note} |")
    lines += picks_table(picks or [])
    lines += ["", "<details><summary>行情时间</summary>", "", "| 序列 | 最新值 | 行情时间（北京） | 已计入 |", "| --- | --- | --- | --- |"]
    for k, q in sorted(est["quotes"].items()):
        qt = datetime.fromisoformat(q["time"]).astimezone(intraday.CN_TZ)
        lines.append(f"| {k} | {q['value']:g} | {qt:%m-%d %H:%M} | {'是' if q['fresh'] else '否'} |")
    lines += ["", "</details>"]
    if errors:
        lines += ["", "拉取失败（这些序列沿用上一交易日）：", ""] + [f"- {k}: {v}" for k, v in sorted(errors.items())]
    lines += ["", "提醒会以评论的形式发在这里：盘中急跌（比上一交易日低 6 分以上）、如果现在收盘会触发减仓或触底提示、"
              "进入极度恐慌或极度贪婪；持有的股票盘中触及止损价或止盈价。每种提醒一天最多一次。"]
    return "\n".join(lines)


class GitHub:
    def __init__(self, token, repo):
        import requests

        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
        self.base = f"https://api.github.com/repos/{repo}"

    def find_issue(self):
        r = self.s.get(f"{self.base}/issues", params={"state": "open", "per_page": 100, "creator": "github-actions[bot]"})
        r.raise_for_status()
        return next((i["number"] for i in r.json() if i["title"] == ISSUE_TITLE), None)

    def upsert_issue(self, number, body):
        if number:
            r = self.s.patch(f"{self.base}/issues/{number}", json={"body": body})
            if r.status_code != 404:
                r.raise_for_status()
                return number
        r = self.s.post(f"{self.base}/issues", json={"title": ISSUE_TITLE, "body": body})
        r.raise_for_status()
        return r.json()["number"]

    def comment(self, number, text):
        self.s.post(f"{self.base}/issues/{number}/comments", json={"body": text}).raise_for_status()


def write_today(out, now_cn):
    """最近 TODAY_HOURS 小时的盘中读数，每行一次运行，供在线总表画当天走势。"""
    frames = []
    for month in sorted({f"{now_cn:%Y-%m}", f"{now_cn - pd.Timedelta(hours=TODAY_HOURS):%Y-%m}"}):
        path = out / "log" / f"{month}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return
    df = pd.concat(frames)
    t = pd.to_datetime(df["time"])
    df = df[t >= pd.Timestamp(now_cn.replace(tzinfo=None)) - pd.Timedelta(hours=TODAY_HOURS)]
    df = df[["time"] + ORDER].drop_duplicates("time", keep="last")
    rows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in df.to_dict("records")]
    (out / "today.json").write_text(json.dumps(rows, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/intraday")
    ap.add_argument("--any-open", action="store_true")
    args = ap.parse_args()
    if args.any_open:
        return 0 if intraday.any_market_open() else 1
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(args.out)
    (out / "log").mkdir(parents=True, exist_ok=True)

    series = fetch.load_all()
    quotes, errors = intraday.fetch_quotes()
    for k, v in errors.items():
        logging.warning("%s", v)
    est = intraday.estimate(series, quotes)
    est["errors"] = errors
    (out / "latest.json").write_text(json.dumps(est, ensure_ascii=False, indent=2))
    # 在线总表用的精简版：每行一个区域
    t_cn = datetime.fromisoformat(est["as_of"]).astimezone(intraday.CN_TZ)
    page = [dict(key=k, name=r["name"], fresh=r["fresh"], date=r["date"], score=r["score"], change=r["change"],
                 band=r["band"], signal=r["signal_text"], session=r["session"], time=f"{t_cn:%m-%d %H:%M}") for k, r in est["regions"].items()]
    (out / "page.json").write_text(json.dumps(page, ensure_ascii=False, indent=1))

    picks, pick_notes = [], []
    if PICKS_PATH.exists():
        try:
            picks, pick_notes = intraday.pick_rows(json.loads(PICKS_PATH.read_text()))
        except Exception as exc:
            logging.warning("picks: %s: %s", type(exc).__name__, exc)
    (out / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1))

    # 每次运行一行，供当天走势和收盘后与正式读数核对
    now_cn = datetime.fromisoformat(est["as_of"]).astimezone(intraday.CN_TZ)
    row = {"time": now_cn.strftime("%Y-%m-%d %H:%M")}
    for key in ORDER:
        r = est["regions"][key]
        row[key] = r["score"] if r["fresh"] else None
        row[f"{key}_date"] = r["date"] if r["fresh"] else None
    log_path = out / "log" / f"{now_cn:%Y-%m}.csv"
    df = pd.DataFrame([row])
    df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    write_today(out, now_cn)

    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    sent = state.setdefault("sent", {})
    new = []
    for key, kind, text in intraday.alerts(est):
        tag = f"{est['regions'][key]['date']}:{key}:{kind}"
        if tag not in sent:
            new.append((tag, text))
    for code, kind, text in pick_notes:
        tag = f"{now_cn:%Y-%m-%d}:{code}:{kind}"
        if tag not in sent:
            new.append((tag, text))

    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if token and repo:
        gh = GitHub(token, repo)
        number = state.get("issue") or gh.find_issue()
        state["issue"] = gh.upsert_issue(number, issue_body(est, errors, picks))
        for tag, text in new:
            gh.comment(state["issue"], f"{text}\n\n北京时间 {now_cn:%m-%d %H:%M}，盘中预估，收盘后可能改变。")
            sent[tag] = now_cn.isoformat(timespec="minutes")
    else:
        for tag, text in new:
            logging.info("提醒（未发送，没有 GITHUB_TOKEN）：%s", text)
    # 只保留最近 30 天的去重记录
    cutoff = str((pd.Timestamp(now_cn.date()) - pd.Timedelta(days=30)).date())
    state["sent"] = {k: v for k, v in sent.items() if k[:10] >= cutoff}
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))

    for key in ORDER:
        r = est["regions"][key]
        logging.info("%s %s %s %s", r["name"], r["score"], "盘中" if r["fresh"] else f"沿用 {r['official_date']}",
                     r["signal_text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
