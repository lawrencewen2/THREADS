"""行情新闻：市场 7x24 快讯和候选股的公告、新闻，写 docs/astock/news.json。

数据源（都在 GitHub Actions 上抓，东方财富会拒绝 Actions 的 IP 所以不用）：
- 市场快讯：新浪财经 7x24（zhibo.sina.com.cn），失败时退到财联社电报（cls.cn）
- 候选股公告：巨潮资讯（cninfo.com.cn，证监会指定披露网站）
- 候选股新闻：新浪财经个股资讯页
每个源失败都只记一条状态，不影响其他源和当天的选股清单。思路借鉴 daily_stock_analysis、a-stock-data。
"""

import html
import json
import logging
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from . import config

log = logging.getLogger(__name__)

OUT_FILE = config.PAGE_DIR / "news.json"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
TIMEOUT = 20
MARKET_N = 60  # 市场快讯条数
STOCK_DAYS = 30  # 候选股公告、新闻回看天数
STOCK_N = 5  # 每只候选股最多列几条


def _get(url, data=None, headers=None, encoding="utf-8", attempts=3):
    h = {"User-Agent": UA, **(headers or {})}
    body = urllib.parse.urlencode(data).encode() if data else None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, data=body, headers=h)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read().decode(encoding, errors="replace")
        except Exception as exc:
            if i == attempts - 1:
                raise
            log.warning("retry %s: %s", url[:80], exc)
            time.sleep(2 * 2**i)


def _text(s):
    s = re.sub(r"<[^>]+>", "", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def _title_and_body(s):
    """快讯常写成“【标题】正文”；拆开，没有标题时取正文前 60 字。"""
    m = re.match(r"^【([^】]{2,80})】(.*)$", s)
    if m:
        return m.group(1), m.group(2).strip()
    return (s[:60] + ("…" if len(s) > 60 else "")), s


# ---------- 市场快讯 ----------

def sina_live(n=MARKET_N):
    url = f"https://zhibo.sina.com.cn/api/zhibo/feed?zhibo_id=152&page=1&page_size={n}&type=0"
    d = json.loads(_get(url, headers={"Referer": "https://finance.sina.com.cn/7x24/"}))
    rows = []
    for it in d["result"]["data"]["feed"]["list"]:
        title, body = _title_and_body(_text(it.get("rich_text")))
        tags = [t.get("name", "") for t in (it.get("tag") or []) if t.get("name")]
        rows.append(dict(time=it.get("create_time", "")[:16], title=title, body=body, tags="、".join(tags),
                         source="新浪财经 7x24", url=f"https://finance.sina.com.cn/7x24/"))
    return rows


def cls_telegraph(n=MARKET_N):
    d = json.loads(_get(f"https://www.cls.cn/nodeapi/telegraphList?rn={n}", headers={"Referer": "https://www.cls.cn/telegraph"}))
    rows = []
    for it in d["data"]["roll_data"]:
        content = _text(it.get("content"))
        title = _text(it.get("title")) or _title_and_body(content)[0]
        t = pd.Timestamp(int(it.get("ctime", 0)), unit="s", tz="UTC").tz_convert("Asia/Shanghai")
        rows.append(dict(time=t.strftime("%Y-%m-%d %H:%M"), title=title, body=content,
                         tags="重要" if str(it.get("level", "")).upper() in ("A", "B") else "",
                         source="财联社电报", url=it.get("shareurl") or "https://www.cls.cn/telegraph"))
    return rows


# ---------- 候选股 ----------

def cninfo_announcements(code, days=STOCK_DAYS):
    """巨潮资讯公告。code 形如 sz.002772。"""
    num = code.split(".")[1]
    hit = json.loads(_get("http://www.cninfo.com.cn/new/information/topSearch/query",
                          data={"keyWord": num, "maxNum": 5}))
    org = next((h["orgId"] for h in hit if h.get("code") == num), None)
    if not org:
        return []
    end = pd.Timestamp.now(tz="Asia/Shanghai").normalize()
    start = end - pd.Timedelta(days=days)
    d = json.loads(_get("http://www.cninfo.com.cn/new/hisAnnouncement/query", data={
        "stock": f"{num},{org}", "tabName": "fulltext", "pageSize": STOCK_N, "pageNum": 1,
        "column": "szse" if code.startswith("sz") else "sse", "plate": "", "category": "", "trade": "",
        "seDate": f"{start:%Y-%m-%d}~{end:%Y-%m-%d}", "isHLtitle": "true", "sortName": "", "sortType": "",
    }, headers={"Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"}))
    rows = []
    for a in (d.get("announcements") or [])[:STOCK_N]:
        t = pd.Timestamp(int(a["announcementTime"]), unit="ms", tz="UTC").tz_convert("Asia/Shanghai")
        rows.append(dict(time=t.strftime("%Y-%m-%d"), title=_text(a.get("announcementTitle")), kind="公告",
                         source="巨潮资讯", url="http://static.cninfo.com.cn/" + a.get("adjunctUrl", "")))
    return rows


def sina_stock_news(code, days=STOCK_DAYS):
    sym = code.replace(".", "")
    page = _get(f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllNewsStock/symbol/{sym}.phtml", encoding="gbk")
    m = re.search(r'<div class="datelist">(.*?)</div>', page, re.S)
    if not m:
        return []
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    rows = []
    for date, hm, href, title in re.findall(r"(\d{4}-\d{2}-\d{2})&nbsp;(\d{2}:\d{2})&nbsp;&nbsp;<a[^>]*href='([^']+)'[^>]*>(.*?)</a>",
                                            m.group(1)):
        if pd.Timestamp(date) < cutoff:
            break
        rows.append(dict(time=f"{date} {hm}", title=_text(title), kind="新闻", source="新浪财经", url=href))
        if len(rows) >= STOCK_N:
            break
    return rows


# ---------- 汇总 ----------

def tag_market(rows, names, industries):
    """标出提到候选股或强弱行业的快讯，方便先看和自己相关的。"""
    for r in rows:
        text = r["title"] + r["body"]
        hits = [n for n in names if n and n in text]
        hits += [i for i in industries if i and i in text]
        r["related"] = "、".join(dict.fromkeys(hits))
    return rows


def short_industry(name):
    """证监会行业名很长（如“计算机、通信和其他电子设备制造业”），取第一个词用于在快讯里匹配。"""
    s = re.split(r"[、和及与]", name)[0]
    return re.sub(r"(制造业|业)$", "", s)


def build(picks, sectors):
    status = []
    market = []
    for name, fn in (("新浪财经 7x24", sina_live), ("财联社电报", cls_telegraph)):
        try:
            market = fn()
            status.append(dict(source=name, ok=True, rows=len(market)))
            break
        except Exception as exc:
            log.warning("%s failed: %s", name, exc)
            status.append(dict(source=name, ok=False, error=str(exc)[:200]))
    cands = [(pk["code"], pk.get("name", "")) for pk in (picks or {}).get("picks", [])]
    inds = [short_industry(s["industry"]) for s in (sectors or [])[:5] + (sectors or [])[-5:]]
    tag_market(market, [n for _, n in cands], [i for i in inds if len(i) >= 2])
    stock = []
    for fn, name in ((cninfo_announcements, "巨潮资讯公告"), (sina_stock_news, "新浪个股新闻")):
        ok = 0
        err = None
        for code, nm in cands:
            try:
                for r in fn(code):
                    stock.append(dict(code=code, name=nm, **r))
                ok += 1
            except Exception as exc:
                err = str(exc)[:200]
                log.warning("%s %s failed: %s", name, code, exc)
            time.sleep(0.5)
        status.append(dict(source=name, ok=ok > 0, rows=ok, error=err))
    stock.sort(key=lambda r: (r["code"], r["time"]), reverse=False)
    for i, r in enumerate(market):
        r["id"] = f"m{i}"
    for i, r in enumerate(stock):
        r["id"] = f"s{i}"
    return dict(updated=pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M"),
                market=market, stocks=stock, status=status)


def write(out):
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False))
