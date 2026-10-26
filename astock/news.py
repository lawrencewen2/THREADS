"""行情新闻：市场 7x24 快讯和候选股的公告、新闻，写 docs/astock/news.json。

数据源（都在 GitHub Actions 上抓，东方财富会拒绝 Actions 的 IP 所以不用）：
- 市场快讯：新浪财经 7x24（zhibo.sina.com.cn），失败时退到财联社电报（cls.cn）
- 候选股公告：巨潮资讯（cninfo.com.cn，证监会指定披露网站）
- 候选股新闻：新浪财经个股资讯页
每个源失败都只记一条状态，不影响其他源和当天的选股清单。思路借鉴 daily_stock_analysis、a-stock-data。
"""

import html
import json
import math
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
WEEK_DAYS = 7  # 本周大事回看天数
WEEK_N = 15  # 本周大事列几件
WEEK_PAGES = 160  # 往回翻快讯的最多页数（每页 100 条，一天约 1700 条）
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


def sina_live_pages(days=WEEK_DAYS, page_size=100, max_pages=WEEK_PAGES):
    """往回翻新浪 7x24，直到早于 days 天前。保留标签，用来判断重要性。"""
    cutoff = (pd.Timestamp.now(tz="Asia/Shanghai") - pd.Timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
    rows = []
    for page in range(1, max_pages + 1):
        url = f"https://zhibo.sina.com.cn/api/zhibo/feed?zhibo_id=152&page={page}&page_size={page_size}&type=0"
        items = json.loads(_get(url, headers={"Referer": "https://finance.sina.com.cn/7x24/"}))["result"]["data"]["feed"]["list"]
        if not items:
            break
        for it in items:
            title, body = _title_and_body(_text(it.get("rich_text")))
            tags = [t.get("name", "") for t in (it.get("tag") or []) if t.get("name")]
            rows.append(dict(time=it.get("create_time", "")[:16], title=title, body=body, tags=tags))
        if rows[-1]["time"] < cutoff:
            break
        time.sleep(0.3)
    return [r for r in rows if r["time"] >= cutoff]


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


# ---------- 本周大事 ----------

# (类别, 权重, 关键词)。权重是对 A 股影响面大小的粗略排序，不是回测出来的。
CATEGORIES = [
    ("货币政策", 5, r"降准|降息|LPR|MLF|逆回购利率|央行宣布|人民银行宣布|美联储|加息|议息|利率决议|欧洲央行|日本央行|日银"),
    ("重大政策", 5, r"国务院|政治局|中央经济工作会议|国常会|全会|两会|特别国债|财政部.*(发行|赤字)|发改委.*(出台|印发)|稳增长|刺激"),
    ("监管与资本市场", 4, r"证监会|印花税|汇金|平准|国家队|减持新规|退市新规|交易所.*(发布|修订)|融资融券|程序化交易"),
    ("地缘与贸易", 4, r"关税|制裁|贸易战|出口管制|实体清单|停火|战争|冲突升级|导弹|空袭"),
    ("宏观数据", 3, r"GDP|CPI|PPI|PMI|社融|新增贷款|M2|非农|失业率|零售销售|工业增加值|出口同比|进口同比|外汇储备"),
    ("市场大波动", 3, r"暴跌|暴涨|大跌|大涨|熔断|跌停潮|涨停潮|千股|创.{0,6}(新高|新低)|成交额.{0,6}(突破|超|万亿)|汇率.{0,6}(破|跌破|升破)"),
    ("大宗商品", 2, r"原油|油价|OPEC|金价|黄金|铜价|铁矿石"),
    ("海外市场", 2, r"美股|纳指|标普|道指|日经|恒指|港股|欧股|美债收益率"),
    ("公司大事", 2, r"IPO|递交.{0,6}上市申请|并购|重组|破产|财报|业绩预告"),
]
# 行情自动播报（某合约日内涨 X%、资金净买入）和每日固定栏目不算“大事”，否则会把真正的新闻挤下去
ROUTINE_RE = re.compile(r"日内(涨|跌)|现报|涨超|跌超|涨幅扩大|跌幅扩大|净(买入|卖出)|异动|逆回购操作|新闻联播|头版|主要内容"
                        r"|早餐|晚报|午报|早报|收评|午评|盘前|要闻|一览|汇总|集锦")
INDUSTRY_STOP = {"研究", "专业", "其他", "综合", "批发", "零售", "商务服务", "食品"}
CAT_RE = [(n, w, re.compile(k)) for n, w, k in CATEGORIES]
FOCUS_TAGS = ("焦点", "重要", "头条")


def importance(text, tags):
    """单条快讯的重要性分和命中的类别。"""
    hits = [(n, w) for n, w, rx in CAT_RE if rx.search(text)]
    score = (max(w for _, w in hits) + 0.5 * (len(hits) - 1)) if hits else 0.0
    if any(f in t for t in tags for f in FOCUS_TAGS):
        score += 2
    return score, [n for n, _ in hits]


def _grams(s):
    s = re.sub(r"[\s\W_]+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def week_events(rows, names=(), industries=(), n=WEEK_N, sim=0.3):
    """把一周快讯里讲同一件事的合并，按 单条最高分 + 报道条数 + 和候选股相关 排序。"""
    events = []
    for r in sorted(rows, key=lambda r: r["time"]):
        text = r["title"] + r["body"]
        if ROUTINE_RE.search(r["title"]):
            continue
        score, cats = importance(text, r.get("tags") or [])
        if score <= 0:
            continue
        g = _grams(r["title"])
        best = None
        for e in events[-300:]:
            inter = len(g & e["grams"])
            if inter and inter / len(g | e["grams"]) >= sim:
                best = e
                break
        if best is None:
            best = dict(grams=g, items=[], cats=set())
            events.append(best)
        best["items"].append(dict(r, score=score))
        best["cats"].update(cats)
    out = []
    for e in events:
        top = max(e["items"], key=lambda x: (x["score"], len(x["body"])))
        text = " ".join(x["title"] + x["body"] for x in e["items"])
        related = [k for k in list(names) + [i for i in industries if i not in INDUSTRY_STOP] if k and k in text]
        weight = top["score"] + 2 * math.log(len(e["items"]))
        weight += 1 if related else 0
        out.append(dict(title=top["title"], body=top["body"][:300], first=e["items"][0]["time"], last=e["items"][-1]["time"],
                        reports=len(e["items"]), category="、".join(c for c, _, _ in CATEGORIES if c in e["cats"]),
                        focus=any(f in t for x in e["items"] for t in (x.get("tags") or []) for f in FOCUS_TAGS),
                        related="、".join(dict.fromkeys(related)), weight=round(weight, 2)))
    out.sort(key=lambda x: (x["weight"], x["last"]), reverse=True)
    for i, x in enumerate(out[:n]):
        x["rank"] = i + 1
        x["id"] = f"w{i}"
    return out[:n]


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
    week = []
    try:
        raw = sina_live_pages()
        tag_count = pd.Series([t for r in raw for t in r["tags"]]).value_counts().head(15).to_dict()
        week = week_events(raw, [n for _, n in cands], [i for i in inds if len(i) >= 2])
        status.append(dict(source="本周大事（新浪 7x24 近 7 天）", ok=True, rows=len(raw), tags=tag_count,
                           oldest=raw[-1]["time"] if raw else None))
    except Exception as exc:
        log.warning("week events failed: %s", exc)
        status.append(dict(source="本周大事（新浪 7x24 近 7 天）", ok=False, error=str(exc)[:200]))
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
                market=market, week=week, stocks=stock, status=status)


def write(out):
    config.PAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False))
