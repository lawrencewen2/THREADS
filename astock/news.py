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


def _ext_stocks(it):
    """新浪快讯 ext 里点名的股票，转成 (代码, 名称)。A股写成 sh.600000，港美股保留原样。"""
    ext = it.get("ext") or {}
    if isinstance(ext, str):
        try:
            ext = json.loads(ext)
        except ValueError:
            return []
    out = []
    for st in (ext.get("stocks") or []) if isinstance(ext, dict) else []:
        sym, name, mkt = str(st.get("symbol", "")).lower(), st.get("key") or st.get("name") or "", st.get("market", "")
        m = re.match(r"^(sh|sz|bj)(\d{6})$", sym)
        if m:
            sym = f"{m.group(1)}.{m.group(2)}"
        elif mkt == "hk":
            sym = sym.upper().lstrip("HK") + ".HK"
        elif mkt == "us":
            sym = sym.upper()
        if sym and name:
            out.append((sym, name))
    return out


def sina_live_pages(days=WEEK_DAYS, page_size=100, max_pages=WEEK_PAGES):
    """往回翻新浪 7x24，直到早于 days 天前。保留标签，用来判断重要性。"""
    cutoff = (pd.Timestamp.now(tz="Asia/Shanghai") - pd.Timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
    rows, seen = [], set()
    for page in range(1, max_pages + 1):
        url = f"https://zhibo.sina.com.cn/api/zhibo/feed?zhibo_id=152&page={page}&page_size={page_size}&type=0"
        items = json.loads(_get(url, headers={"Referer": "https://finance.sina.com.cn/7x24/"}))["result"]["data"]["feed"]["list"]
        # 翻过新浪能给的最深一页后，接口会一直返回同一页，按 id 去重并在没有新内容时停下
        items = [it for it in items if it.get("id") not in seen]
        if not items:
            break
        seen.update(it.get("id") for it in items)
        for it in items:
            title, body = _title_and_body(_text(it.get("rich_text")))
            tags = [t.get("name", "") for t in (it.get("tag") or []) if t.get("name")]
            rows.append(dict(time=it.get("create_time", "")[:16], title=title, body=body, tags=tags, stocks=_ext_stocks(it)))
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
    ("重大会议", 7, r"中全会|政治局(会议|召开)|中央经济工作会议|中央金融工作会议|国务院常务会议|国常会|两会|陆家嘴论坛"),
    ("货币政策", 5, r"降准|降息|LPR|MLF|逆回购利率|央行宣布|人民银行宣布|美联储|加息|议息|利率决议|欧洲央行|日本央行|日银"),
    ("重大政策", 5, r"(?<!美国)(?<![美俄英日韩法德])国务院|中办|国办|中共中央|特别国债|财政部.*(发行|赤字)|发改委.*(出台|印发)|稳增长|刺激"),
    ("监管与资本市场", 4, r"证监会|印花税|汇金|平准|国家队|减持新规|退市新规|交易所.*(发布|修订)|融资融券|程序化交易"),
    ("地缘与贸易", 4, r"关税|制裁|贸易战|出口管制|实体清单|停火|战争|冲突升级|导弹|空袭|撤离|袭击|霍尔木兹|中东局势"),
    ("宏观数据", 3, r"GDP|CPI|PPI|PMI|社融|新增贷款|M2|非农|失业率|零售销售|工业增加值|出口同比|进口同比|外汇储备"),
    ("市场大波动", 3, r"暴跌|暴涨|大跌|大涨|熔断|跌停潮|涨停潮|千股|创.{0,6}(新高|新低)|成交额.{0,6}(突破|超|万亿)|汇率.{0,6}(破|跌破|升破)"),
    ("大宗商品", 2, r"原油|油价|OPEC|金价|黄金|铜价|铁矿石"),
    ("海外市场", 2, r"美股|纳指|标普|道指|日经|恒指|港股|欧股|美债收益率"),
    ("公司大事", 2, r"IPO|递交.{0,6}上市申请|并购|重组|破产|财报|业绩预告"),
]
# 行情自动播报（某合约日内涨 X%、资金净买入）和每日固定栏目不算“大事”，否则会把真正的新闻挤下去
ROUTINE_RE = re.compile(r"日内(涨|跌)|现报|涨超|跌超|涨幅扩大|跌幅扩大|净(买入|卖出)|异动|逆回购操作|新闻联播|头版|主要内容"
                        r"|早餐|晚报|午报|早报|收评|午评|盘前|要闻|一览|汇总|集锦"
                        r"|成交额超|成交额突破|异常交易行为")
# 对 A 股来说国内的事优先；只讲其他国家、和中国无关的降一档（美国、美联储不降，影响面大）
CHINA_RE = re.compile(r"中国|我国|(?<!美国)(?<![美俄英日韩法德])国务院|中办|国办|中共中央|政治局|全会|证监会|人民银行|中国央行|A股|沪指|深成指|创业板|科创板|上证|人民币"
                      r"|发改委|财政部|商务部|国家统计局|工信部|外汇局|两市|北向|南向|沪深|港股|恒指")
FOREIGN_RE = re.compile(r"韩国|印尼|印度|英国|日本|日银|澳洲|澳大利亚|新西兰|巴西|土耳其|加拿大|墨西哥|南非|新台币|台湾|欧元区|欧洲央行|瑞士|挪威|瑞典")
INDUSTRY_STOP = {"研究", "专业", "其他", "综合", "批发", "零售", "商务服务", "食品"}
CAT_RE = [(n, w, re.compile(k)) for n, w, k in CATEGORIES]

# 事件主题对应的 A股核心标的（行业龙头，受影响最直接的方向）。快讯没点名股票时用它。
# 只是“和这件事关联最直接的股票”，不是买卖建议。
THEMES = [
    ("原油", r"原油|油价|OPEC|欧佩克|布伦特|WTI|泵站|炼厂", [("sh.600938", "中国海油"), ("sh.601857", "中国石油"), ("sh.600028", "中国石化")]),
    ("中东航运", r"红海|霍尔木兹|胡塞|苏伊士|航运|集运|油轮", [("sh.601919", "中远海控"), ("sh.601872", "招商轮船")]),
    ("黄金", r"黄金|金价|贵金属", [("sh.601899", "紫金矿业"), ("sh.600547", "山东黄金"), ("sh.600489", "中金黄金")]),
    ("铜铝", r"铜价|铜矿|电解铝|铝价|铝企", [("sh.601899", "紫金矿业"), ("sh.600362", "江西铜业"), ("sh.601600", "中国铝业")]),
    ("稀土", r"稀土", [("sh.600111", "北方稀土"), ("sh.600392", "盛和资源")]),
    ("人工智能", r"人工智能|(?<![A-Za-z])AI(?![A-Za-z])|大模型|算力|智能体", [("sh.688256", "寒武纪"), ("sz.300308", "中际旭创"), ("sz.002230", "科大讯飞")]),
    ("半导体", r"集成电路|半导体|芯片|光刻|工业母机", [("sh.688981", "中芯国际"), ("sz.002371", "北方华创")]),
    ("机器人", r"机器人|人形", [("sz.300124", "汇川技术"), ("sz.002472", "双环传动")]),
    ("军工", r"军工|导弹|空袭|无人机袭击|战机|国防", [("sh.600760", "中航沈飞"), ("sh.600893", "航发动力")]),
    ("银行", r"降准|降息|LPR|MLF|逆回购|存款利率|信贷|新增贷款|社融", [("sh.600036", "招商银行"), ("sh.601398", "工商银行")]),
    ("券商", r"证监会|资本市场|汇金|平准|印花税|融资融券|IPO|注册制|成交额|两融", [("sh.600030", "中信证券"), ("sz.300059", "东方财富")]),
    ("保险", r"保险|险资", [("sh.601318", "中国平安"), ("sh.601628", "中国人寿")]),
    ("基建", r"地方政府债|专项债|新增地方|基建|城市更新|水利|特别国债", [("sh.601668", "中国建筑"), ("sh.601390", "中国中铁"), ("sh.600585", "海螺水泥")]),
    ("地产", r"房地产|楼市|房贷|住房|保交楼|房企", [("sh.600048", "保利发展"), ("sz.000002", "万科A")]),
    ("消费", r"消费|零售|以旧换新|家电|白酒|免税|服务业", [("sh.600519", "贵州茅台"), ("sz.000333", "美的集团"), ("sh.601888", "中国中免")]),
    ("汽车", r"汽车|新能源车|电动车|柴油车|燃油车", [("sz.002594", "比亚迪"), ("sh.601127", "赛力斯")]),
    ("电池储能", r"电池|锂|储能", [("sz.300750", "宁德时代"), ("sz.300014", "亿纬锂能")]),
    ("光伏", r"光伏|硅料|组件", [("sh.601012", "隆基绿能"), ("sh.600438", "通威股份")]),
    ("电力", r"电力|电网|特高压|核电|水电", [("sh.600900", "长江电力"), ("sh.600406", "国电南瑞")]),
    ("煤炭", r"煤炭|煤价|动力煤|焦煤", [("sh.601088", "中国神华"), ("sh.601225", "陕西煤业")]),
    ("医药", r"医药|创新药|医保|集采|疫苗", [("sh.600276", "恒瑞医药"), ("sh.603259", "药明康德")]),
    ("出口", r"关税|出口管制|贸易战|实体清单|出口", [("sz.002475", "立讯精密"), ("sz.000333", "美的集团")]),
    ("航空", r"航空|航班|航油", [("sh.601111", "中国国航"), ("sh.600029", "南方航空")]),
    ("美联储", r"美联储|美债|美元指数|加息|议息", [("sh.601899", "紫金矿业"), ("sh.600030", "中信证券")]),
]
THEME_RE = [(n, re.compile(k), st) for n, k, st in THEMES]
BROAD = [("sh.510300", "沪深300ETF"), ("sh.510050", "上证50ETF")]


def event_targets(items, text, n=3):
    """一件事最直接关联的标的：快讯里点名次数最多的股票优先，不够再按主题补龙头；都没有时给宽基 ETF。"""
    named = pd.Series([s for it in items for s in (it.get("stocks") or [])], dtype=object).value_counts()
    out = [dict(code=c, name=nm, why="快讯点名") for (c, nm) in list(named.index)[:n]]
    themes = sorted(((len(rx.findall(text)), name, st) for name, rx, st in THEME_RE if rx.search(text)), key=lambda t: -t[0])
    for k, (_, name, st) in enumerate(themes):
        cap = len(out) + (2 if k < len(themes) - 1 else n)  # 多个主题时每个主题最多两只，让标的覆盖不同方向
        for c, nm in st:
            if len(out) >= min(n, cap):
                break
            if c not in {o["code"] for o in out}:
                out.append(dict(code=c, name=nm, why=name))
        if len(out) >= n:
            break
    if not out:
        out = [dict(code=c, name=nm, why="大盘") for c, nm in BROAD[:1]]
    return out
FOCUS_TAGS = ("焦点", "重要", "头条")


def importance(text, tags, title=None):
    """单条快讯的重要性分和命中的类别。重大会议只看标题：正文里“学习贯彻全会精神”之类的套话不算。"""
    hits = [(n, w) for n, w, rx in CAT_RE if rx.search(title if (n == "重大会议" and title) else text)]
    score = (max(w for _, w in hits) + min(1.0, 0.5 * (len(hits) - 1))) if hits else 0.0
    if hits and CHINA_RE.search(text):
        score += 2
    elif hits and FOREIGN_RE.search(text):
        score -= 2
    if any(f in t for t in tags for f in FOCUS_TAGS):
        score += 2
    return score, [n for n, _ in hits]


def _grams(s):
    s = re.sub(r"[\s\W_]+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _minutes(a, b):
    return abs((pd.Timestamp(b) - pd.Timestamp(a)).total_seconds()) / 60


def week_events(rows, names=(), industries=(), n=WEEK_N, sim=0.3):
    """把一周快讯里讲同一件事的合并，按 单条最高分 + 报道条数 + 和候选股相关 排序。"""
    events = []
    for r in sorted(rows, key=lambda r: r["time"]):
        text = r["title"] + r["body"]
        if ROUTINE_RE.search(r["title"]):
            continue
        score, cats = importance(text, r.get("tags") or [], r["title"])
        if score <= 0:
            continue
        g = _grams(r["title"])
        m = re.match(r"^([^：:]{2,12})[：:]", r["title"])
        speaker = m.group(1) if m else None
        best = None
        for e in events[-300:]:
            inter = len(g & e["grams"])
            # 标题相近，或一条标题基本包含在另一条里（短讯和详讯），或同一发布方半小时内连发（一份文件拆成多条）
            if inter and (inter / len(g | e["grams"]) >= sim or inter / min(len(g), len(e["grams"])) >= 0.7):
                best = e
                break
            if speaker and speaker == e.get("speaker") and _minutes(e["items"][-1]["time"], r["time"]) <= 30:
                best = e
                break
        if best is None:
            best = dict(grams=g, items=[], cats=set(), speaker=speaker)
            events.append(best)
        best["items"].append(dict(r, score=score))
        best["cats"].update(cats)
    out = []
    for e in events:
        # 代表标题取分最高的；同分时取最早的一条，通常是总标题而不是拆分出来的细项
        top = max(enumerate(e["items"]), key=lambda t: (t[1]["score"], -t[0]))[1]
        text = " ".join(x["title"] + x["body"] for x in e["items"])
        related = [k for k in list(names) + [i for i in industries if i not in INDUSTRY_STOP] if k and k in text]
        weight = top["score"] + 2 * math.log(len(e["items"]))
        weight += 1 if related else 0
        out.append(dict(title=top["title"], body=top["body"][:300], first=e["items"][0]["time"], last=e["items"][-1]["time"],
                        reports=len(e["items"]), category="、".join(c for c, _, _ in CATEGORIES if c in e["cats"]),
                        focus=any(f in t for x in e["items"] for t in (x.get("tags") or []) for f in FOCUS_TAGS),
                        related="、".join(dict.fromkeys(related)), weight=round(weight, 2),
                        targets=event_targets(e["items"], text)))
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
                           with_stocks=sum(1 for r in raw if r.get("stocks")),
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
