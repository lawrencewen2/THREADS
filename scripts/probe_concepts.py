"""探测哪些概念板块数据源在 GitHub Actions 上能用（一次性，用完删除）。"""
import json, re, time, traceback, urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def get(url, enc="utf-8", ref=None):
    h = {"User-Agent": UA}
    if ref:
        h["Referer"] = ref
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=20) as r:
        return r.read().decode(enc, errors="replace")


def probe(name, fn):
    t = time.time()
    try:
        out = fn()
        print(f"OK {name} ({time.time()-t:.1f}s): {str(out)[:600]}")
    except Exception as exc:
        print(f"FAIL {name}: {exc!r}")


def sina_members():
    out = {}
    for node in ("gn_hwqc", "gn_syht", "gn_htjg", "gn_xp"):
        u = ("http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
             f"?page=1&num=100&sort=symbol&asc=1&node={node}&symbol=&_s_r_a=init")
        out[node] = get(u, "gbk", "http://vip.stock.finance.sina.com.cn/")[:300]
    cnt = get("http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeStockCount?node=gn_hwqc", "gbk")
    out["count"] = cnt
    return out


def sina_classes_aero():
    s = get("http://vip.stock.finance.sina.com.cn/q/view/newFLJK.php?param=class", "gbk")
    items = re.findall(r'"(gn_\w+)":"gn_\w+,([^,]+),(\d+)', s)
    return [i for i in items if re.search("航天|芯片|半导体|卫星|AI|人工智能|算力", i[1])]


def ths_detail():
    out = []
    for page in (1, 2):
        u = f"http://q.10jqka.com.cn/gn/detail/field/264648/order/desc/page/{page}/ajax/1/code/308614"
        try:
            s = get(u, "gbk", "http://q.10jqka.com.cn/gn/detail/code/308614/")
            out.append((page, len(s), re.findall(r'stockpage.10jqka.com.cn/(\d{6})/" target="_blank">([^<]+)<', s)[:5], s[:200]))
        except Exception as exc:
            out.append((page, repr(exc)))
    s = get("http://q.10jqka.com.cn/gn/detail/code/308614/", "gbk", "http://q.10jqka.com.cn/gn/")
    out.append(("main", len(s), re.findall(r'page_info">(\d+/\d+)<', s), re.findall(r'stockpage.10jqka.com.cn/(\d{6})/', s)[:5]))
    return out


def tencent_members():
    out = []
    for u in ["https://proxy.finance.qq.com/cgi/cgi-bin/rank/pt/getRank?board_type=gn&sort_type=price&direct=down&offset=0&count=3",
              "https://proxy.finance.qq.com/cgi/cgi-bin/rank/hs/getBoardRankList?_appver=11.17.0&board_code=pt01801081&sort_type=price&direct=down&offset=0&count=20"]:
        try:
            out.append(get(u)[:700])
        except Exception as exc:
            out.append(repr(exc))
    return out


for n, f in [("sina_members", sina_members), ("sina_classes_aero", sina_classes_aero),
             ("ths_detail", ths_detail), ("tencent_members", tencent_members)]:
    probe(n, f)
