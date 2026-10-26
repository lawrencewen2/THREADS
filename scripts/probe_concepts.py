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


def sina_classes():
    s = get("http://vip.stock.finance.sina.com.cn/q/view/newFLJK.php?param=class", "gbk")
    items = re.findall(r'"(gn_\w+)":"gn_\w+,([^,]+),(\d+)', s)
    return len(items), items[:15], s[:200]


def sina_members():
    s = get("http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=1&num=80&sort=symbol&asc=1&node=gn_syht&symbol=&_s_r_a=init", "gbk")
    return s[:500]


def ak_ths():
    import akshare as ak
    df = ak.stock_board_concept_name_ths()
    return df.shape, df.head(5).to_dict("records")


def ak_ths_cons():
    import akshare as ak
    fn = getattr(ak, "stock_board_concept_cons_ths", None)
    if fn is None:
        return "no stock_board_concept_cons_ths in this akshare"
    df = fn(symbol="商业航天")
    return df.shape, df.head(5).to_dict("records")


def ak_em():
    import akshare as ak
    df = ak.stock_board_concept_name_em()
    return df.shape, df.head(3).to_dict("records")


def ak_em_cons():
    import akshare as ak
    df = ak.stock_board_concept_cons_em(symbol="商业航天")
    return df.shape, df.head(3).to_dict("records")


def ths_page():
    s = get("http://q.10jqka.com.cn/gn/", "gbk", "http://q.10jqka.com.cn/")
    return len(s), re.findall(r'href="http://q.10jqka.com.cn/gn/detail/code/(\d+)/"[^>]*>([^<]+)<', s)[:10]


def tencent_concepts():
    s = get("https://proxy.finance.qq.com/cgi/cgi-bin/rank/pt/getRank?board_type=gn&sort_type=price&direct=down&offset=0&count=20")
    return s[:600]


for n, f in [("sina_classes", sina_classes), ("sina_members", sina_members), ("ths_page", ths_page),
             ("tencent_concepts", tencent_concepts), ("ak_ths", ak_ths), ("ak_ths_cons", ak_ths_cons),
             ("ak_em", ak_em), ("ak_em_cons", ak_em_cons)]:
    probe(n, f)
