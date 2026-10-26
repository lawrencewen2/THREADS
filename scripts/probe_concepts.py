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


def tencent_list():
    out = []
    for off in (0, 100, 200, 300):
        d = json.loads(get(f"https://proxy.finance.qq.com/cgi/cgi-bin/rank/pt/getRank?board_type=gn&sort_type=price&direct=down&offset={off}&count=100"))
        out += [(x["code"], x["name"]) for x in d["data"]["rank_list"]]
    hits = [x for x in out if re.search("航天|半导体|芯片|算力", x[1])]
    return len(out), d["data"].get("total"), hits


def tencent_members():
    out = []
    for u in ["https://proxy.finance.qq.com/cgi/cgi-bin/rank/hs/getBoardRankList?_appver=11.17.0&board_code=pt02031398&sort_type=price&direct=down&offset=0&count=20",
              "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/mktHs/rank?l=20&p=1&t=01/averatio&ordertype=&o=0&bkid=pt02031398",
              "https://proxy.finance.qq.com/cgi/cgi-bin/rank/hs/getBoardRankList?board_code=pt02031398&sort_type=price&direct=down&offset=0&count=20"]:
        try:
            out.append(get(u)[:500])
        except Exception as exc:
            out.append(repr(exc))
    return out


for n, f in [("tencent_list", tencent_list), ("tencent_members", tencent_members)]:
    probe(n, f)
