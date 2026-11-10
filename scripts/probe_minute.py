"""探测分钟线数据源在 Actions 上能拿到多长历史（早盘放量研究用，一次性）。"""
import json, re, requests
UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}
def show(name, f):
    try:
        print(name, f())
    except Exception as e:
        print(name, "ERR", repr(e)[:200])
def sina(scale, n=1023):
    r = requests.get("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData",
                     params=dict(symbol="sz000001", scale=scale, ma="no", datalen=n), headers=UA, timeout=20)
    d = json.loads(r.text) if r.text.strip().startswith("[") else []
    return len(d), d[:2], d[-1:] if d else r.text[:200]
def tencent(k, n):
    r = requests.get("https://ifzq.gtimg.cn/appstock/app/kline/mkline", params=dict(param=f"sz000001,{k},,{n}"), headers=UA, timeout=20)
    d = r.json()["data"]["sz000001"][k]
    return len(d), d[:2], d[-1:]
def tmin():
    r = requests.get("https://web.ifzq.gtimg.cn/appstock/app/minute/query", params=dict(code="sz000001"), headers=UA, timeout=20)
    d = r.json()["data"]["sz000001"]["data"]["data"]
    return len(d), d[:3]
def tday5():
    r = requests.get("https://web.ifzq.gtimg.cn/appstock/app/day/query", params=dict(code="sz000001"), headers=UA, timeout=20)
    d = r.json()["data"]["sz000001"]["data"]
    return len(d), [x["date"] for x in d], d[0]["data"][:3]
def bao():
    import baostock as bs
    bs.login()
    rs = bs.query_history_k_data_plus("sz.000001", "date,time,open,close,volume,amount", start_date="2018-01-01", end_date="2026-11-10", frequency="60", adjustflag="3")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return rs.error_code, len(rows), rows[:2], rows[-1:]
for s in (5, 30, 60, 240):
    show(f"sina{s}", lambda: sina(s))
for k, n in (("m1", 2000), ("m5", 2000), ("m30", 2000), ("m60", 2000)):
    show(f"tencent{k}", lambda: tencent(k, n))
show("tencent_minute_today", tmin)
show("tencent_5day", tday5)
show("baostock60", bao)
