"""akshare 接口更名兼容层。

akshare 1.20.44 起把一批接口改了名（旧名直接抛 AttributeError）。这里按新名优先、旧名兜底取函数，
新旧版本的 akshare 都能用。新增更名时只需在 RENAMES 里加一行。
"""

RENAMES = {
    # 旧名: 新名
    "stock_zh_index_daily_em": "index_zh_daily_em",
    "stock_zh_index_daily_tx": "index_zh_daily_tx",
    "stock_zh_index_daily": "index_zh_daily_sina",
    "stock_hk_index_daily_sina": "index_hk_daily_sina",
    "index_us_stock_sina": "index_us_daily_sina",
    "futures_foreign_hist": "futures_foreign_hist_sina",
    "stock_us_daily": "stock_us_hist_sina",
    "macro_china_market_margin_sh": "stock_margin_sse_js",
}


def ak_func(name):
    """按旧名或新名返回 akshare 函数：先试新名，再试旧名。"""
    import akshare as ak

    names = [RENAMES.get(name, name), name]
    errors = []
    for n in dict.fromkeys(names):
        try:
            return getattr(ak, n)
        except AttributeError as exc:
            errors.append(str(exc)[:80])
    raise AttributeError(f"akshare 没有 {' / '.join(dict.fromkeys(names))}: {'; '.join(errors)}")


def call(name, *args, **kw):
    return ak_func(name)(*args, **kw)
