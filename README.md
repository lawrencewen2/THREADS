# 全球市场恐慌指数

每天自动更新的 0–100 恐慌指数，覆盖美国、欧洲、日本、A股、港股。0 = 极度贪婪，100 = 极度恐慌。仅供个人判断市场情绪参考，不构成投资建议。

<!-- latest:start -->
**最新读数（2026-10-23 收盘）**

| 区域 | 恐慌指数 | 状态 | 5日变化 |
| --- | --- | --- | --- |
| 全球 | 67 | 恐慌 | +9.4 |
| 美国 | 62 | 恐慌 | +6.0 |
| 欧洲 | 79 | 恐慌 | +8.6 |
| 日本 | 82 | 极度恐慌 | +15.9 |
| A股 | — | — | — |
| 港股 | 63 | 恐慌 | +13.5 |

⚠️ us_hy_oas 拉取失败，沿用缓存（ReadTimeout: HTTPSConnectionPool(host='fred.stlouisfed.org', port=443): Read tim）；vstoxx 拉取失败，沿用缓存（ValueError: empty yahoo data for ^V2TX）；eu_hy_oas 拉取失败，沿用缓存（ReadTimeout: HTTPSConnectionPool(host='fred.stlouisfed.org', port=443): Read tim）；csi300 拉取失败，沿用缓存（ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed c）；a_turnover 拉取失败，沿用缓存（ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed c）；hstech 拉取失败，沿用缓存（ValueError: empty yahoo data for ^HSTECH）；缺少序列：us_hy_oas, vstoxx, eu_hy_oas, csi300, a_turnover, hstech

![近两年走势](docs/chart.svg)
<!-- latest:end -->

## 怎么算的

1. 每个区域 4–5 个成分：波动率指数、实际波动率、回撤、偏离均线、信用利差、避险资产等（完整清单见 `fearindex/config.py`）。
2. 每个成分换算成过去 756 个交易日（约 3 年）里的分位数 × 100，数值越高越恐慌。
3. 区域分 = 成分加权平均；全球分 = 区域按市值大致加权（美 45%、欧 20%、日 10%、A 15%、港 10%）× 90% + 全球公共成分（美债波动率、黄金、美债避险）× 10%。
4. 最终做 3 日指数平滑。

| 分数 | 状态 |
| --- | --- |
| 0–20 | 极度贪婪 |
| 20–40 | 贪婪 |
| 40–60 | 中性 |
| 60–80 | 恐慌 |
| 80–100 | 极度恐慌 |

## 文件

- `data/index.csv`：每日全球及各区域分数
- `data/components.csv`：各成分分位得分；`data/components_raw.csv`：成分原始值
- `data/latest.json`：最新读数与数据健康状态
- `docs/index.html`：交互式仪表盘；`docs/chart.svg`：近两年走势图
- `reports/backtest.md`：回测报告

## 运行

```bash
pip install -r requirements.txt
python scripts/run_daily.py          # 增量更新（首次会自动拉全部历史）
python scripts/run_backtest.py       # 生成回测报告
python -m pytest -q                  # 单元测试（合成数据）
```

GitHub Actions 每个交易日北京时间 07:30 自动运行 `daily-update`，失败时 GitHub 会发邮件提醒。

数据源：Yahoo Finance、FRED（ICE BofA 高收益债利差）、AkShare（东方财富、上交所、期权论坛 QVIX）。
