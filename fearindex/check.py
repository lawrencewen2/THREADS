"""每日运行后的健康检查：数据缺失、数据过期、异常跳变、历史值漂移。"""

import pandas as pd

from . import config

STALE_DAYS = 5
JUMP = 25
DRIFT = 1.0


def run_checks(series, errors, index_df, previous_index=None, method_changed=False):
    warnings, failures = [], []
    for key, msg in errors.items():
        warnings.append(f"{key} 拉取失败，沿用缓存（{msg[:80]}）")
    missing = [k for k in config.SERIES if k not in series]
    if missing:
        warnings.append("缺少序列：" + ", ".join(missing))

    last = index_df.index[-1]
    for key, s in series.items():
        lag = (last - s.index[-1]).days
        if lag > STALE_DAYS:
            warnings.append(f"{key} 已 {lag} 天未更新（最后 {s.index[-1].date()}）")

    if len(index_df) > 1:
        diff = (index_df.iloc[-1] - index_df.iloc[-2]).abs()
        for col, d in diff.items():
            if pd.notna(d) and d > JUMP:
                warnings.append(f"{col} 单日变动 {d:.1f} 分，请核对数据")

    # 已发布的历史值不应改变（除最近几天可能因数据补录而修正）
    if method_changed:
        warnings.append("算法参数已更新，本次跳过历史值漂移检查")
    elif previous_index is not None and len(previous_index):
        cutoff = last - pd.Timedelta(days=10)
        common = previous_index.index.intersection(index_df.index)
        common = common[common < cutoff]
        if len(common):
            delta = (index_df.loc[common, "global"] - previous_index.loc[common, "global"]).abs()
            if delta.max() > DRIFT:
                failures.append(f"历史值漂移：{delta.idxmax().date()} 的全球分变化 {delta.max():.1f}")

    if index_df["global"].iloc[-1:].isna().all():
        failures.append("最新一天全球分为空")
    return dict(warnings=warnings, failures=failures, ok=not failures)
