"""空仓开关触发后，同一批股票不能每天重复列为卖出（2026-11-19 清单 bug）。"""
from astock import guards
from astock.live_slots import held_action


def test_first_cash_day_sells_once():
    assert held_action(False, False, False) == "空仓卖出"
    assert held_action(True, False, False) == "到期卖出"


def test_later_cash_days_do_not_repeat_sells():
    assert held_action(False, False, True) == "已空仓"
    assert held_action(True, False, True) == "已空仓"


def test_reopen_buys_back_unexpired_only():
    assert held_action(False, True, True) == "买回"
    assert held_action(True, True, True) == "已空仓"


def test_normal_days_unchanged():
    assert held_action(False, True, False) == "继续持有"
    assert held_action(True, True, False) == "到期卖出"


def test_block_buys_also_blocks_buyback():
    res = dict(picks=[dict(action="买回", selected=True), dict(action="继续持有", selected=True)])
    guards.block_buys(res, "x")
    assert [r["action"] for r in res["picks"]] == ["观察", "继续持有"]
    assert res["n_selected"] == 1
