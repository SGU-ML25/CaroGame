"""Cấp 7 — PUCTAgent (mạng khởi tạo ngẫu nhiên, không cần checkpoint)."""

from __future__ import annotations

import pytest
import torch

from caro.agents import make_agent
from caro.agents.base import Limits
from caro.agents.puct import PUCTAgent, PUCTNode, select_child, terminal_value
from caro.engine import CaroGame

FIXED = Limits(mode="fixed", max_sims=300)


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    from caro.nn.model import CaroResNet
    return CaroResNet({"num_channels": 16, "num_blocks": 2})


def agent(net, **kw) -> PUCTAgent:
    a = PUCTAgent(model=net, **kw)
    a.reset(0)
    return a


def game_from(xs, os_):
    """Dựng thế cờ: X và O đi xen kẽ (X trước); tới lượt X nếu hai bên đã đi bằng nhau."""
    g = CaroGame()
    for i in range(max(len(xs), len(os_))):
        if i < len(xs):
            g.step(xs[i])
        if i < len(os_):
            g.step(os_[i])
    return g


def test_registry_and_config():
    a = make_agent({"id": "t", "level": 7, "kind": "puct", "limits": {}, "params": {"checkpoint": None, "batch_leaves": 4}})
    assert isinstance(a, PUCTAgent) and a.batch_leaves == 4


def test_terminal_value_is_loss_for_side_to_move():
    g = game_from([0, 1, 2, 3, 4], [11, 12, 13, 14])         # X vừa thắng, tới lượt O
    assert g.winner == 0 and terminal_value(g) == -1.0
    assert terminal_value(CaroGame()) == 0.0


def test_select_child_prefers_prior_then_value():
    root = PUCTNode(None, 1.0)
    root.N = 10
    a, b = PUCTNode(1, 0.7), PUCTNode(2, 0.3)
    root.children = [a, b]
    assert select_child(root, 1.5) is a                         # chưa thăm: theo prior
    a.N, a.W = 5, 5.0                                           # bên đi ở a thắng → tệ cho cha
    assert select_child(root, 1.5) is b


def test_virtual_loss_discourages_same_child():
    root = PUCTNode(None, 1.0)
    root.N = 10
    a, b = PUCTNode(1, 0.5), PUCTNode(2, 0.4)
    root.children = [a, b]
    assert select_child(root, 1.5) is a
    a.vl = 3
    assert select_child(root, 1.5) is b


def test_returns_legal_move_and_stats(net):
    g = CaroGame()
    g.step(60)
    a = agent(net)
    m = a.select_action(g, FIXED)
    assert m in g.legal_actions()
    st = a.last_stats()
    assert st["sims"] == 300 and 0 < sum(st["visits"].values()) <= 300
    assert -1 <= st["value"] <= 1 and st["pv"][0] == m
    assert a.last_pi is not None and abs(float(a.last_pi.sum()) - 1.0) < 1e-5


def test_takes_immediate_win(net):
    g = game_from([60, 61, 62, 63], [0, 11, 22, 33])            # X có 4 liên tiếp ở hàng 5, tới lượt X
    assert g.side == 0
    assert agent(net).select_action(g, FIXED) in (59, 64)


def test_blocks_open_four_threat(net):
    g = game_from([5, 20, 40, 100], [60, 61, 62, 63])           # O có 4 liên tiếp ở hàng 5, tới lượt X
    assert g.side == 0
    assert agent(net).select_action(g, Limits(mode="fixed", max_sims=600)) in (59, 64)


@pytest.mark.parametrize("batch", [1, 8])
def test_no_leftover_virtual_loss(net, batch, monkeypatch):
    g = CaroGame()
    g.step(60)
    a = agent(net, batch_leaves=batch)
    seen = []
    orig = PUCTAgent._finish

    def spy(self, root, *args, **kw):
        stack = [root]
        while stack:
            n = stack.pop()
            seen.append(n.vl)
            stack.extend(n.children)
        return orig(self, root, *args, **kw)

    monkeypatch.setattr(PUCTAgent, "_finish", spy)
    a.select_action(g, Limits(mode="fixed", max_sims=100))
    assert seen and set(seen) == {0}


def test_same_seed_same_move(net):
    g = CaroGame()
    g.step(60)
    moves = {agent(net, dirichlet={"alpha": 0.3, "epsilon": 0.25}, temperature=1.0).select_action(g, FIXED)
             for _ in range(2)}
    assert len(moves) == 1


def test_mix_and_noise_modes_run(net):
    g = CaroGame()
    g.step(60)
    a = agent(net, leaf_eval="mix", dirichlet={"alpha": 0.3, "epsilon": 0.25}, temperature=1.0)
    assert a.select_action(g, Limits(mode="fixed", max_sims=40)) in g.legal_actions()


def test_time_budget_and_default_budget(net):
    g = CaroGame()
    g.step(60)
    a = agent(net)
    assert a.select_action(g, Limits(mode="time", time_ms=100)) in g.legal_actions()
    assert a.select_action(g, Limits()) in g.legal_actions()
    assert a.last_stats()["sims"] == 200


def test_terminal_state_returns_minus_one(net):
    g = game_from([0, 1, 2, 3, 4], [11, 12, 13, 14])
    assert agent(net).select_action(g, FIXED) == -1


def test_rejects_bad_params(net):
    with pytest.raises(ValueError):
        PUCTAgent(model=net, leaf_eval="x")
    with pytest.raises(ValueError):
        PUCTAgent(model=net, batch_leaves=0)
