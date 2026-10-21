import time

import pytest
import yaml

from caro.agents import REGISTRY, Agent, BaseAgent, Deadline, Limits, agent_class, make_agent
from caro.agents.random_agent import RandomAgent
from caro.config import CONFIG_ROOT, ConfigError, load_config, resolve_agent
from caro.engine import CaroGame

AGENT_FILES = sorted(CONFIG_ROOT.glob("agents/*.yaml"))
IMPLEMENTED = [p for p in AGENT_FILES if yaml.safe_load(p.read_text(encoding="utf-8"))["kind"] in REGISTRY]


def play(agent_x, agent_o, seed_x=1, seed_o=2):
    agent_x.reset(seed_x)
    agent_o.reset(seed_o)
    g = CaroGame()
    while not g.is_terminal():
        agent = (agent_x, agent_o)[g.side]
        a = agent.select_action(g, agent.limits)
        assert a in g.legal_actions()
        g.step(a)
    return g


# ---- Limits / Deadline -----------------------------------------------------
def test_limits_from_budget_files():
    a = Limits.from_dict(load_config("budgets/mode_a_time", ["seed=1"])["budget"])
    b = Limits.from_dict(load_config("budgets/mode_b_fixed", ["seed=1"])["budget"])
    assert a == Limits(mode="time", time_ms=1000)
    assert b == Limits(mode="fixed", max_depth=4, max_sims=800)
    assert Limits.from_dict(None) == Limits()


@pytest.mark.parametrize("data", [
    {"mode": "fast"},
    {"mode": "time"},
    {"max_sims": 0},
    {"max_depth": 2.5},
    {"max_depth": True},
    {"max_sim": 100},
])
def test_limits_rejects_bad_budget(data):
    with pytest.raises(ValueError):
        Limits.from_dict(data)


def test_deadline():
    assert not Limits(max_depth=3).deadline().expired()
    assert Limits().deadline().remaining_ms() is None
    d = Deadline(time_ms=20)
    assert not d.expired() and 0 < d.remaining_ms() <= 20
    time.sleep(0.03)
    assert d.expired() and d.remaining_ms() == 0 and d.elapsed_ms() >= 20
    assert Deadline(time_ms=50, margin_ms=60).expired()


# ---- REGISTRY / make_agent -------------------------------------------------
@pytest.mark.parametrize("path", IMPLEMENTED, ids=lambda p: p.stem)
def test_make_agent_from_every_implemented_config(path):
    ref = f"agents/{path.stem}"
    agent = make_agent(ref)
    assert isinstance(agent, Agent)
    assert agent.name == resolve_agent(ref)["id"]
    assert isinstance(agent.limits, Limits)


def test_make_agent_from_experiment_config():
    cfg = load_config("experiments/smoke")
    entry = next(a for a in cfg["agents"] if a["kind"] == "random")
    agent = make_agent(entry)
    assert isinstance(agent, RandomAgent) and agent.name == entry["id"]
    assert agent.limits == Limits.from_dict(cfg["budget"])


def test_make_agent_budget_override():
    agent = make_agent({"from": "agents/l1_random", "id": "R", "budget": {"mode": "time", "time_ms": 50}})
    assert agent.limits == Limits(mode="time", time_ms=50)


def test_make_agent_errors():
    with pytest.raises(ConfigError, match="chưa được cài"):
        agent_class("khong_co")
    with pytest.raises(ConfigError, match="không nhận tham số"):
        make_agent({"from": "agents/l1_random", "params": {"depth": 3}})
    with pytest.raises(ConfigError, match="budget"):
        make_agent({"from": "agents/l1_random", "budget": {"mode": "time"}})


def test_registry_entries_import():
    for kind in REGISTRY:
        assert issubclass(agent_class(kind), BaseAgent)


# ---- Cấp 1 — Random --------------------------------------------------------
def test_random_plays_legal_moves_to_the_end():
    for seed in range(20):
        g = play(make_agent("agents/l1_random"), make_agent("agents/l1_random"), seed, seed + 100)
        assert g.result() in (-1, 0, 1)


def test_random_same_seed_same_game():
    def moves(seed):
        return play(RandomAgent("a"), RandomAgent("b"), seed, seed + 1).history

    assert moves(7) == moves(7)
    assert moves(7) != moves(8)


def test_reset_restarts_sequence():
    agent, g = RandomAgent(), CaroGame()
    agent.reset(3)
    first = [agent.select_action(g, agent.limits) for _ in range(5)]
    agent.reset(3)
    assert [agent.select_action(g, agent.limits) for _ in range(5)] == first
    assert agent.last_stats() == {}
