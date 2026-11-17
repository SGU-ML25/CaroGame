"""Lõi Cython (caro/engine/_native.pyx) phải cho đúng kết quả của bản Python.

Bỏ qua nếu chưa build: uv run python setup_native.py build_ext --inplace
"""

import random

import pytest

from caro.agents.alphabeta import AlphaBetaAgent
from caro.agents.base import Limits
from caro.engine import CaroGame
from caro.engine import bitboard as bb

nat = pytest.importorskip("caro.engine._native", reason="chưa build lõi native")


def _random_boards(seed: int, count: int):
    rng = random.Random(seed)
    for t in range(count):
        me = opp = 0
        for c in rng.sample(range(bb.CELLS), rng.randint(0, 60)):
            if rng.random() < 0.5:
                me |= 1 << c
            else:
                opp |= 1 << c
        if t % 5 == 0:                       # chèn dãy 3–5 quân để có đe doạ
            r, c0 = rng.randrange(11), rng.randrange(7)
            for k in range(rng.randint(3, 5)):
                me |= 1 << (r * 11 + c0 + k)
            opp &= ~me
        yield me, opp


def test_bitboard_primitives_match_python():
    for me, opp in _random_boards(1, 1500):
        assert nat.has_five(me) == bb.has_five(me)
        assert nat.winning_squares(me, opp) == bb.winning_squares(me, opp)
        assert nat.side_score(me, opp, bb.WEIGHT) == bb.side_score(me, opp, bb.WEIGHT)
        assert nat.evaluate(me, opp, bb.WEIGHT) == bb.evaluate(me, opp, bb.WEIGHT)
        for radius in (1, 2):
            assert nat.candidates(me, opp, radius) == bb.candidates(me, opp, radius)


def test_empty_and_full_edges():
    assert nat.candidates(0, 0, 2) == bb.candidates(0, 0, 2) == 1 << bb.CENTER
    assert nat.has_five(bb.FULL) and bb.has_five(bb.FULL)
    assert nat.winning_squares(0, 0) == 0


CONFIGS = [
    {},
    {"pruning": False, "tt": False, "move_ordering": False},
    {"move_ordering": False},
    {"tt": False},
    {"forced_moves": False},
    {"eval_weights": "no_defense"},
    {"radius": 1},
]


def _pair(cfg):
    py = AlphaBetaAgent(backend="python", tt_size_mb=8, **cfg)
    na = AlphaBetaAgent(backend="native", tt_size_mb=8, **cfg)
    py.reset(0)
    na.reset(0)
    return py, na


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: ",".join(f"{k}={v}" for k, v in c.items()) or "default")
def test_select_action_matches_python(cfg):
    """Cùng nước đi, điểm, PV và số nút trên các ván ngẫu nhiên (TT, history giữ lại qua các nước)."""
    depth = 2 if cfg.get("pruning") is False else 3
    limits = Limits(max_depth=depth)
    for seed in range(3):
        py, na = _pair(cfg)
        rng = random.Random(seed)
        game = CaroGame()
        for _ in range(rng.randint(0, 4)):
            game.step(rng.choice(game.legal_actions()))
        for _ in range(8):
            if game.is_terminal():
                break
            a, b = py.select_action(game, limits), na.select_action(game, limits)
            sp, sn = py.last_stats(), na.last_stats()
            assert a == b
            keys = ("score", "pv", "depth", "nodes")
            assert [sp.get(k) for k in keys] == [sn.get(k) for k in keys]   # thoát sớm: cả hai cùng không có
            game.step(a if rng.random() < 0.7 else rng.choice(game.legal_actions()))


def test_scored_moves_matches_python_and_pruning_is_exact():
    limits = Limits(max_depth=3)
    py, na = _pair({})
    rng = random.Random(5)
    for _ in range(6):
        game = CaroGame()
        for _ in range(rng.randint(2, 14)):
            if game.is_terminal():
                break
            game.step(rng.choice(game.legal_actions()))
        if game.is_terminal():
            continue
        full = py.scored_moves(game, limits, top_k=3, prune_root=False)
        pruned_py = py.scored_moves(game, limits, top_k=3)
        pruned_na = na.scored_moves(game, limits, top_k=3)
        assert [s for _, s in pruned_py] == [s for _, s in full]      # điểm top-3 không đổi
        assert pruned_py == pruned_na                                   # hai backend trùng nhau


def test_backend_validation_and_auto():
    with pytest.raises(ValueError):
        AlphaBetaAgent(backend="gpu")
    assert AlphaBetaAgent(backend="auto").backend == "native"
    assert AlphaBetaAgent().backend == "python"          # mặc định không đổi hành vi cũ


def test_native_is_faster_on_a_real_position():
    import time

    limits = Limits(max_depth=3)
    py, na = _pair({})
    game = CaroGame()
    for m in (60, 61, 49, 72, 50):
        game.step(m)
    t0 = time.perf_counter()
    py.scored_moves(game, limits, top_k=3)
    t1 = time.perf_counter()
    na.scored_moves(game, limits, top_k=3)
    t2 = time.perf_counter()
    assert (t2 - t1) < (t1 - t0)


@pytest.mark.parametrize("cfg", [{}, {"pruning": False, "tt": False, "move_ordering": False}], ids=["default", "A1-no-pruning"])
def test_scored_moves_without_pruning_full_list_matches(cfg):
    """Không cắt tỉa ở gốc (và với pruning=False của A1): toàn bộ danh sách (nước, điểm) trùng nhau."""
    limits = Limits(max_depth=2)
    py, na = _pair(cfg)
    rng = random.Random(11)
    checked = 0
    while checked < 8:
        game = CaroGame()
        for _ in range(rng.randint(2, 14)):
            if game.is_terminal():
                break
            game.step(rng.choice(game.legal_actions()))
        if game.is_terminal():
            continue
        assert py.scored_moves(game, limits, top_k=1000, prune_root=False) == \
            na.scored_moves(game, limits, top_k=1000, prune_root=False)
        checked += 1
