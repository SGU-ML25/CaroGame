import random

import pytest

from caro.engine import CELLS, CaroGame, IllegalMoveError, O, X
from caro.engine.bitboard import N, index
from caro.engine.zobrist import full_key

# Bàn đầy không có 5 liền: ô (r, c) là X khi ((r + 2c) // 2) chẵn → 61 X, 60 O
DRAW_X = [index(r, c) for r in range(N) for c in range(N) if ((r + 2 * c) // 2) % 2 == 0]
DRAW_O = [index(r, c) for r in range(N) for c in range(N) if ((r + 2 * c) // 2) % 2 == 1]
DRAW_MOVES = [m for pair in zip(DRAW_X, DRAW_O + [None], strict=True) for m in pair if m is not None]


def play_random(rng, game=None, stop_at=None):
    g = game or CaroGame()
    while not g.is_terminal() and (stop_at is None or len(g.history) < stop_at):
        g.step(rng.choice(g.legal_actions()))
    return g


def snapshot(g):
    return (tuple(g.bb), g.side, tuple(g.history), g.key, g.winner)


def test_new_game():
    g = CaroGame()
    assert g.side == X and g.history == [] and g.key == 0
    assert g.legal_actions() == list(range(CELLS))
    assert g.result() is None and not g.is_terminal() and g.termination() is None


def test_x_wins_horizontal():
    g = CaroGame()
    xs = [index(5, c) for c in range(3, 8)]
    os_ = [index(0, c) for c in range(4)]
    for a, b in zip(xs[:4], os_, strict=True):
        assert g.step(a) == (0.0, False, {"winner": None})
        g.step(b)
    reward, done, info = g.step(xs[4])
    assert (reward, done, info["winner"]) == (1.0, True, X)
    assert g.result() == 1 and g.termination() == "five"
    assert g.legal_actions() == [] and g.legal_mask() == 0


def test_o_wins_gives_minus_one():
    g = CaroGame.from_moves([0, 60, 2, 61, 4, 62, 6, 63, 120])
    reward, done, _ = g.step(64)
    assert (reward, done, g.result(), g.winner) == (1.0, True, -1, O)


def test_draw_full_board():
    assert len(DRAW_X) == 61 and len(DRAW_O) == 60
    g = CaroGame()
    for a in DRAW_MOVES[:-1]:
        assert g.step(a)[1] is False
    reward, done, _ = g.step(DRAW_MOVES[-1])
    assert (reward, done, g.result(), g.termination()) == (0.0, True, 0, "full")
    assert g.legal_actions() == []


@pytest.mark.parametrize("bad", [-1, CELLS, 999, 1.5, "a", None, True])
def test_illegal_values_leave_state_unchanged(bad):
    g = CaroGame.from_moves([60, 61])
    before = snapshot(g)
    with pytest.raises(IllegalMoveError):
        g.step(bad)
    assert snapshot(g) == before


def test_occupied_and_after_terminal():
    g = CaroGame.from_moves([60])
    with pytest.raises(IllegalMoveError):
        g.step(60)
    g = CaroGame.from_moves([0, 60, 1, 61, 2, 62, 3, 63, 4])
    before = snapshot(g)
    with pytest.raises(IllegalMoveError, match="kết thúc"):
        g.step(100)
    assert snapshot(g) == before


def test_accepts_numpy_like_ints():
    np = pytest.importorskip("numpy")
    g = CaroGame()
    g.step(np.int64(60))
    assert g.history == [60] and isinstance(g.history[0], int)


def test_undo_everything_returns_to_empty():
    rng = random.Random(3)
    for _ in range(30):
        g = play_random(rng)
        n = len(g.history)
        assert [g.undo() for _ in range(n)] and g.history == []
        assert g.bb == [0, 0] and g.side == X and g.key == 0 and g.winner is None
    with pytest.raises(IllegalMoveError):
        CaroGame().undo()


def test_undo_after_win_reopens_game():
    g = CaroGame.from_moves([0, 60, 1, 61, 2, 62, 3, 63, 4])
    assert g.undo() == 4
    assert g.winner is None and g.result() is None and 4 in g.legal_actions()


def test_incremental_zobrist_matches_recomputed():
    rng = random.Random(5)
    for _ in range(20):
        g = CaroGame()
        while not g.is_terminal():
            g.step(rng.choice(g.legal_actions()))
            assert g.key == full_key(g.bb[X], g.bb[O], g.side)


def test_same_position_different_order_same_key():
    a = CaroGame.from_moves([60, 61, 70, 71])
    b = CaroGame.from_moves([70, 71, 60, 61])
    assert a.key == b.key and a.bb == b.bb


def test_clone_is_independent():
    rng = random.Random(9)
    g = play_random(rng, stop_at=20)
    before = snapshot(g)
    c = g.clone()
    play_random(rng, c)
    c.undo()
    assert snapshot(g) == before
    assert snapshot(g.clone()) == before


def test_random_games_always_legal_and_consistent():
    rng = random.Random(11)
    for _ in range(100):
        g = play_random(rng)
        assert g.result() in (-1, 0, 1)
        assert (g.bb[X] & g.bb[O]) == 0
        nx, no = g.bb[X].bit_count(), g.bb[O].bit_count()
        assert nx - no in (0, 1) and nx + no == len(g.history)
        assert len(set(g.history)) == len(g.history)


def test_me_opp_and_render():
    g = CaroGame.from_moves([index(0, 0), index(10, 10)])
    me, opp = g.me_opp()
    assert me == g.bb[X] and opp == g.bb[O]
    rows = g.render().splitlines()
    assert len(rows) == N and rows[0].startswith("X .") and rows[-1].endswith(". O")
    assert g.last_move == index(10, 10) and "2 nước" in repr(g)
