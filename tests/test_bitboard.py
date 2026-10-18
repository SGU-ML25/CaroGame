import random

import naive
import pytest

from caro.engine.bitboard import (
    CELLS,
    CENTER,
    FULL,
    LINES,
    NOT_COL0,
    NOT_COL10,
    N,
    bit,
    candidates,
    dilate,
    evaluate,
    has_five,
    index,
    iter_bits,
    popcount,
    rc,
    side_score,
    winning_squares,
)

DENSITIES = (0.1, 0.25, 0.4, 0.6, 0.85)


def cells(*rcs):
    m = 0
    for r, c in rcs:
        m |= bit(r, c)
    return m


# ---- hằng số và tiện ích -------------------------------------------------

def test_constants():
    assert (N, CELLS, FULL.bit_length()) == (11, 121, 121)
    assert rc(CENTER) == (5, 5)
    assert [d for d, _ in LINES] == [1, 11, 12, 10]
    # số cửa sổ 5 ô mỗi hướng: 7·11 ngang, 11·7 dọc, 7·7 mỗi chéo
    assert [popcount(s) for _, s in LINES] == [77, 77, 49, 49]
    assert popcount(NOT_COL0) == popcount(NOT_COL10) == 110


def test_index_rc_bit_roundtrip():
    for i in range(CELLS):
        r, c = rc(i)
        assert index(r, c) == i and bit(r, c) == 1 << i


def test_iter_bits_ascending():
    rng = random.Random(1)
    for _ in range(200):
        b = rng.getrandbits(CELLS)
        idx = list(iter_bits(b))
        assert idx == [i for i in range(CELLS) if (b >> i) & 1]
        assert len(idx) == popcount(b)
    assert list(iter_bits(0)) == []


# ---- thắng ---------------------------------------------------------------

def test_every_five_line_wins_and_four_does_not():
    for w in naive.WINDOWS:              # mọi dãy 5 trên bàn, cả 4 hướng, sát mép và góc
        x = sum(1 << i for i in w)
        assert has_five(x), [rc(i) for i in w]
        for drop in w:                   # bỏ 1 quân → không còn thắng
            assert not has_five(x & ~(1 << drop))


@pytest.mark.parametrize("stones, wins", [
    ([(5, c) for c in range(2, 8)], True),                          # dãy 6 vẫn thắng (luật ≥ 5)
    ([(0, c) for c in range(11)], True),                             # cả hàng
    ([(2, 8), (2, 9), (2, 10), (3, 0), (3, 1)], False),              # chỉ số 30..34 liên tiếp nhưng lọt hàng
    ([(0, 8), (1, 9), (2, 10), (4, 0), (5, 1)], False),              # ↘ giả: 8,20,32,44,56
    ([(0, 2), (1, 1), (2, 0), (2, 10), (3, 9)], False),              # ↙ giả: 2,12,22,32,42
    ([(6, 10), (7, 0), (8, 1), (9, 2), (10, 3)], False),
])
def test_edge_cases(stones, wins):
    assert has_five(cells(*stones)) is wins


def test_blocked_both_ends_still_wins():
    x = cells(*[(4, c) for c in range(3, 8)])
    o = cells((4, 2), (4, 8))
    assert has_five(x) and not has_five(o)


@pytest.mark.parametrize("density", DENSITIES)
def test_has_five_matches_naive(density):
    rng = random.Random(int(density * 100))
    for _ in range(400):
        x, o = naive.random_position(rng, density)
        assert has_five(x) == naive.has_five(x)
        assert has_five(o) == naive.has_five(o)


# ---- giãn nở và ứng viên -------------------------------------------------

def test_dilate_single_cell_is_3x3_without_wrap():
    for i in range(CELLS):
        r, c = rc(i)
        expect = cells(*[(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                         if 0 <= r + dr < N and 0 <= c + dc < N])
        assert dilate(1 << i) == expect, (r, c)


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_candidates_match_naive(radius):
    rng = random.Random(radius)
    for _ in range(150):
        x, o = naive.random_position(rng, rng.choice((0.02, 0.05, 0.15, 0.4)))
        assert candidates(x, o, radius) == naive.candidates(x, o, radius)


def test_candidates_empty_board_is_center():
    assert candidates(0, 0) == 1 << CENTER


def test_candidates_radius2_from_right_edge_does_not_wrap():
    got = candidates(bit(3, 10), 0, 2)
    assert all(rc(i)[1] >= 8 for i in iter_bits(got))
    assert popcount(got) == 5 * 3 - 1


# ---- ô thắng ngay và đánh giá -------------------------------------------

@pytest.mark.parametrize("density", DENSITIES)
def test_winning_squares_matches_naive(density):
    rng = random.Random(1000 + int(density * 100))
    for _ in range(60):
        x, o = naive.random_position(rng, density)
        assert winning_squares(x, o) == naive.winning_squares(x, o)
        assert winning_squares(o, x) == naive.winning_squares(o, x)


@pytest.mark.parametrize("line, expect", [
    ("..XXXX..", {1, 6}),                # tứ mở → 2 ô
    ("OXXXX...", {5}),                   # tứ bị chặn
    ("XX.XX...", {2}),                   # tứ có lỗ
    ("XXX.....", set()),
])
def test_winning_squares_patterns(line, expect):
    x = cells(*[(5, c) for c, ch in enumerate(line) if ch == "X"])
    o = cells(*[(5, c) for c, ch in enumerate(line) if ch == "O"])
    assert {rc(i)[1] for i in iter_bits(winning_squares(x, o))} == expect


def test_evaluate_matches_naive_and_is_antisymmetric():
    rng = random.Random(7)
    for _ in range(80):
        x, o = naive.random_position(rng, rng.choice((0.05, 0.2, 0.4)))
        assert side_score(x, o) == naive.side_score(x, o)
        assert evaluate(x, o) == -evaluate(o, x)
    assert evaluate(0, 0) == 0


def test_open_three_beats_blocked_three():
    open3 = cells((5, 4), (5, 5), (5, 6))
    blocked3 = cells((5, 1), (5, 2), (5, 3))
    assert side_score(open3, 0) > side_score(blocked3, cells((5, 0)))
