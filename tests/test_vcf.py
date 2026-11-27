"""Kiểm bộ giải VCF dùng trong báo cáo: lời giải phải thắng trước MỌI đáp trả của bên thủ."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "report" / "scripts" / "chuyen_sau"))

import vcf

from caro.engine import CaroGame
from caro.engine.bitboard import (
    FULL,
    candidates,
    has_five,
    index,
    iter_bits,
    winning_squares,
)


def stones(*cells):
    mask = 0
    for r, c in cells:
        mask |= 1 << index(r, c)
    return mask


def forced(me: int, opp: int, attacks: int) -> bool:
    """Kiểm vét cạn: có nước tấn công mà mọi đáp trả của bên thủ vẫn dẫn tới thắng."""
    if winning_squares(me, opp):
        return True
    if attacks <= 1:
        return False
    line = vcf.solve(me, opp, attacks)
    if line is None:
        return False
    me |= 1 << line[0]
    if has_five(me):
        return True
    for reply in iter_bits(~(me | opp) & FULL):
        after = opp | 1 << reply
        if has_five(after):
            return False
        if not winning_squares(me, after) and not forced(me, after, attacks - 1):
            return False
    return True


def test_win_in_one():
    me = stones((5, 1), (5, 2), (5, 3), (5, 4))

    assert vcf.attacks_in(vcf.solve(me, 0)) == 1


def test_open_three_becomes_double_threat():
    me = stones((5, 3), (5, 4), (5, 5))

    line = vcf.solve(me, 0)

    assert vcf.attacks_in(line) == 2
    assert forced(me, 0, 2)


def test_defender_five_first_refutes_attack():
    me = stones((5, 3), (5, 4), (5, 5))
    opp = stones((0, 0), (0, 1), (0, 2), (0, 3))      # bên thủ có ô thắng, bên công không chặn bằng tứ được

    assert vcf.solve(me, opp) is None


def test_no_threat_no_solution():
    assert vcf.solve(stones((5, 5)), stones((4, 4))) is None


@pytest.mark.parametrize("seed", range(4))
def test_solutions_are_sound_on_random_positions(seed):
    rng = random.Random(seed)
    checked = 0
    while checked < 5:
        game = CaroGame()
        for _ in range(rng.randrange(10, 36)):
            if game.is_terminal():
                break
            game.step(rng.choice(list(iter_bits(candidates(*game.me_opp(), 1)))))
        if game.is_terminal():
            continue
        me, opp = game.me_opp()
        line = vcf.solve(me, opp, 4)
        if line is None or vcf.attacks_in(line) < 2:
            continue
        assert forced(me, opp, vcf.attacks_in(line))
        assert vcf.winning_first_moves(me, opp, 4) >> line[0] & 1
        checked += 1
