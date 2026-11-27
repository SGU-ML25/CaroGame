"""Kiểm bộ giải VCT dùng để phân tích ván: cây chứng minh phải thắng trước MỌI đáp trả của bên thủ."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "report" / "scripts" / "chuyen_sau"))

import vcf
import vct

from caro.engine import CaroGame
from caro.engine.bitboard import FULL, candidates, has_five, index, iter_bits, winning_squares


def stones(*cells):
    mask = 0
    for r, c in cells:
        mask |= 1 << index(r, c)
    return mask


def check(me: int, opp: int, tree: dict) -> None:
    """Đi theo cây chứng minh: với mọi đáp trả hợp lệ, bên công vẫn thắng."""
    if "vcf" in tree:
        assert vct.line_still_wins(me, opp, tuple(tree["vcf"]))
        return
    me |= 1 << tree["move"]
    if has_five(me):
        return
    assert not winning_squares(opp, me)
    for reply in iter_bits(~(me | opp) & FULL):
        after = opp | 1 << reply
        assert not has_five(after)
        if reply in tree["replies"]:
            check(me, after, tree["replies"][reply])
        else:
            assert vcf.solve(me, after) is not None, reply


def test_double_open_three_wins_although_no_vcf():
    me = stones((5, 3), (5, 4), (3, 5), (4, 5))
    opp = stones((0, 0), (0, 2), (10, 10), (10, 8))

    assert vcf.solve(me, opp) is None
    tree = vct.solve(me, opp, depth=1)

    assert tree is not None and tree["depth"] == 1
    check(me, opp, tree)


def test_defender_four_refutes_the_threat():
    me = stones((5, 3), (5, 4), (3, 5), (4, 5))
    opp = stones((0, 0), (0, 1), (0, 2), (0, 3))        # bên thủ đã có tứ: phải chặn trước, không dựng được đôi ba

    tree = vct.solve(me, opp, depth=1)

    assert tree is None or tree["move"] == index(0, 4)


def test_quiet_position_has_no_forced_win():
    assert vct.solve(stones((5, 5)), stones((4, 4)), depth=2) is None


@pytest.mark.parametrize("seed", range(3))
def test_trees_are_sound_on_random_positions(seed):
    rng = random.Random(seed)
    checked = 0
    for _ in range(400):
        if checked >= 3:
            break
        game = CaroGame()
        for _ in range(rng.randrange(8, 24)):
            if game.is_terminal():
                break
            game.step(rng.choice(list(iter_bits(candidates(*game.me_opp(), 1)))))
        if game.is_terminal() or vcf.solve(*game.me_opp()) is not None:
            continue
        tree = vct.solve(*game.me_opp(), depth=1)
        if tree is None:
            continue
        check(*game.me_opp(), tree)
        checked += 1
    assert checked >= 1
