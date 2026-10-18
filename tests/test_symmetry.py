import random

import naive

from caro.engine.bitboard import CELLS, CENTER, has_five, winning_squares
from caro.engine.symmetry import COUNT, INVERSE, PERMS, canonical_key, transform_action, transform_bb


def test_perms_are_distinct_bijections():
    assert COUNT == 8
    assert all(sorted(p) == list(range(CELLS)) for p in PERMS)
    assert len(set(PERMS)) == 8
    assert all(transform_action(CENTER, k) == CENTER for k in range(COUNT))


def test_inverse():
    for k in range(COUNT):
        for i in range(CELLS):
            assert transform_action(transform_action(i, k), INVERSE[k]) == i


def test_rules_are_invariant():
    rng = random.Random(2)
    for _ in range(40):
        x, o = naive.random_position(rng, 0.4)
        for k in range(COUNT):
            tx, to = transform_bb(x, k), transform_bb(o, k)
            assert tx.bit_count() == x.bit_count()
            assert has_five(tx) == has_five(x)
            assert winning_squares(tx, to) == transform_bb(winning_squares(x, o), k)


def test_canonical_key_same_for_all_symmetric_positions():
    rng = random.Random(4)
    for _ in range(20):
        x, o = naive.random_position(rng, 0.2)
        keys = {canonical_key(transform_bb(x, k), transform_bb(o, k), 0) for k in range(COUNT)}
        assert len(keys) == 1
        assert canonical_key(x, o, 0) != canonical_key(x, o, 1)
