"""Khoá băm Zobrist (mục 01.9): cập nhật bằng XOR khi đi / lùi nước."""

from __future__ import annotations

import random

from caro.engine.bitboard import CELLS, iter_bits

_rng = random.Random(2026)           # seed cố định → khoá giống nhau giữa các lần chạy và các máy
ZOBRIST: tuple[tuple[int, ...], tuple[int, ...]] = (
    tuple(_rng.getrandbits(64) for _ in range(CELLS)),
    tuple(_rng.getrandbits(64) for _ in range(CELLS)),
)
SIDE_KEY = _rng.getrandbits(64)      # bật khi O đến lượt


def full_key(x: int, o: int, side: int) -> int:
    """Tính khoá từ đầu — dùng để kiểm tra khoá cập nhật dần và cho các thế dựng sẵn."""
    k = SIDE_KEY if side else 0
    for i in iter_bits(x):
        k ^= ZOBRIST[0][i]
    for i in iter_bits(o):
        k ^= ZOBRIST[1][i]
    return k
