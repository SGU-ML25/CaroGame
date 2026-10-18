"""8 phép đối xứng của bàn vuông (nhóm D4): augmentation ×8 và khoá chuẩn hoá để loại trùng."""

from __future__ import annotations

from caro.engine.bitboard import CELLS, N, iter_bits
from caro.engine.zobrist import full_key

_M = N - 1
_MAPS = (
    lambda r, c: (r, c),             # 0 giữ nguyên
    lambda r, c: (c, _M - r),        # 1 xoay 90° theo chiều kim đồng hồ
    lambda r, c: (_M - r, _M - c),   # 2 xoay 180°
    lambda r, c: (_M - c, r),        # 3 xoay 270°
    lambda r, c: (r, _M - c),        # 4 lật trái–phải
    lambda r, c: (_M - r, c),        # 5 lật trên–dưới
    lambda r, c: (c, r),             # 6 lật qua đường chéo chính
    lambda r, c: (_M - c, _M - r),   # 7 lật qua đường chéo phụ
)
COUNT = len(_MAPS)

# PERMS[k][i] = ô mà ô i chuyển tới dưới phép k
PERMS: tuple[tuple[int, ...], ...] = tuple(
    tuple(f(*divmod(i, N))[0] * N + f(*divmod(i, N))[1] for i in range(CELLS)) for f in _MAPS
)
# INVERSE[k] = phép đưa kết quả của phép k về lại như cũ
INVERSE: tuple[int, ...] = tuple(
    next(j for j in range(COUNT) if all(PERMS[j][PERMS[k][i]] == i for i in range(CELLS))) for k in range(COUNT)
)


def transform_action(a: int, k: int) -> int:
    return PERMS[k][a]


def transform_bb(b: int, k: int) -> int:
    if k == 0:
        return b
    p = PERMS[k]
    out = 0
    for i in iter_bits(b):
        out |= 1 << p[i]
    return out


def canonical_key(x: int, o: int, side: int) -> int:
    """Khoá chuẩn hoá = min khoá Zobrist qua 8 phép đối xứng (mục 07.2)."""
    return min(full_key(transform_bb(x, k), transform_bb(o, k), side) for k in range(COUNT))
