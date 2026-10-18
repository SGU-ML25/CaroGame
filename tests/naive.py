"""Bản tham chiếu duyệt mảng 2D — chậm nhưng hiển nhiên đúng, dùng để đối chiếu với bitboard."""

import random

from caro.engine.bitboard import CELLS, CENTER, DIRECTIONS, WEIGHT, WIN, N, inside


def on(x: int, r: int, c: int) -> bool:
    return inside(r) and inside(c) and bool((x >> (r * N + c)) & 1)


def windows():
    """Mọi cửa sổ 5 ô nằm trọn trong bàn: danh sách chỉ số ô."""
    for r in range(N):
        for c in range(N):
            for dr, dc in DIRECTIONS:
                if inside(r + (WIN - 1) * dr) and inside(c + (WIN - 1) * dc):
                    yield [(r + k * dr) * N + c + k * dc for k in range(WIN)]


WINDOWS = list(windows())


def has_five(x: int) -> bool:
    return any(all((x >> i) & 1 for i in w) for w in WINDOWS)


def winning_squares(me: int, opp: int) -> int:
    """Ô trống i mà đặt vào sẽ tạo một dãy 5 ĐI QUA i (kể cả khi bàn đã có dãy 5 ở chỗ khác)."""
    occ = me | opp
    res = 0
    for w in WINDOWS:
        empties = [i for i in w if not (me >> i) & 1]
        if len(empties) == 1 and not (occ >> empties[0]) & 1:
            res |= 1 << empties[0]
    return res


def candidates(me: int, opp: int, radius: int = 2) -> int:
    occ = me | opp
    if occ == 0:
        return 1 << CENTER
    res = 0
    for i in range(CELLS):
        if (occ >> i) & 1:
            continue
        r, c = divmod(i, N)
        if any(on(occ, r + dr, c + dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)):
            res |= 1 << i
    return res


def side_score(me: int, opp: int) -> int:
    s = 0
    for w in WINDOWS:
        if any((opp >> i) & 1 for i in w):
            continue
        k = sum((me >> i) & 1 for i in w)
        if 1 <= k < WIN:
            s += WEIGHT[k]
    return s


def random_position(rng: random.Random, density: float) -> tuple[int, int]:
    """Hai bitboard rời nhau, mỗi ô có quân với xác suất ~density."""
    x = o = 0
    for i in range(CELLS):
        u = rng.random()
        if u < density / 2:
            x |= 1 << i
        elif u < density:
            o |= 1 << i
    return x, o
