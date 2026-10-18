"""Bitboard cho Caro 11×11 (tài liệu thiết kế, Phần 01).

Bàn cờ của một bên là một số nguyên 121 bit: ô (r, c) ↔ bit i = r·11 + c (layout stride 11).
Bốn hướng là bốn độ dịch: ngang 1, dọc 11, chéo ↘ 12, chéo ↙ 10. Vì không có cột đệm, phép
dịch có thể "lọt" từ mép phải hàng này sang mép trái hàng kế bên — mọi kết quả theo dãy 5 ô
phải AND với mặt nạ START của hướng đó, mọi phép dịch ngang 1 ô phải AND với NOT_COL0 / NOT_COL10.

Mọi mặt nạ được sinh bằng vòng lặp lúc import, không chép tay hằng số.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

N = 11                       # cạnh bàn
CELLS = N * N                # 121 ô
WIN = 5                      # ≥ 5 quân liên tiếp là thắng, không luật chặn 2 đầu
FULL = (1 << CELLS) - 1
CENTER = (N // 2) * N + N // 2


def index(r: int, c: int) -> int:
    return r * N + c


def rc(i: int) -> tuple[int, int]:
    return divmod(i, N)


def bit(r: int, c: int) -> int:
    return 1 << (r * N + c)


def inside(v: int) -> bool:
    return 0 <= v < N


def mask_of(pred: Callable[[int, int], bool]) -> int:
    m = 0
    for r in range(N):
        for c in range(N):
            if pred(r, c):
                m |= bit(r, c)
    return m


# (dr, dc) cho 4 hướng: ngang →, dọc ↓, chéo ↘, chéo ↙
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

# (độ dịch d, START): START = các ô đầu của một dãy 5 nằm trọn trong bàn theo hướng đó
LINES: tuple[tuple[int, int], ...] = tuple(
    (dr * N + dc, mask_of(lambda r, c, dr=dr, dc=dc: inside(r + (WIN - 1) * dr) and inside(c + (WIN - 1) * dc)))
    for dr, dc in DIRECTIONS
)

NOT_COL0 = FULL & ~mask_of(lambda r, c: c == 0)
NOT_COL10 = FULL & ~mask_of(lambda r, c: c == N - 1)


def popcount(b: int) -> int:
    return b.bit_count()


def iter_bits(b: int) -> Iterator[int]:
    """Duyệt chỉ số các bit 1 theo thứ tự tăng dần."""
    while b:
        low = b & -b                 # tách bit thấp nhất
        yield low.bit_length() - 1
        b ^= low


def has_five(x: int) -> bool:
    """Bàn x có ≥ 5 quân liên tiếp theo một hướng nào đó không (3 phép dịch mỗi hướng)."""
    for d, start in LINES:
        t = x & (x >> d)             # cặp 2
        t &= t >> (2 * d)            # dãy 4
        t &= x >> (4 * d)            # dãy 5
        if t & start:
            return True
    return False


def dilate(x: int) -> int:
    """Thêm 8 ô lân cận của mọi quân (vùng 3×3), không lọt qua mép."""
    v = x | (x << N) | (x >> N)
    h = v | ((v << 1) & NOT_COL0) | ((v >> 1) & NOT_COL10)
    return h & FULL


def candidates(me: int, opp: int, radius: int = 2) -> int:
    """Ô trống cách quân gần nhất ≤ radius (khoảng cách Chebyshev). Bàn trống → ô giữa."""
    occ = me | opp
    if occ == 0:
        return 1 << CENTER
    area = occ
    for _ in range(radius):
        area = dilate(area)          # mỗi vòng đã cắt về FULL nên không tích luỹ lỗi lọt hàng
    return area & ~occ & FULL


def window_counts(me: int, opp: int, d: int, start: int) -> tuple[int, int, int, int]:
    """Đếm song song số quân 'me' trong mọi cửa sổ 5 ô theo hướng d (bit-sliced).

    Trả (c0, c1, c2, free): bit i của c2c1c0 là số quân của cửa sổ bắt đầu ở ô i;
    free = các cửa sổ hợp lệ không chứa quân 'opp'.
    """
    free = start
    for k in range(WIN):
        free &= ~(opp >> (k * d))
    c0 = c1 = c2 = 0
    for k in range(WIN):
        p = (me >> (k * d)) & free
        carry = c0 & p
        c0 ^= p
        carry2 = c1 & carry
        c1 ^= carry
        c2 |= carry2                 # tối đa 5 = 101b, không tràn
    return c0, c1, c2, free


def exactly(k: int, c0: int, c1: int, c2: int, free: int) -> int:
    """Các cửa sổ (trong free) có đúng k quân."""
    m = free
    m &= c0 if k & 1 else ~c0
    m &= c1 if k & 2 else ~c1
    m &= c2 if k & 4 else ~c2
    return m


def winning_squares(me: int, opp: int) -> int:
    """Các ô trống mà 'me' đặt vào là có ≥ 5 liền ngay."""
    empty = ~(me | opp) & FULL
    res = 0
    for d, start in LINES:
        w4 = exactly(4, *window_counts(me, opp, d, start))
        for j in range(WIN):         # đưa bit ô đầu cửa sổ về đúng ô trống trong cửa sổ
            res |= (w4 << (j * d)) & empty
    return res


WEIGHT = (0, 1, 8, 64, 512)          # 0..4 quân trong một cửa sổ còn sống; 5 quân = đã thắng


def side_score(me: int, opp: int, weight: tuple[int, ...] = WEIGHT) -> int:
    s = 0
    for d, start in LINES:
        cnt = window_counts(me, opp, d, start)
        for k in range(1, WIN):
            s += weight[k] * exactly(k, *cnt).bit_count()
    return s


def evaluate(me: int, opp: int, weight: tuple[int, ...] = WEIGHT) -> int:
    """Điểm thế cờ theo góc nhìn 'me': cửa sổ sống của mình trừ của đối thủ."""
    return side_score(me, opp, weight) - side_score(opp, me, weight)
