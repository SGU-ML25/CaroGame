"""Bộ giải VCF (victory by continuous fours): thắng ép bằng chuỗi nước tạo tứ liên tiếp.

Bên tấn công chỉ được đi nước tạo ít nhất một ô thắng ngay. Bên thủ, khi chưa tự thắng được, buộc phải
chặn đúng ô đó; mọi nước khác thua ngay ở lượt sau. Vì vậy lời giải tìm được là chắc chắn (sound). Bộ giải
không xét chuỗi ép bằng ba (VCT), nên "không có VCF" không có nghĩa là không có đường thắng.

Độ dài tính bằng số nước của bên tấn công: thắng ngay là 1, đòn kép rồi thắng là 2.
"""

from __future__ import annotations

from caro.engine.bitboard import (
    FULL,
    LINES,
    WIN,
    exactly,
    iter_bits,
    window_counts,
    winning_squares,
)

MAX_ATTACKS = 10


def lsb(b: int) -> int:
    return (b & -b).bit_length() - 1


def four_moves(me: int, opp: int) -> int:
    """Ô trống mà đặt vào thì tạo thêm cửa sổ sống 4 quân (nằm trong cửa sổ sống đúng 3 quân)."""
    empty = ~(me | opp) & FULL
    res = 0
    for d, start in LINES:
        w3 = exactly(3, *window_counts(me, opp, d, start))
        if w3:
            for j in range(WIN):
                res |= (w3 << (j * d)) & empty
    return res


def _search(me: int, opp: int, attacks: int, memo: dict) -> tuple[int, ...] | None:
    """Chuỗi nước (tấn công, thủ, tấn công, …) thắng trong tối đa `attacks` nước tấn công."""
    wins = winning_squares(me, opp)
    if wins:
        return (lsb(wins),)
    if attacks <= 1:
        return None
    key = (me, opp)
    failed = memo.get(key)
    if failed is not None and failed >= attacks:
        return None
    candidates = four_moves(me, opp)
    threat = winning_squares(opp, me)
    if threat:
        if threat & (threat - 1):            # đối thủ có từ hai ô thắng: không thể vừa chặn vừa tấn công
            candidates = 0
        else:
            candidates &= threat
    for m in iter_bits(candidates):
        line = _after_attack(me | 1 << m, opp, m, attacks, memo)
        if line is not None:
            return line
    memo[key] = attacks
    return None


def _after_attack(me: int, opp: int, m: int, attacks: int, memo: dict) -> tuple[int, ...] | None:
    """`me` đã gồm nước tấn công m. Trả chuỗi thắng bắt đầu bằng m, hoặc None."""
    if winning_squares(opp, me):             # bên thủ thắng trước
        return None
    wins = winning_squares(me, opp)
    if not wins:
        return None
    block = lsb(wins)
    if wins & (wins - 1):                    # hai ô thắng trở lên: chặn một, thắng ở ô còn lại
        return (m, block, lsb(wins & (wins - 1)))
    rest = _search(me, opp | 1 << block, attacks - 1, memo)
    return None if rest is None else (m, block, *rest)


def solve(me: int, opp: int, max_attacks: int = MAX_ATTACKS) -> tuple[int, ...] | None:
    """Chuỗi VCF ngắn nhất của bên đang đi (tăng dần độ sâu), hoặc None."""
    wins = winning_squares(me, opp)
    if wins:
        return (lsb(wins),)
    if not four_moves(me, opp):
        return None
    memo: dict = {}
    if _search(me, opp, max_attacks, memo) is None:
        return None
    for attacks in range(2, max_attacks + 1):
        line = _search(me, opp, attacks, {})
        if line is not None:
            return line
    raise AssertionError("tìm thấy VCF nhưng không tái lập được theo độ sâu")


def has_vcf(me: int, opp: int, max_attacks: int = MAX_ATTACKS) -> bool:
    return _search(me, opp, max_attacks, {}) is not None


def attacks_in(line: tuple[int, ...]) -> int:
    return (len(line) + 1) // 2


def winning_first_moves(me: int, opp: int, max_attacks: int = MAX_ATTACKS) -> int:
    """Mọi nước mở đầu cho thắng ép trong phạm vi VCF (gồm cả nước thắng ngay)."""
    wins = winning_squares(me, opp)
    result = wins
    candidates = four_moves(me, opp) & ~wins
    threat = winning_squares(opp, me)
    if threat:
        candidates = 0 if threat & (threat - 1) else candidates & threat
    memo: dict = {}
    for m in iter_bits(candidates):
        if _after_attack(me | 1 << m, opp, m, max_attacks, memo) is not None:
            result |= 1 << m
    return result


def safe_moves(me: int, opp: int, max_attacks: int = MAX_ATTACKS) -> int:
    """Nước mà sau đó đối thủ (đến lượt) không có VCF; nước thắng ngay luôn an toàn."""
    empty = ~(me | opp) & FULL
    result = winning_squares(me, opp)
    for m in iter_bits(empty & ~result):
        if not has_vcf(opp, me | 1 << m, max_attacks):
            result |= 1 << m
    return result
