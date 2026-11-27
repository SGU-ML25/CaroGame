"""Bộ giải VCT (victory by continuous threats): thắng ép bằng chuỗi đe doạ, gồm cả tứ và ba hở.

Một nước của bên công là *đe doạ* nếu sau nó, giả sử bên thủ bỏ lượt, bên công có VCF. Bên thủ được thử **mọi** ô
trống: ô nào không phá được VCF thì thua theo bộ giải VCF; các ô còn lại (kể cả nước phản công, nước thắng của bên
thủ) phải bị bác bỏ bằng đệ quy. Vì vét hết đáp trả, lời giải tìm được là chắc chắn. Độ sâu `depth` là số nước đe doạ
tối đa trước khi tới VCF; không tìm thấy trong độ sâu đó không chứng minh thế an toàn.
"""

from __future__ import annotations

import vcf

from caro.engine.bitboard import FULL, candidates, has_five, iter_bits, winning_squares


def line_still_wins(me: int, opp: int, line: tuple[int, ...]) -> bool:
    """Chuỗi VCF `line` (tấn công, chặn, …) còn thắng được trên thế mới không (kiểm lại từng bước)."""
    for k in range(0, len(line), 2):
        attack = line[k]
        if (me | opp) >> attack & 1:
            return False
        me |= 1 << attack
        if has_five(me):
            return True
        if winning_squares(opp, me):
            return False
        wins = winning_squares(me, opp)
        if not wins:
            return False
        if wins & (wins - 1):
            return True
        opp |= wins
    return False


def defences(me: int, opp: int, line: tuple[int, ...]) -> list[int] | None:
    """Mọi đáp trả của bên thủ mà bên công (đến lượt sau đó) không còn VCF. None nếu bên thủ thắng được ngay."""
    if winning_squares(opp, me):
        return None
    forced = winning_squares(me, opp)
    if forced:                                          # tứ: chỉ ô chặn mới không thua ngay
        if forced & (forced - 1):
            return []
        block = vcf.lsb(forced)
        return [] if vcf.has_vcf(me, opp | 1 << block) else [block]
    result = []
    for reply in iter_bits(~(me | opp) & FULL):
        after = opp | 1 << reply
        if line_still_wins(me, after, line):
            continue
        if not vcf.has_vcf(me, after):
            result.append(reply)
    return result


def _search(me: int, opp: int, depth: int, memo: dict) -> dict | None:
    """Cây chứng minh {'move', 'vcf'} hoặc {'move', 'replies': {đáp trả: cây con}}; None nếu không tìm được."""
    line = vcf.solve(me, opp)
    if line is not None:
        return {"move": line[0], "vcf": list(line)}
    if depth <= 0:
        return None
    key = (me, opp)
    if memo.get(key, -1) >= depth:
        return None
    pool = candidates(me, opp, 2)
    threat = winning_squares(opp, me)
    if threat:
        pool = 0 if threat & (threat - 1) else threat
    threats = []
    for move in iter_bits(pool):
        after = me | 1 << move
        if winning_squares(opp, after):
            continue
        follow = vcf.solve(after, opp)
        if follow is None:
            continue
        replies = defences(after, opp, follow)
        if replies is not None:
            threats.append((len(replies), move, replies))
    for _, move, replies in sorted(threats):            # thử đe doạ có ít đáp trả trước
        after = me | 1 << move
        subtree = {}
        for reply in replies:
            child = _search(after, opp | 1 << reply, depth - 1, memo)
            if child is None:
                break
            subtree[reply] = child
        else:
            return {"move": move, "replies": subtree}
    memo[key] = depth
    return None


def solve(me: int, opp: int, depth: int = 2) -> dict | None:
    """Đường thắng ép ngắn nhất tìm được (thử độ sâu 0, 1, …, `depth`)."""
    for limit in range(depth + 1):
        tree = _search(me, opp, limit, {})
        if tree is not None:
            return {**tree, "depth": limit}
    return None


def main_line(tree: dict) -> list[int]:
    """Một biến đại diện: mỗi tầng chọn đáp trả dẫn tới nhánh dài nhất."""
    if "vcf" in tree:
        return list(tree["vcf"])
    if not tree["replies"]:
        return [tree["move"]]
    reply, child = max(tree["replies"].items(), key=lambda kv: len(main_line(kv[1])))
    return [tree["move"], reply, *main_line(child)]
