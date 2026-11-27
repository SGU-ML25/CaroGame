def scored_moves(
    self, game: CaroGame, limits: Limits, top_k: int = 5, prune_root: bool = True,
) -> list[tuple[int, int]]:
    """Trả [(move, score), ...] top-k nước tốt nhất theo góc nhìn bên đang đi.

    Dùng để sinh dữ liệu expert cho cấp 5 (mục 05.1-05.2).

    prune_root: sau khi đã có k nước, gọi search con với cửa sổ (-INF, -theta + 1) với theta là điểm hạng k hiện có.
    Điểm là số nguyên nên nước nào có điểm > theta vẫn chính xác; nước còn lại chỉ nhận một cận và nằm ngoài
    top-k nên không ảnh hưởng kết quả. False = search đầy đủ từng nước như bản gốc (dùng để đối chiếu).
    """
    me, opp = game.me_opp()
    key = game.key
    side = game.side
    deadline = limits.deadline(margin_ms=10.0)

    cand_bb = bb_candidates(me, opp, self.radius)
    moves = list(iter_bits(cand_bb))
    if not moves:
        return []

    self._nodes = 0
    max_depth = limits.max_depth if limits.max_depth is not None else 4
    results: list[tuple[int, int]] = []
    prune = prune_root and top_k > 0
    kth = None                      # điểm hạng top_k hiện có

    for m in moves:
        new_opp = me | (1 << m)
        new_key = key ^ ZOBRIST[side][m] ^ SIDE_KEY
        new_side = side ^ 1
        beta = INF if kth is None else -kth + 1
        score, _ = self._negamax(
            opp, new_opp, new_key, new_side,
            max_depth - 1, -INF, beta, 1, deadline
        )
        results.append((m, -score))
        if prune and len(results) >= top_k:
            kth = sorted((s for _, s in results), reverse=True)[top_k - 1]

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
