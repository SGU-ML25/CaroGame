def _negamax(
    self, me: int, opp: int, key: int, side: int,
    depth: int, alpha: int, beta: int, ply: int, deadline: Any,
) -> tuple[int, list[int]]:
    if self._nat is not None:
        self._nat.set_deadline(getattr(deadline, "end", None))
        result = self._nat.negamax(me, opp, key, side, depth, alpha, beta, ply)
        self._nodes += self._nat.take_nodes()
        return result
    self._nodes += 1

    # opp vừa đi - kiểm tra opp thắng
    if has_five(opp):
        return -(WIN_SCORE - ply), []

    # Tra TT
    if self.use_tt:
        entry = self._tt.get(key)
        if entry is not None:
            tt_val, tt_depth, tt_flag, tt_move = entry
            if tt_depth >= depth:
                if tt_flag == EXACT:
                    return tt_val, [tt_move] if tt_move >= 0 else []
                elif tt_flag == LOWER:
                    alpha = max(alpha, tt_val)
                elif tt_flag == UPPER:
                    beta = min(beta, tt_val)
                if self.pruning and alpha >= beta:
                    return tt_val, [tt_move] if tt_move >= 0 else []

    # Node lá
    if depth == 0 or deadline.expired():
        return self._eval_leaf(me, opp), []

    # Tập ứng viên
    cand_bb = bb_candidates(me, opp, self.radius)
    moves = list(iter_bits(cand_bb))
    if not moves:
        return self._eval_leaf(me, opp), []

    # Nước bắt buộc
    if self.forced_moves:
        forced = self._forced(me, opp, moves)
        if forced:
            moves = forced

    ordered = self._order_moves(moves, me, opp, key, ply)

    orig_alpha = alpha
    best_score = -INF
    best_pv: list[int] = []
    best_m = ordered[0]

    for m in ordered:
        new_opp  = me | (1 << m)
        new_key  = key ^ ZOBRIST[side][m] ^ SIDE_KEY
        new_side = side ^ 1

        score, pv = self._negamax(
            opp, new_opp, new_key, new_side,
            depth - 1, -beta, -alpha, ply + 1, deadline
        )
        score = -score

        if score > best_score:
            best_score = score
            best_m = m
            best_pv = [m] + pv
        alpha = max(alpha, score)
        if self.pruning and alpha >= beta:
            self._update_killer(m, ply)
            self._history[m] = self._history.get(m, 0) + depth * depth
            break
        if deadline.expired():
            break

    # Ghi TT
    if self.use_tt and not deadline.expired():
        flag = EXACT if orig_alpha < best_score < beta else \
               LOWER if best_score >= beta else UPPER
        self._tt.put(key, best_score, depth, flag, best_m)

    return best_score, best_pv
