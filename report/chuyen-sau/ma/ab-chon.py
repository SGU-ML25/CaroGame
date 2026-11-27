def select_action(self, game: CaroGame, limits: Limits) -> int:
    me, opp = game.me_opp()
    key = game.key
    side = game.side          # 0=X, 1=O - dùng để tính Zobrist đúng
    deadline = limits.deadline(margin_ms=10.0)

    self._nodes = 0
    self._killer = [[-1, -1] for _ in range(64)]
    self._history.clear()
    if self._nat is not None:
        self._nat.reset_search()
    t0 = time.perf_counter()

    cand_bb = bb_candidates(me, opp, self.radius)
    moves = list(iter_bits(cand_bb))
    if not moves:
        return -1
    if len(moves) == 1:
        return moves[0]

    best_move = moves[0]
    best_score = -INF
    self._pv: list[int] = [best_move]
    depth_reached = 1

    max_depth = limits.max_depth if limits.max_depth is not None else 6

    if self.iterative_deepening:
        for depth in range(1, max_depth + 1):
            if deadline.expired():
                break
            score, pv = self._negamax_root(
                me, opp, key, side, moves, depth, deadline
            )
            if deadline.expired() and depth > 1:
                break
            best_score = score
            best_move = pv[0] if pv else best_move
            self._pv = pv
            depth_reached = depth
            if best_move in moves:
                moves.remove(best_move)
                moves.insert(0, best_move)
            if best_score >= WIN_SCORE - 100:
                break
    else:
        best_score, pv = self._negamax_root(
            me, opp, key, side, moves, max_depth, deadline
        )
        best_move = pv[0] if pv else best_move
        self._pv = pv
        depth_reached = max_depth

    elapsed = (time.perf_counter() - t0) * 1000
    self._stats = {
        "nodes":   self._nodes,
        "depth":   depth_reached,
        "time_ms": round(elapsed, 1),
        "pv":      self._pv,
        "score":   best_score,
    }
    return best_move
