cdef int64_t nm(self, u128 me, u128 opp, uint64_t key, int side, int depth,
                int64_t alpha, int64_t beta, int ply) noexcept nogil:
    cdef unordered_map[uint64_t, TTE].iterator it
    cdef TTE e
    cdef int mv[CELLS]
    cdef int n, i, m, best_m, k, flag
    cdef u128 cand, win_sqs, opp_win, forced
    cdef int64_t orig_alpha, best, score
    cdef u128 new_opp
    cdef uint64_t new_key

    self.nodes += 1
    self.pvlen[ply] = 0

    if has_five_c(opp):
        return -(WIN_SCORE - ply)

    if self.use_tt:
        it = self.tt.find(key)
        if it != self.tt.end():
            e = deref(it).second
            if e.depth >= depth:
                if e.flag == EXACT:
                    if e.move >= 0:
                        self.pv[ply][0] = e.move
                        self.pvlen[ply] = 1
                    return e.value
                elif e.flag == LOWER:
                    alpha = max(alpha, e.value)
                elif e.flag == UPPER:
                    beta = min(beta, e.value)
                if self.pruning and alpha >= beta:
                    if e.move >= 0:
                        self.pv[ply][0] = e.move
                        self.pvlen[ply] = 1
                    return e.value

    if depth == 0 or self.expired():
        return self.eval_leaf(me, opp)

    cand = candidates_c(me, opp, self.radius)
    n = bits_of(cand, mv)
    if n == 0:
        return self.eval_leaf(me, opp)

    if self.forced_moves:
        win_sqs = winning_squares_c(me, opp)
        forced = cand & win_sqs
        if not forced:
            opp_win = winning_squares_c(opp, me)
            forced = cand & opp_win
        if forced:
            n = bits_of(forced, mv)

    self.order_moves(mv, n, me, opp, key, ply)

    orig_alpha = alpha
    best = -INF
    best_m = mv[0]

    for i in range(n):
        m = mv[i]
        new_opp = me | ((<u128>1) << m)
        new_key = key ^ ZOB[side][m] ^ SIDEKEY
        score = -self.nm(opp, new_opp, new_key, side ^ 1, depth - 1, -beta, -alpha, ply + 1)

        if score > best:
            best = score
            best_m = m
            self.pv[ply][0] = m
            for k in range(self.pvlen[ply + 1]):
                self.pv[ply][k + 1] = self.pv[ply + 1][k]
            self.pvlen[ply] = 1 + self.pvlen[ply + 1]
        if score > alpha:
            alpha = score
        if self.pruning and alpha >= beta:
            if ply < KILLERN and m != self.killer[ply][0]:
                self.killer[ply][1] = self.killer[ply][0]
                self.killer[ply][0] = m
            self.hist[m] += <int64_t>depth * depth
            self.hist_has[m] = True
            break
        if self.expired():
            break

    if self.use_tt and not self.expired():
        if orig_alpha < best and best < beta:
            flag = EXACT
        elif best >= beta:
            flag = LOWER
        else:
            flag = UPPER
        self.tt_put(key, best, depth, flag, best_m)

    return best
