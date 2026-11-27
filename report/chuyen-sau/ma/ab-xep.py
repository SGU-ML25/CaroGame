def _forced(self, me: int, opp: int, moves: list[int]) -> list[int]:
    win_sqs = winning_squares(me, opp)
    wins = [m for m in moves if (1 << m) & win_sqs]
    if wins:
        return wins
    opp_win = winning_squares(opp, me)
    blocks = [m for m in moves if (1 << m) & opp_win]
    if blocks:
        return blocks
    return []


def _order_moves(
    self, moves: list[int], me: int, opp: int, key: int, ply: int,
) -> list[int]:
    if not self.move_ordering or len(moves) == 1:
        return moves

    tt_move = -1
    if self.use_tt:
        entry = self._tt.get(key)
        if entry is not None:
            tt_move = entry[3]

    # Thông tin chiến thuật tính một lượt cho cả nút; thứ tự thu được giống hệt việc đi thử
    # từng ứng viên rồi gọi has_five / winning_squares / side_score (tests/test_alphabeta.py).
    my_wins, my_four, my_gain, my_counts = _threat_info(me, opp)
    opp_wins, _, opp_gain, _ = _threat_info(opp, me)
    killer = self._killer[ply] if ply < len(self._killer) else (-1, -1)
    history = self._history
    position = None              # chỉ tính khi có ứng viên cần điểm vị trí

    scores: list[tuple[int, int]] = []
    for m in moves:
        bit_m = 1 << m
        if m == tt_move:
            s = 10_000_000
        elif bit_m & my_wins:
            s = 9_500_000
        elif (my_wins | my_gain.get(m, 0)).bit_count() >= 2:
            s = 9_000_000
        elif my_four or m in my_gain:
            s = 8_000_000
        elif ((opp_wins & ~bit_m) | opp_gain.get(m, 0)).bit_count() >= 2:
            s = 7_000_000
        elif m == killer[0]:
            s = 6_000_000
        elif m == killer[1]:
            s = 5_000_000
        elif m in history:
            s = 1_000_000 + history[m]
        else:
            if position is None:
                position = _position_terms(my_counts, self._weight)
            at = _WINDOWS_AT[m]
            s = position[0]
            for i, windows, delta in position[1]:
                s += delta * (windows & at[i]).bit_count()
        scores.append((s, m))

    scores.sort(reverse=True)
    return [m for _, m in scores]
