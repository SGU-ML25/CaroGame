def select_action(self, game: CaroGame, limits: Limits) -> int:
    """Chọn nước đi theo thứ tự ưu tiên P1->P7; trả chỉ số ô (0-120)."""
    me, opp = game.me_opp()

    # Tập ứng viên: ô trống trong bán kính Chebyshev quanh các quân đã đặt
    legal_moves = list(iter_bits(bb_candidates(me, opp, self.radius)))
    if not legal_moves:
        return -1

    pri = self.priorities

    def choose(move: int, reason: str) -> int:
        self._stats = {"reason": reason}
        return move

    # --- P1: Thắng ngay ---
    # Đặt vào ô trong winning_squares(me, opp) là thắng luôn
    if "P1" in pri:
        win_sqs = winning_squares(me, opp)
        for m in legal_moves:
            if (1 << m) & win_sqs:
                return choose(m, "P1_win")

    # --- P2: Chặn thua ---
    # Đối thủ có ô thắng ngay -> phải đặt vào đó để chặn
    opp_win_sqs = winning_squares(opp, me)
    if "P2" in pri:
        for m in legal_moves:
            if (1 << m) & opp_win_sqs:
                return choose(m, "P2_block_loss")

    # --- P3: Tạo đòn kép ---
    # Đặt tại m tạo ra >=2 ô thắng -> đối thủ không thể chặn hết trong 1 nước
    if "P3" in pri:
        for m in legal_moves:
            future_me = me | (1 << m)
            if popcount(winning_squares(future_me, opp)) >= 2:
                return choose(m, "P3_double_attack")

    # --- P4: Phá đòn kép của đối thủ ---
    if "P4" in pri:
        # TH1: Đối thủ hiện đã có sẵn >=2 ô thắng -> chặn một trong số đó
        if popcount(opp_win_sqs) >= 2:
            for m in legal_moves:
                if (1 << m) & opp_win_sqs:
                    return choose(m, "P4_block_double_attack")

        # TH2: Đối thủ sẽ có đòn kép nếu ta không chiếm m trước
        # me không đổi vì ta chưa đặt - chỉ xét "nếu opp đặt m thì sao"
        for m in legal_moves:
            future_opp = opp | (1 << m)
            if popcount(winning_squares(future_opp, me)) >= 2:
                return choose(m, "P4_block_double_attack")

    # --- P5: Tạo tứ ---
    # Đặt tại m tạo cửa sổ 4 quân sống; nhiều nước thì chọn điểm P6 cao nhất
    if "P5" in pri:
        p5_candidates = [
            m for m in legal_moves
            if _makes_four(me | (1 << m), opp)
        ]
        if p5_candidates:
            best = max(p5_candidates, key=lambda m: evaluate(me | (1 << m), opp, self._weight))
            return choose(best, "P5_four_attack")

    # --- P6: Đánh giá chung bằng evaluate ---
    # Chọn nước có side_score cao nhất theo eval_weights
    best_score = -float("inf")
    best_moves: list[int] = []
    if "P6" in pri or "P7" in pri:
        for m in legal_moves:
            score = evaluate(me | (1 << m), opp, self._weight)
            if score > best_score:
                best_score = score
                best_moves = [m]
            elif score == best_score:
                best_moves.append(m)

        if best_moves and len(best_moves) == 1 and "P6" in pri:
            return choose(best_moves[0], "P6_evaluate")

        # --- P7: Hoà điểm -> gần tâm -> ngẫu nhiên theo seed (self.rng) ---
        # Tâm bàn 11x11: hàng 5, cột 5 -> index 60
        if best_moves and "P7" in pri:
            center_row, center_col = 5, 5
            min_dist = min(
                abs((m // 11) - center_row) + abs((m % 11) - center_col)
                for m in best_moves
            )
            closest = [
                m for m in best_moves
                if abs((m // 11) - center_row) + abs((m % 11) - center_col) == min_dist
            ]
            return choose(self.rng.choice(closest), "P7_center_random")

    # Fallback: chưa đủ luật được bật để phủ hết -> ngẫu nhiên
    return choose(self.rng.choice(legal_moves), "Fallback_random")
