def pick_random_bit(b: int, count: int, rng: random.Random) -> int:
    """Chọn ngẫu nhiên 1 bit 1 từ bitmask b có đúng count bit 1."""
    n = rng.randrange(count)
    for _ in range(n):
        b &= b - 1
    return (b & -b).bit_length() - 1


def first_winning_square(me: int, opp: int) -> int | None:
    """Trả về một ô thắng ngay nếu có, hoặc None."""
    empty = ~(me | opp) & FULL
    for d, start in LINES:
        w4 = exactly(4, *window_counts(me, opp, d, start))
        if w4:
            for j in range(WIN):
                sq = (w4 << (j * d)) & empty
                if sq:
                    return (sq & -sq).bit_length() - 1
    return None


def rollout_fast(
    bb_x: int,
    bb_o: int,
    side: int,
    rng: random.Random,
    rollout_mode: str,
    rollout_radius: int,
) -> int:
    """Mô phỏng đến hết ván. Trả kết quả theo góc nhìn X: +1 X thắng, -1 O thắng, 0 hoà.

    Kiểm tra kết thúc: chỉ gọi has_five trên bitboard của người vừa đi (mục 06.2).
    """
    bb = [bb_x, bb_o]
    cur_side = side
    while True:
        m_me = bb[cur_side]
        m_opp = bb[cur_side ^ 1]
        empty = ~(m_me | m_opp) & FULL
        if not empty:
            return 0

        move = None
        if rollout_mode == "light":
            # 1. Thắng ngay
            move = first_winning_square(m_me, m_opp)
            if move is None:
                # 2. Chặn thua ngay
                move = first_winning_square(m_opp, m_me)

        if move is None:
            # 3. Ngẫu nhiên trong bán kính rollout_radius
            cands = bb_candidates(m_me, m_opp, rollout_radius)
            cnt = cands.bit_count()
            if not cnt:
                cands = empty
                cnt = empty.bit_count()
            move = pick_random_bit(cands, cnt, rng)

        bb[cur_side] |= 1 << move
        if has_five(bb[cur_side]):
            return 1 if cur_side == 0 else -1
        cur_side ^= 1
