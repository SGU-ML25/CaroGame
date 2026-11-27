def _threat_info(me: int, opp: int) -> tuple[int, bool, dict[int, int], list[tuple]]:
    """Tính một lượt cho cả nút những gì cần biết về đe doạ của 'me'.

    Trả (wins, has_four, gain, counts):
        wins     -- các ô thắng ngay của me, giống winning_squares(me, opp)
        has_four -- me có cửa sổ sống đúng 4 quân không
        gain     -- ô m -> các ô thắng ngay MỚI nếu me đi m. Mỗi cửa sổ sống đúng 3 quân có đúng
                    2 ô trống a, b: đi a thì b thành ô thắng và ngược lại
        counts   -- kết quả window_counts của 4 hướng, để dùng lại khi chấm điểm vị trí

    Nhờ đó winning_squares(me | 1 << m, opp) = (wins bỏ m) | gain[m] với mọi ô trống m,
    không phải đếm lại cửa sổ cho từng ứng viên.
    """
    empty = ~(me | opp) & FULL
    wins = 0
    has_four = False
    gain: dict[int, int] = {}
    counts = []
    for d, start in LINES:
        cnt = window_counts(me, opp, d, start)
        counts.append(cnt)
        w4 = exactly(4, *cnt)
        if w4:
            has_four = True
            for j in range(WIN):
                wins |= (w4 << (j * d)) & empty
        for s in iter_bits(exactly(3, *cnt)):
            a, b = (i for i in range(s, s + WIN * d, d) if not (me >> i) & 1)
            gain[a] = gain.get(a, 0) | (1 << b)
            gain[b] = gain.get(b, 0) | (1 << a)
    return wins, has_four, gain, counts


def _position_terms(
    counts: list[tuple], weight: tuple[int, ...],
) -> tuple[int, list[tuple[int, int, int]]]:
    """Chuẩn bị tính side_score sau một nước: trả (điểm hiện tại, [(hướng, cửa sổ k quân, chênh lệch)])."""
    base = 0
    terms = []
    for i, cnt in enumerate(counts):
        for k in range(WIN):
            windows = exactly(k, *cnt)
            if not windows:
                continue
            # side_score không tính cửa sổ 0 quân và 5 quân
            before = weight[k] if k else 0
            after = weight[k + 1] if k + 1 < WIN else 0
            base += before * windows.bit_count()
            if after != before:
                terms.append((i, windows, after - before))
    return base, terms
