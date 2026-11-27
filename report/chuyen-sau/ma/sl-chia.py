def mover_relative_values(states, values):
    """Đổi nhãn value từ tuyệt đối (+1 nếu X thắng) sang góc nhìn người đang đi (+1 nếu người đang đi thắng).

    Cùng quy ước với đầu vào mạng (kênh 0 = quân người đi) và với `z` của train_pg, nên value head dùng được
    cho RL / PUCT / AlphaZero. Kênh 2 của `states` bằng 1 khi X đi, 0 khi O đi.
    """
    sign = np.where(states[:, 2, 0, 0] == 1, 1.0, -1.0)
    return (values * sign).astype(values.dtype)


def split_groups(game_ids, seed):
    """Chia 80/10/10 THEO VÁN: mỗi mẫu nhận 0 (train), 1 (val) hoặc 2 (test).

    Các nước của cùng một ván không rơi vào hai tập khác nhau. Script phân tích phải gọi đúng hàm này với
    đúng seed lúc huấn luyện, nếu không "test" sẽ chứa mẫu đã học.
    """
    rng = np.random.default_rng(int(seed))
    unique_games = rng.permutation(np.unique(game_ids))
    n_train, n_val = int(len(unique_games) * 0.8), int(len(unique_games) * 0.9)
    group = np.zeros(len(game_ids), dtype=np.int8)
    group[np.isin(game_ids, unique_games[n_train:n_val])] = 1
    group[np.isin(game_ids, unique_games[n_val:])] = 2
    return group
