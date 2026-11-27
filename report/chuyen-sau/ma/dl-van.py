def build_policy_and_sample(scored_moves, top_k, temperature, rng):
    """Tạo policy target trên 121 ô và lấy mẫu một nước từ Top-K."""
    if not scored_moves:
        return None, None

    scored_moves = scored_moves[:top_k] if top_k > 0 else scored_moves
    moves = np.asarray([int(move) for move, _ in scored_moves], dtype=np.int64)
    logits = np.asarray([float(score) for _, score in scored_moves], dtype=np.float64)

    temperature = max(float(temperature), 1e-5)
    logits = logits / temperature
    logits -= np.max(logits)

    weights = np.exp(logits)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.full(len(moves), 1.0 / len(moves), dtype=np.float64)
    else:
        probs = weights / total

    policy = np.zeros(121, dtype=np.float32)
    policy[moves] = probs.astype(np.float32)

    selected_index = int(rng.choice(len(moves), p=probs))
    return policy, int(moves[selected_index])


def generate_single_game(game_id, openings, config):
    """Sinh dữ liệu của một ván độc lập."""
    seed = int(config.get("seed", 42)) + int(game_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    datagen_cfg = config.get("datagen", config)
    # Giữ nguyên cả `params` (vd. backend: auto) thay vì chỉ lấy tên file cấu hình
    expert_cfg = datagen_cfg.get("expert") or "agents/l3_alphabeta"
    expert_depth = int(datagen_cfg.get("expert_depth", 3))
    top_k = int(datagen_cfg.get("top_k", 3))
    prune_root = bool(datagen_cfg.get("prune_root", True))   # False: search đầy đủ từng nước ở gốc
    temperature = float(datagen_cfg.get("temperature", 0.5))

    expert = make_agent(expert_cfg)
    expert.reset(seed)

    game = CaroGame()

    if openings:
        opening = random.choice(openings)
        if isinstance(opening, list):
            for move in opening:
                if game.is_terminal():
                    break
                game.step(int(move))

    limits = Limits(max_depth=expert_depth)
    game_history = []

    while not game.is_terminal():
        state_tensor = board_to_tensor(game)
        scored = expert.scored_moves(game, limits=limits, top_k=top_k, prune_root=prune_root)

        if not scored:
            legal_moves = game.legal_actions()
            if not legal_moves:
                break
            action = int(legal_moves[0])
            policy_target = np.zeros(121, dtype=np.float32)
            policy_target[action] = 1.0
        else:
            policy_target, action = build_policy_and_sample(
                scored, top_k, temperature, rng
            )

        if action is None or action < 0 or action >= 121:
            break

        game_history.append((state_tensor.numpy(), policy_target))
        game.step(action)

    winner = game.winner
    # Nhãn TUYỆT ĐỐI (+1 nếu X thắng). train_sl đổi sang góc nhìn người đi khi nạp dữ liệu.
    value_target = 1.0 if winner == X else -1.0 if winner is not None else 0.0

    records = []
    for state, policy in game_history:
        records.append({
            "state": state.tolist(),
            "policy": policy.tolist(),
            "value": value_target,
        })

    return int(game_id), records
