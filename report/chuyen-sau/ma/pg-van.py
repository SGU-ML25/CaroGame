def play_iteration(
    model: CaroResNet, opp_model: CaroResNet | None, n_games: int, p_latest: float, temperature: float,
    opening_plies: int, device: torch.device, rng: random.Random,
) -> dict:
    """Chơi n_games ván song song. Trả các mẫu (x, mask, action, z) của những nước được học + thống kê."""
    games = [CaroGame() for _ in range(n_games)]
    self_play = [opp_model is None or rng.random() < p_latest for _ in range(n_games)]
    learner = [i % 2 for i in range(n_games)]            # bên học xen kẽ X/O
    for g in games:
        random_opening(g, opening_plies, rng)
    traj: list[list[tuple]] = [[] for _ in games]
    active = [i for i, g in enumerate(games) if not g.is_terminal()]

    model.eval()
    if opp_model is not None:
        opp_model.eval()
    while active:
        mine = [i for i in active if self_play[i] or games[i].side == learner[i]]
        mine_set = set(mine)
        theirs = [i for i in active if i not in mine_set]
        for idxs, net, record in ((mine, model, True), (theirs, opp_model, False)):
            if not idxs:
                continue
            xs, ms, acts = policy_step(net, [games[i] for i in idxs], temperature, device)
            for k, i in enumerate(idxs):
                if record:
                    traj[i].append((xs[k], ms[k], acts[k], games[i].side))
                games[i].step(acts[k])
        active = [i for i in active if not games[i].is_terminal()]

    xs, ms, acts, zs = [], [], [], []
    wins = draws = losses = 0                              # bên học trong các ván với checkpoint cũ
    for i, g in enumerate(games):
        for x, m, a, mover in traj[i]:
            xs.append(x)
            ms.append(m)
            acts.append(a)
            zs.append(0.0 if g.winner is None else (1.0 if g.winner == mover else -1.0))
        if not self_play[i]:
            if g.winner is None:
                draws += 1
            elif g.winner == learner[i]:
                wins += 1
            else:
                losses += 1
    return {
        "x": np.stack(xs), "mask": np.stack(ms), "action": np.asarray(acts, dtype=np.int64),
        "z": np.asarray(zs, dtype=np.float32),
        "n_self": sum(self_play), "wins": wins, "draws": draws, "losses": losses,
        "plies": sum(len(g.history) for g in games),
    }
