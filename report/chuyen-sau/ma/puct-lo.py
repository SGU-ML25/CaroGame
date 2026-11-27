def select_actions(agents: Sequence[PUCTAgent], games: Sequence[CaroGame]) -> list[int]:
    """Ghép lượt suy luận giữa các ván dùng cùng mạng và ngân sách cố định."""
    matches = list(zip(agents, games, strict=True))
    if not matches:
        return []
    if len(matches) == 1:
        agent, game = matches[0]
        return [agent.select_action(game, agent.limits)]
    if len(set(agents)) != len(agents):
        raise ValueError("Mỗi ván cần tác tử PUCT riêng để giữ RNG và thống kê.")
    if any(a.model is not agents[0].model or a.limits.time_ms is not None for a in agents):
        raise ValueError("PUCT ghép lô cần dùng chung mạng và không đặt giới hạn thời gian.")
    active = [(i, a._search(g, a.limits), None) for i, (a, g) in enumerate(matches)]
    moves = [-1] * len(matches)
    while active:
        waiting, states, masks = [], [], []
        for i, search, result in active:
            try:
                xs, ms = search.send(result)
            except StopIteration as finished:
                moves[i] = finished.value
                continue
            start = len(states)
            states.extend(xs)
            masks.extend(ms)
            waiting.append((i, search, slice(start, len(states))))
        if not waiting:
            break
        probs, values = agents[0]._evaluate(states, masks)
        active = [(i, search, (probs[span], values[span])) for i, search, span in waiting]
    return moves
