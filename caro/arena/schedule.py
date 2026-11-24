
def build_match_schedule(
    agents,
    openings,
    pairs_mode="round_robin",
    games_per_pair=200,
    swap_colors=True,
    seed=42,
    baselines=None,
):
    """Tạo lịch đấu xác định và có thể tái lập."""

    if not agents:
        raise ValueError("Danh sách agents không được rỗng.")

    if not openings:
        raise ValueError("Danh sách openings không được rỗng.")

    if not isinstance(games_per_pair, int) or games_per_pair <= 0:
        raise ValueError("games_per_pair phải là số nguyên dương.")

    if pairs_mode == "round_robin":
        pairs = [
            (agents[i], agents[j])
            for i in range(len(agents))
            for j in range(i + 1, len(agents))
        ]

    elif pairs_mode == "challengers_vs_baselines":
        if not baselines:
            raise ValueError(
                "pairs_mode='challengers_vs_baselines' "
                "yêu cầu danh sách baselines."
            )

        pairs = [
            (agent, baseline)
            for agent in agents
            for baseline in baselines
        ]

    elif pairs_mode == "self_play":
        pairs = [(agent, agent) for agent in agents]

    else:
        raise ValueError(
            f"pairs_mode không hợp lệ: {pairs_mode}"
        )

    if not pairs:
        raise ValueError("Không có cặp đấu nào được tạo.")

    matches = []
    game_id = 0

    for left_agent, right_agent in pairs:
        for pair_game_index in range(games_per_pair):

            if swap_colors:
                # Hai trận liên tiếp dùng cùng opening,
                # nhưng đổi vị trí X/O cho nhau.
                opening_id = (
                    pair_game_index // 2
                ) % len(openings)

                reverse_colors = pair_game_index % 2 == 1
            else:
                opening_id = pair_game_index % len(openings)
                reverse_colors = False

            if reverse_colors:
                agent_x_cfg = right_agent
                agent_o_cfg = left_agent
            else:
                agent_x_cfg = left_agent
                agent_o_cfg = right_agent

            matches.append({
                "game_id": game_id,
                "agent_x_cfg": agent_x_cfg,
                "agent_o_cfg": agent_o_cfg,
                "opening_id": opening_id,
                "opening_moves": list(openings[opening_id]),
                "seed": seed,
            })

            game_id += 1

    return matches
