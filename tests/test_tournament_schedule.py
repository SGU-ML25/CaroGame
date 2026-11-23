
import pytest

from caro.arena.schedule import build_match_schedule


@pytest.fixture
def agents():
    return [
        {"id": "A"},
        {"id": "B"},
        {"id": "C"},
    ]


@pytest.fixture
def openings():
    return [
        [60, 61],
        [50, 51],
        [40, 41],
    ]


def test_round_robin_pair_count(agents, openings):
    matches = build_match_schedule(
        agents,
        openings,
        pairs_mode="round_robin",
        games_per_pair=4,
    )

    # 3 agents tạo 3 cặp, mỗi cặp 4 trận.
    assert len(matches) == 3 * 4


def test_round_robin_has_no_self_pair(agents, openings):
    matches = build_match_schedule(
        agents,
        openings,
        pairs_mode="round_robin",
        games_per_pair=2,
    )

    for match in matches:
        assert match["agent_x_cfg"]["id"] != match["agent_o_cfg"]["id"]


def test_challengers_vs_baselines(agents, openings):
    baselines = [
        {"id": "BASE-1"},
        {"id": "BASE-2"},
    ]

    matches = build_match_schedule(
        agents,
        openings,
        pairs_mode="challengers_vs_baselines",
        games_per_pair=2,
        baselines=baselines,
    )

    assert len(matches) == 3 * 2 * 2

    for match in matches:
        ids = {
            match["agent_x_cfg"]["id"],
            match["agent_o_cfg"]["id"],
        }
        assert any(agent["id"].startswith("BASE-") for agent in [
            match["agent_x_cfg"],
            match["agent_o_cfg"],
        ])
        assert len(ids) == 2


def test_self_play(agents, openings):
    matches = build_match_schedule(
        agents,
        openings,
        pairs_mode="self_play",
        games_per_pair=2,
    )

    assert len(matches) == len(agents) * 2

    for match in matches:
        assert match["agent_x_cfg"]["id"] == match["agent_o_cfg"]["id"]


def test_same_opening_is_played_with_swapped_colors(agents, openings):
    matches = build_match_schedule(
        agents[:2],
        openings,
        pairs_mode="round_robin",
        games_per_pair=6,
        swap_colors=True,
    )

    for i in range(0, 6, 2):
        first = matches[i]
        second = matches[i + 1]

        assert first["opening_id"] == second["opening_id"]
        assert first["agent_x_cfg"]["id"] == second["agent_o_cfg"]["id"]
        assert first["agent_o_cfg"]["id"] == second["agent_x_cfg"]["id"]


def test_openings_cycle_correctly(agents, openings):
    matches = build_match_schedule(
        agents[:2],
        openings,
        pairs_mode="round_robin",
        games_per_pair=6,
        swap_colors=True,
    )

    assert [
        matches[i]["opening_id"]
        for i in range(0, 6, 2)
    ] == [0, 1, 2]


def test_swap_colors_false(agents, openings):
    matches = build_match_schedule(
        agents[:2],
        openings,
        pairs_mode="round_robin",
        games_per_pair=4,
        swap_colors=False,
    )

    for match in matches:
        assert match["agent_x_cfg"]["id"] == "A"
        assert match["agent_o_cfg"]["id"] == "B"


def test_game_ids_are_sequential(agents, openings):
    matches = build_match_schedule(
        agents,
        openings,
        pairs_mode="round_robin",
        games_per_pair=2,
    )

    assert [match["game_id"] for match in matches] == list(range(len(matches)))


def test_empty_openings_raises_error(agents):
    with pytest.raises(ValueError, match="openings"):
        build_match_schedule(agents, [])


def test_missing_baselines_raises_error(agents, openings):
    with pytest.raises(ValueError, match="baselines"):
        build_match_schedule(
            agents,
            openings,
            pairs_mode="challengers_vs_baselines",
        )


def test_invalid_pairs_mode_raises_error(agents, openings):
    with pytest.raises(ValueError, match="pairs_mode"):
        build_match_schedule(
            agents,
            openings,
            pairs_mode="invalid",
        )


def test_invalid_games_per_pair_raises_error(agents, openings):
    with pytest.raises(ValueError, match="games_per_pair"):
        build_match_schedule(
            agents,
            openings,
            games_per_pair=0,
        )
