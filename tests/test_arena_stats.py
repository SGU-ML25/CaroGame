
import pandas as pd
import pytest

from caro.arena.stats import (
    bootstrap,
    calculate_performance,
    detailed_record_by_turn,
    elo,
    generate_matrix,
    summarize_self_play,
    wilson,
)


def test_wilson():
    low, high = wilson(120, 200)

    assert 0 < low < 0.6
    assert 0.6 < high < 1


def test_bootstrap_is_reproducible():
    pairs = [
        (1.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (0.0, 0.0),
    ]

    result_1 = bootstrap(pairs, n_boot=500, seed=7)
    result_2 = bootstrap(pairs, n_boot=500, seed=7)

    assert result_1 == result_2


def test_bootstrap_empty():
    assert bootstrap([]) == (0.0, 0.0)


def test_bootstrap_invalid_confidence():
    with pytest.raises(ValueError):
        bootstrap([(1.0, 0.0)], confidence=1.0)


def test_elo_is_deterministic_when_game_ids_are_supplied():
    matches = [
        (2, "A", "B", 1),
        (0, "A", "B", 1),
        (1, "B", "A", -1),
    ]

    result_1 = elo(matches)
    result_2 = elo(list(reversed(matches)))

    assert result_1 == result_2


def test_performance_accounts_for_odd_opening_plies():
    df = pd.DataFrame([
        {
            "agent_x": "A",
            "agent_o": "B",
            "opening_plies": 3,
            "time_ms": [10, 20, 30],
            "nodes": [100, 200, 300],
            "sims": [0, 0, 0],
        }
    ])

    result = calculate_performance(df)

    # Opening có 3 nước nên lượt đo đầu tiên thuộc O.
    assert result["B"]["time_mean_ms"] == 20
    assert result["A"]["time_mean_ms"] == 20
    assert result["B"]["nodes_mean"] == 200
    assert result["A"]["nodes_mean"] == 200


def test_performance_old_log_defaults_to_even_opening():
    df = pd.DataFrame([
        {
            "agent_x": "A",
            "agent_o": "B",
            "time_ms": [10, 20],
            "nodes": [100, 200],
            "sims": [0, 0],
        }
    ])

    result = calculate_performance(df)

    assert result["A"]["time_mean_ms"] == 10
    assert result["B"]["time_mean_ms"] == 20

def test_performance_separates_nodes_and_sims():
    df = pd.DataFrame([
        {
            "agent_x": "L7-PUCT",
            "agent_o": "L8-AlphaZero",
            "opening_plies": 0,
            "time_ms": [10, 20],
            "nodes": [0, 0],
            "sims": [800, 600],
        }
    ])

    result = calculate_performance(df)

    assert result["L7-PUCT"]["nodes_mean"] == 0
    assert result["L7-PUCT"]["sims_mean"] == 800

    assert result["L8-AlphaZero"]["nodes_mean"] == 0
    assert result["L8-AlphaZero"]["sims_mean"] == 600
def test_detailed_record_by_turn():
    df = pd.DataFrame([
        {
            "agent_x": "A",
            "agent_o": "B",
            "result": 1,
        },
        {
            "agent_x": "B",
            "agent_o": "A",
            "result": -1,
        },
    ])

    result = detailed_record_by_turn(df)

    assert result["A"]["total"]["wins"] == 2
    assert result["B"]["total"]["losses"] == 2


def test_generate_matrix_reports_draws_separately(tmp_path):
    df = pd.DataFrame([
        {
            "agent_x": "A",
            "agent_o": "B",
            "result": 1,
        },
        {
            "agent_x": "B",
            "agent_o": "A",
            "result": 0,
        },
    ])

    matrix = generate_matrix(df, tmp_path)

    assert "W " in matrix.loc["A", "B"]
    assert "D " in matrix.loc["A", "B"]
    assert (tmp_path / "matrix.csv").exists()
    assert (tmp_path / "matrix.md").exists()


def test_generate_matrix_empty_matchup(tmp_path):
    df = pd.DataFrame([
        {"agent_x": "A", "agent_o": "B", "result": 1}
    ])

    matrix = generate_matrix(df, tmp_path)

    assert matrix.loc["B", "A"] != "—"
    assert matrix.loc["A", "B"] != "—"


def test_summarize_self_play():
    df = pd.DataFrame([
        {
            "game_id": 0,
            "opening_id": 0,
            "agent_x": "A",
            "agent_o": "A",
            "result": 1,
        },
        {
            "game_id": 1,
            "opening_id": 0,
            "agent_x": "A",
            "agent_o": "A",
            "result": -1,
        },
        {
            "game_id": 2,
            "opening_id": 1,
            "agent_x": "A",
            "agent_o": "A",
            "result": 1,
        },
        {
            "game_id": 3,
            "opening_id": 1,
            "agent_x": "A",
            "agent_o": "A",
            "result": 0,
        },
        {
            "game_id": 4,
            "opening_id": 2,
            "agent_x": "B",
            "agent_o": "C",
            "result": 1,
        },
    ])

    result = summarize_self_play(df)

    assert result["A"]["games"] == 4

    assert result["A"]["as_X"] == {
        "wins": 2,
        "draws": 1,
        "losses": 1,
        "score": 0.625,
    }

    assert result["A"]["as_O"] == {
        "wins": 1,
        "draws": 1,
        "losses": 2,
        "score": 0.375,
    }

    assert result["A"]["first_move_advantage"] == 0.25

    ci = result["A"]["first_move_advantage_ci"]

    assert ci["confidence"] == 0.95
    assert ci["n_boot"] == 1000
    assert ci["opening_pairs"] == 2
    assert ci["low"] <= 0.25 <= ci["high"]

    assert "B" not in result

def test_summarize_self_play_corrects_odd_opening_plies():
    df = pd.DataFrame([
        {
            "game_id": 0,
            "opening_id": 0,
            "opening_plies": 2,
            "agent_x": "A",
            "agent_o": "A",
            "result": 1,
        },
        {
            "game_id": 1,
            "opening_id": 0,
            "opening_plies": 2,
            "agent_x": "A",
            "agent_o": "A",
            "result": -1,
        },
        {
            "game_id": 2,
            "opening_id": 1,
            "opening_plies": 3,
            "agent_x": "A",
            "agent_o": "A",
            "result": -1,
        },
        {
            "game_id": 3,
            "opening_id": 1,
            "opening_plies": 3,
            "agent_x": "A",
            "agent_o": "A",
            "result": 1,
        },
    ])

    result = summarize_self_play(df, n_boot=200)

    # Opening 2 nước: X đi trước.
    # Opening 3 nước: O đi trước.
    assert result["A"]["as_first"]["wins"] == 2
    assert result["A"]["as_first"]["losses"] == 2
    assert result["A"]["as_first"]["score"] == 0.5
    assert result["A"]["first_move_advantage"] == 0.0