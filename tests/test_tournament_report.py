
import json

from caro.arena.tournament import generate_tournament_report


def test_generate_tournament_report_round_robin(tmp_path):
    game_records = [
        {
            "game_id": 0,
            "agent_x": "A",
            "agent_o": "B",
            "result": 1,
            "opening_plies": 0,
            "time_ms": [10, 20],
            "nodes": [100, 200],
            "sims": [],
        },
        {
            "game_id": 1,
            "agent_x": "B",
            "agent_o": "A",
            "result": -1,
            "opening_plies": 0,
            "time_ms": [15, 25],
            "nodes": [150, 250],
            "sims": [],
        },
    ]

    report_dir = tmp_path / "reports"

    report = generate_tournament_report(
        game_records=game_records,
        pairs_mode="round_robin",
        report_dir=report_dir,
        arena_cfg={"stats": {"elo": True}},
    )

    assert "detailed_record_by_turn" in report
    assert "performance" in report
    assert "elo" in report
    assert "self_play" not in report

    assert (report_dir / "matrix.csv").exists()
    assert (report_dir / "matrix.md").exists()
    assert (report_dir / "stats.json").exists()

    with open(report_dir / "stats.json", encoding="utf-8") as f:
        saved_report = json.load(f)

    assert saved_report == report


def test_generate_tournament_report_self_play(tmp_path):
    game_records = [
        {
            "game_id": 0,
            "agent_x": "A",
            "agent_o": "A",
            "result": 1,
            "opening_id": 0,
            "opening_plies": 2,
            "time_ms": [10, 20],
            "nodes": [100, 200],
            "sims": [],
        },
        {
            "game_id": 1,
            "agent_x": "A",
            "agent_o": "A",
            "result": -1,
            "opening_id": 0,
            "opening_plies": 2,
            "time_ms": [15, 25],
            "nodes": [150, 250],
            "sims": [],
        },
    ]

    report_dir = tmp_path / "reports"

    report = generate_tournament_report(
        game_records=game_records,
        pairs_mode="self_play",
        report_dir=report_dir,
        arena_cfg={
            "stats": {
                "elo": True,
                "bootstrap": 500,
                "confidence": 0.90,
            }
        },
    )

    assert "self_play" in report
    assert "A" in report["self_play"]

    agent_stats = report["self_play"]["A"]

    assert agent_stats["games"] == 2
    assert agent_stats["as_X"]["wins"] == 1
    assert agent_stats["as_O"]["wins"] == 1
    assert agent_stats["first_move_advantage"] == 0.0

    ci = agent_stats["first_move_advantage_ci"]
    assert ci["confidence"] == 0.90
    assert ci["n_boot"] == 500
    assert ci["opening_pairs"] == 1

    assert (report_dir / "stats.json").exists()
    assert not (report_dir / "matrix.csv").exists()
    assert "elo" not in report

    with open(report_dir / "stats.json", encoding="utf-8") as f:
        saved_report = json.load(f)

    assert saved_report["self_play"]["A"]["first_move_advantage_ci"] == ci
