import json

import pytest

from caro.config import load_config
from caro.runlog import Run, derive_seed


def _cfg(tmp_path, *extra):
    return load_config("experiments/smoke",
                       [f"run.root={tmp_path}/runs", f"run.index={tmp_path}/runs/index.jsonl",
                        "log.console=false", *extra])


def _game(**kw):
    return {"game_id": 0, "tournament_id": "t", "agent_x": "L1-Random", "agent_o": "L2-Rule",
            "opening_id": 3, "seed": 42, "moves": [60, 61], "time_ms": [1, 2], "nodes": [0, 5],
            "result": 1, "termination": "five", **kw}


def test_run_dir_layout(tmp_path):
    with Run(_cfg(tmp_path)) as run:
        run.log.info("xin chào")
        run.log_game(_game())
        run.log_metric(1, loss=0.5)
    d = run.dir
    assert d.parent == tmp_path / "runs" / "tournament" / "smoke"
    for name in ("config.yaml", "meta.json", "run.log", "games.jsonl", "metrics.jsonl", "summary.json"):
        assert (d / name).is_file(), name
    assert (d / "checkpoints").is_dir()
    assert (d.parent / "latest").resolve() == d.resolve()

    game = json.loads((d / "games.jsonl").read_text())
    assert game["run_id"] == run.id and game["moves"] == [60, 61]
    assert json.loads((d / "summary.json").read_text())["games"] == 1
    meta = json.loads((d / "meta.json").read_text())
    assert meta["seed"] == 2026 and len(meta["config_hash"]) == 12
    assert "xin chào" in (d / "run.log").read_text()

    events = [json.loads(line) for line in (tmp_path / "runs" / "index.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["start", "end"]
    assert events[1]["status"] == "ok"


def test_tag_and_same_second_runs(tmp_path):
    cfg = _cfg(tmp_path, "run.tag=thu")
    a, b = Run(cfg), Run(cfg)
    a.finish(); b.finish()
    assert a.dir != b.dir and a.dir.name.endswith("_thu")


def test_error_status_and_validation(tmp_path):
    with pytest.raises(RuntimeError), Run(_cfg(tmp_path)) as run:
        with pytest.raises(ValueError, match="thiếu"):
            run.log_game({"game_id": 1})
        raise RuntimeError("hỏng")
    assert json.loads((run.dir / "summary.json").read_text())["status"] == "error"


def test_per_move_stats_off(tmp_path):
    with Run(_cfg(tmp_path, "log.per_move_stats=false")) as run:
        run.log_game(_game())
    game = json.loads((run.dir / "games.jsonl").read_text())
    assert "time_ms" not in game and "nodes" not in game


def test_derive_seed_stable_and_distinct():
    assert derive_seed(2026, 5, "X") == derive_seed(2026, 5, "X")
    seeds = {derive_seed(2026, g, c) for g in range(200) for c in "XO"}
    assert len(seeds) == 400
    assert 0 <= derive_seed(1) < 2**63
