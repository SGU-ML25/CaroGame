"""API demo: người–máy, máy–máy, lùi nước, replay, tác tử chậm không chặn server."""

import asyncio
import json
import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx
from fastapi.testclient import TestClient

from caro.agents.base import BaseAgent
from caro.engine import CaroGame
from caro.ui.server import MAX_GAMES, agent_catalog, create_app, winning_cells


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(runs_root=tmp_path / "runs", openings_file=tmp_path / "openings.json"))


def play_human(client, gid, action):
    return client.post(f"/games/{gid}/move", json={"action": action})


def new_game(client, **body):
    r = client.post("/games", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def test_agents_lists_every_config_and_marks_unavailable(client):
    agents = client.get("/agents").json()
    assert [a["level"] for a in agents] == list(range(1, 9))
    by_level = {a["level"]: a for a in agents}
    assert all(by_level[k]["available"] for k in (1, 2, 3))
    assert by_level[3]["budget"] == ["time_ms", "max_depth"]
    assert by_level[1]["budget"] == []


def test_unavailable_or_unknown_agent_is_rejected(client):
    unavailable = [a["id"] for a in client.get("/agents").json() if not a["available"]]
    for name in [*unavailable, "khong-co"]:
        assert client.post("/games", json={"x": name}).status_code == 400


@pytest.mark.parametrize("level_id", ["L1-Random", "L2-Rule"])
def test_full_human_vs_ai_game_through_api(client, level_id):
    game = new_game(client, x="human", o=level_id, seed=3)
    gid = game["id"]
    assert game["to_play"] == "human"
    while game["result"] is None:
        replay = CaroGame.from_moves(game["moves"])
        action = replay.legal_actions()[0]
        r = play_human(client, gid, action)
        assert r.status_code == 200, r.text
        game = r.json()
        if game["result"] is not None:
            break
        assert game["to_play"] == level_id
        r = client.post(f"/games/{gid}/ai-move")
        assert r.status_code == 200, r.text
        game = r.json()
        assert game["moves"][-1] == game["action"]
        assert "time_ms" in game["last_stats"]
    assert client.get(f"/games/{gid}").json()["result"] == game["result"]
    assert game["termination"] in ("five", "full")
    assert len(game["stats"]) == len(game["moves"])


def test_illegal_moves_return_400_and_keep_state(client):
    gid = new_game(client)["id"]
    assert play_human(client, gid, 60).status_code == 200
    for bad in (60, -1, 121):
        r = play_human(client, gid, bad)
        assert r.status_code == 400, bad
    assert client.get(f"/games/{gid}").json()["moves"] == [60]


def test_wrong_turn_is_conflict(client):
    game = new_game(client, x="L1-Random", o="human")
    gid = game["id"]
    assert play_human(client, gid, 60).status_code == 409
    client.post(f"/games/{gid}/ai-move")
    assert client.post(f"/games/{gid}/ai-move").status_code == 409


def test_win_reports_line_and_blocks_more_moves(client):
    gid = new_game(client)["id"]
    for a in (0, 11, 1, 12, 2, 13, 3, 14, 4):
        game = play_human(client, gid, a).json()
    assert game["result"] == 1
    assert game["winning_line"] == [0, 1, 2, 3, 4]
    assert game["to_play"] is None
    assert play_human(client, gid, 50).status_code == 400
    assert client.post(f"/games/{gid}/ai-move").status_code == 409


def test_winning_cells_covers_overline_without_row_wrap():
    assert winning_cells(sum(1 << i for i in range(6))) == list(range(6))
    assert winning_cells(sum(1 << i for i in range(30, 35))) == []       # lọt hàng 30…34 không phải dãy 5


def test_undo_returns_to_human_turn(client):
    gid = new_game(client, x="human", o="L1-Random", seed=1)["id"]
    assert client.post(f"/games/{gid}/undo").status_code == 400
    play_human(client, gid, 60)
    client.post(f"/games/{gid}/ai-move")
    game = client.post(f"/games/{gid}/undo").json()
    assert game["moves"] == [] and game["to_play"] == "human" and game["stats"] == []


def test_undo_human_as_o_never_rewinds_past_first_ai_move(client):
    gid = new_game(client, x="L1-Random", o="human", seed=1)["id"]
    first = client.post(f"/games/{gid}/ai-move").json()["moves"]
    assert client.post(f"/games/{gid}/undo").status_code == 400
    assert client.get(f"/games/{gid}").json()["moves"] == first


def test_undo_after_human_win_and_explicit_plies(client):
    gid = new_game(client)["id"]
    for a in (0, 11, 1, 12, 2, 13, 3, 14, 4):
        play_human(client, gid, a)
    game = client.post(f"/games/{gid}/undo").json()
    assert game["result"] is None and game["winning_line"] == [] and len(game["moves"]) == 8
    assert len(client.post(f"/games/{gid}/undo", params={"plies": 3}).json()["moves"]) == 5
    assert client.post(f"/games/{gid}/undo", params={"plies": 99}).status_code == 400


def test_ai_vs_ai_alphabeta_is_deterministic_for_seed(client):
    def play():
        game = new_game(client, x="L3-AlphaBeta", o="L2-Rule", seed=7, opening_plies=2,
                        budget={"mode": "fixed", "max_depth": 1})
        gid = game["id"]
        for _ in range(4):
            game = client.post(f"/games/{gid}/ai-move").json()
        return game
    a, b = play(), play()
    assert len(a["moves"]) == 6
    assert a["last_stats"]["reason"]
    assert a["stats"][2]["depth"] == 1 and a["stats"][2]["nodes"] > 0
    assert isinstance(a["stats"][2]["pv"], list)
    assert b["moves"] == a["moves"]                # cùng seed + ngân sách fixed → cùng ván


def test_time_budget_is_accepted_and_invalid_budgets_rejected(client):
    game = new_game(client, x="L3-AlphaBeta", budget={"mode": "time", "time_ms": 50})
    assert game["budget"]["time_ms"] == 50
    assert client.post(f"/games/{game['id']}/ai-move").status_code == 200
    for bad in ({"mode": "time"}, {"mode": "x"}, {"max_depth": 0}, {"lol": 1}, {"max_depth": 99},
                {"mode": "time", "time_ms": 10**7}):
        assert client.post("/games", json={"budget": bad}).status_code == 400, bad


def test_default_budget_comes_from_config(client):
    assert new_game(client)["budget"] == {"mode": "fixed", "time_ms": None, "max_depth": 4, "max_sims": 800}


def test_budget_defaults_for_page(client):
    budgets = client.get("/budgets").json()
    assert budgets["fixed"]["max_depth"] == 4 and budgets["time"]["time_ms"] == 1000


def test_openings(tmp_path):
    path = tmp_path / "openings.json"
    path.write_text(json.dumps([{"id": 7, "moves": [60, 61]}, [50, 51, 52]]), encoding="utf-8")
    client = TestClient(create_app(runs_root=tmp_path, openings_file=path))
    assert [o["id"] for o in client.get("/openings").json()] == [7, 1]
    game = new_game(client, opening_id=7)
    assert game["moves"] == [60, 61] and game["opening_plies"] == 2
    assert client.post(f"/games/{game['id']}/undo").status_code == 400
    assert client.post("/games", json={"opening_id": 99}).status_code == 404
    assert client.post("/games", json={"opening_id": 7, "opening_plies": 2}).status_code == 400


def test_random_opening_depends_on_seed_and_stays_in_region(client):
    a = new_game(client, seed=1, opening_plies=4)["moves"]
    assert a == new_game(client, seed=1, opening_plies=4)["moves"]
    assert a != new_game(client, seed=2, opening_plies=4)["moves"]
    assert all(3 <= i // 11 <= 7 and 3 <= i % 11 <= 7 for i in a)


def test_replays(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "tournament" / "t" / "20261103-1000"
    run.mkdir(parents=True)
    rec = {"run_id": "r1", "game_id": 5, "tournament_id": "t", "agent_x": "L1-Random", "agent_o": "L2-Rule",
           "opening_id": 0, "seed": 1, "moves": [0, 11, 1, 12, 2, 13, 3, 14, 4], "result": 1, "termination": "five",
           "time_ms": [1] * 9}
    other = {**rec, "run_id": "r2", "tournament_id": "khac", "agent_x": "L3-AlphaBeta"}
    (run / "games.jsonl").write_text(
        json.dumps(rec) + "\nhỏng\n" + json.dumps(other) + "\n" + json.dumps({**rec, "game_id": 6, "moves": [0, 0]}) + "\n",
        encoding="utf-8")
    client = TestClient(create_app(runs_root=runs))

    assert len(client.get("/replays").json()) == 3
    assert [r["run_id"] for r in client.get("/replays", params={"q": "alphabeta"}).json()] == ["r2"]
    assert client.get("/replays", params={"limit": 1}).json()[0]["plies"] == 9

    assert client.get("/replays/5").status_code == 409
    got = client.get("/replays/5", params={"run_id": "r1"}).json()
    assert got["winning_line"] == [0, 1, 2, 3, 4] and got["time_ms"] == [1] * 9
    assert client.get("/replays/5", params={"tournament_id": "khac"}).status_code == 200
    assert client.get("/replays/404").status_code == 404
    assert client.get("/replays/6").status_code == 422


def test_replays_without_runs_dir(client):
    assert client.get("/replays").json() == []
    assert client.get("/replays/1").status_code == 404
    assert client.get("/openings").json() == []


def test_old_games_are_evicted(client):
    first = new_game(client)["id"]
    for _ in range(MAX_GAMES):
        new_game(client)
    assert client.get(f"/games/{first}").status_code == 404


def test_serves_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Caro 11×11" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_catalog_is_read_from_configs():
    assert agent_catalog()["L4-MCTS"]["source"] == "agents/l4_mcts"


class _SlowAgent(BaseAgent):
    """Tác tử chờ tới khi test cho đi, trả stats kiểu cấp 5–8 (khoá ô là int, số dạng numpy)."""

    def __init__(self, gate: threading.Event, started: threading.Event) -> None:
        super().__init__(name="slow")
        self.gate, self.started = gate, started

    def select_action(self, game, limits):
        self.started.set()
        assert self.gate.wait(5)
        np = pytest.importorskip("numpy")
        self._stats = {"visits": {60: np.int64(3)}, "policy": {60: np.float64(0.5)}, "value": np.float32(0.25)}
        return 60


def test_slow_agent_does_not_block_server_and_stats_are_json():
    gate, started = threading.Event(), threading.Event()
    catalog = {"slow": {"id": "slow", "level": 9, "kind": "slow", "source": "agents/l1_random",
                        "available": True, "budget": []}}
    app = create_app(agent_factory=lambda cfg: _SlowAgent(gate, started), catalog=catalog)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            gid = (await c.post("/games", json={"x": "slow"})).json()["id"]
            move = asyncio.create_task(c.post(f"/games/{gid}/ai-move"))
            await asyncio.to_thread(started.wait, 5)
            during = await asyncio.wait_for(c.get(f"/games/{gid}"), 2)      # không bị chặn khi tác tử đang nghĩ
            assert during.json()["moves"] == []
            gate.set()
            return (await move).json()

    done = asyncio.run(scenario())
    assert done["moves"] == [60]
    assert done["last_stats"]["visits"] == {"60": 3}
    assert done["last_stats"]["policy"] == {"60": 0.5}
    assert done["last_stats"]["value"] == 0.25
