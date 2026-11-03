"""API và trang demo Caro 11×11: người–máy, máy–máy, xem lại ván (tài liệu thiết kế, Phần 11).

    uv sync --extra ui
    uv run uvicorn caro.ui.server:app      # http://127.0.0.1:8000

Tác tử dùng đúng code và config của arena, chỉ đi qua interface `Agent`. Mỗi nước của tác tử chạy trong
thread pool trên bản sao của bàn cờ, nên tác tử chậm không chặn server và GET /games/{id} vẫn trả lời được.
Ván đang chơi chỉ nằm trong bộ nhớ của tiến trình server.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from caro.agents import REGISTRY, Agent, Limits, make_agent
from caro.config import CONFIG_ROOT, ConfigError, load_config, project_path, resolve_agent
from caro.engine import CaroGame, IllegalMoveError, N, O, X
from caro.engine.bitboard import LINES, WIN, index, iter_bits
from caro.runlog import derive_seed

HUMAN = "human"
STATIC_DIR = Path(__file__).parent / "static"
MAX_GAMES = 256           # ván cũ nhất bị bỏ khi vượt số này (mỗi ván giữ tác tử và bảng TT riêng)
MAX_REPLAYS = 500         # số ván tối đa trong một lần GET /replays
# Một nước đang chạy trong thread không huỷ được, nên chặn ngân sách quá lớn ngay từ đầu.
MAX_BUDGET = {"time_ms": 60_000, "max_depth": 8, "max_sims": 100_000}
# Ngân sách mà từng kind của tác tử thực sự dùng; kind không có ở đây chỉ đi một nước theo luật / mạng.
BUDGET_KEYS = {"alphabeta": ["time_ms", "max_depth"], "mcts": ["time_ms", "max_sims"], "puct": ["time_ms", "max_sims"]}


class NewGame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: str = HUMAN
    o: str = HUMAN
    seed: int = 0
    budget: dict[str, Any] | None = None
    opening_id: int | str | None = None                  # một khai cuộc trong file của arena
    opening_plies: int = Field(default=0, ge=0, le=8)    # hoặc sinh ngẫu nhiên theo seed trong vùng giữa bàn


class Move(BaseModel):
    action: int


@dataclass
class Session:
    id: str
    game: CaroGame
    players: tuple[str, str]
    agents: tuple[Agent | None, Agent | None]
    limits: Limits
    seed: int
    opening_id: int | str | None
    floor: int                             # số nước khai cuộc; không lùi qua mốc này
    stats: list[dict] = field(default_factory=list)   # stats[i] là của nước thứ i trong history
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def agent_catalog(root: Path = CONFIG_ROOT) -> OrderedDict[str, dict]:
    """id tác tử → thông tin hiển thị, đọc từ configs/agents/ (kể cả cấp chưa cài, để UI báo rõ)."""
    agents = [resolve_agent(f"agents/{p.stem}", root=root) for p in sorted((root / "agents").glob("*.yaml"))]
    agents.sort(key=lambda a: a.get("level", 0))
    return OrderedDict(
        (a["id"], {
            "id": a["id"],
            "level": a.get("level"),
            "kind": a["kind"],
            "source": a["source"],
            "available": a["kind"] in REGISTRY,
            "budget": BUDGET_KEYS.get(a["kind"], []),
        })
        for a in agents
    )


def parse_budget(budget: dict[str, Any] | None, default_fixed: dict[str, Any]) -> Limits:
    """Ngân sách của ván. Chế độ fixed bổ sung giá trị còn thiếu từ chế độ B của giải chính."""
    budget = budget or {"mode": "fixed"}
    if budget.get("mode", "fixed") == "fixed":
        budget = {**default_fixed, **{k: v for k, v in budget.items() if v is not None}}
    try:
        limits = Limits.from_dict(budget)
    except (TypeError, ValueError) as e:
        raise HTTPException(400, str(e)) from None
    for key, top in MAX_BUDGET.items():
        value = getattr(limits, key)
        if value is not None and value > top:
            raise HTTPException(400, f"budget.{key} tối đa {top}, nhận {value}")
    return limits


def random_opening(plies: int, seed: int, region: int) -> list[int]:
    """Khai cuộc ngẫu nhiên có seed, trong vùng region×region giữa bàn."""
    low = (N - region) // 2
    cells = [index(r, c) for r in range(low, low + region) for c in range(low, low + region)]
    return random.Random(seed).sample(cells, plies)


def read_openings(path: Path) -> list[dict]:
    """Đọc file khai cuộc của arena: danh sách, mỗi mục là {id, moves} hoặc một list nước đi."""
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["openings"] if isinstance(data, dict) else data
    return [
        {"id": item.get("id", i), "moves": item["moves"]} if isinstance(item, dict) else {"id": i, "moves": item}
        for i, item in enumerate(items)
    ]


def winning_cells(board: int) -> list[int]:
    """Các ô thuộc dãy ≥ 5 quân, để tô dãy thắng."""
    cells: set[int] = set()
    for d, start in LINES:
        t = board & (board >> d)
        t &= t >> (2 * d)
        t &= board >> (4 * d)
        for s in iter_bits(t & start):
            cells.update(s + k * d for k in range(WIN))
    return sorted(cells)


def plain(value: Any) -> Any:
    """Đưa last_stats() về kiểu JSON (tác tử cấp 5–8 có thể trả số / mảng numpy, khoá ô là int)."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):      # numpy scalar / array, torch tensor
        return value.tolist()
    return value


def iter_replays(root: Path) -> Iterator[dict]:
    """Mọi bản ghi ván trong <root>/**/games.jsonl, run mới nhất trước; dòng hỏng bị bỏ qua."""
    if not root.is_dir():
        return
    files = sorted(root.rglob("games.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and isinstance(record.get("moves"), list):
                    yield record


def create_app(
    runs_root: Path | None = None,
    openings_file: Path | None = None,
    agent_factory: Callable[[dict], Agent] = make_agent,
    catalog: dict[str, dict] | None = None,
) -> FastAPI:
    cfg = load_config("experiments/main_mode_b")     # ngân sách fixed, khai cuộc, thư mục run của giải chính
    arena, fixed = cfg["arena"]["openings"], cfg["budget"]
    runs_root = runs_root if runs_root is not None else project_path(cfg["run"]["root"])
    openings_file = openings_file if openings_file is not None else project_path(arena["file"])
    agents = catalog if catalog is not None else agent_catalog()
    games: OrderedDict[str, Session] = OrderedDict()

    app = FastAPI(title="Caro 11×11", description="Người–máy, máy–máy, xem lại ván")

    def build_agent(player: str, seed: int, color: str) -> Agent | None:
        if player == HUMAN:
            return None
        info = agents.get(player)
        if info is None:
            raise HTTPException(400, f"không có tác tử '{player}'")
        if not info["available"]:
            raise HTTPException(400, f"tác tử '{player}' chưa được cài (kind '{info['kind']}')")
        try:
            agent = agent_factory(resolve_agent(info["source"]))
        except ConfigError as e:
            raise HTTPException(400, str(e)) from None
        agent.reset(derive_seed(seed, "ui", color))    # cùng seed → cùng ván (ngân sách fixed)
        return agent

    def opening_moves(body: NewGame) -> list[int]:
        if body.opening_id is not None and body.opening_plies:
            raise HTTPException(400, "chỉ chọn một trong opening_id, opening_plies")
        if body.opening_id is not None:
            for item in read_openings(openings_file):
                if str(item["id"]) == str(body.opening_id):
                    return item["moves"]
            raise HTTPException(404, f"không có khai cuộc '{body.opening_id}'")
        return random_opening(body.opening_plies, body.seed, arena["region"])

    def get_session(game_id: str) -> Session:
        session = games.get(game_id)
        if session is None:
            raise HTTPException(404, f"không có ván '{game_id}'")
        return session

    def state(s: Session) -> dict:
        g = s.game
        return {
            "id": s.id,
            "players": {"X": s.players[X], "O": s.players[O]},
            "budget": {"mode": s.limits.mode, "time_ms": s.limits.time_ms,
                       "max_depth": s.limits.max_depth, "max_sims": s.limits.max_sims},
            "seed": s.seed,
            "opening_id": s.opening_id,
            "opening_plies": s.floor,
            "moves": g.history,
            "stats": s.stats,
            "side": "XO"[g.side],
            "to_play": None if g.is_terminal() else s.players[g.side],
            "result": g.result(),
            "termination": g.termination(),
            "winning_line": winning_cells(g.bb[g.winner]) if g.winner is not None else [],
            "board": g.render(),
        }

    @app.get("/agents")
    def list_agents() -> list[dict]:
        return list(agents.values())

    @app.get("/budgets")
    def budgets() -> dict:
        """Giá trị mặc định cho ô nhập ngân sách, đúng với hai chế độ của giải chính."""
        return {"fixed": fixed, "time": load_config("experiments/main_mode_a")["budget"]}

    @app.get("/openings")
    def list_openings() -> list[dict]:
        return read_openings(openings_file)

    @app.post("/games", status_code=201)
    async def new_game(body: NewGame) -> dict:
        limits = parse_budget(body.budget, fixed)
        try:
            game = await run_in_threadpool(CaroGame.from_moves, opening_moves(body))
        except IllegalMoveError as e:
            raise HTTPException(400, f"khai cuộc không hợp lệ: {e}") from None
        built = await run_in_threadpool(
            lambda: (build_agent(body.x, body.seed, "X"), build_agent(body.o, body.seed, "O")))
        session = Session(uuid.uuid4().hex[:12], game, (body.x, body.o), built, limits, body.seed, body.opening_id,
                          floor=len(game.history), stats=[{} for _ in game.history])
        games[session.id] = session
        while len(games) > MAX_GAMES:
            games.popitem(last=False)
        return state(session)

    @app.get("/games/{game_id}")
    def get_game(game_id: str) -> dict:
        return state(get_session(game_id))

    @app.post("/games/{game_id}/move")
    async def human_move(game_id: str, body: Move) -> dict:
        s = get_session(game_id)
        async with s.lock:
            g = s.game
            if not g.is_terminal() and s.players[g.side] != HUMAN:
                raise HTTPException(409, f"đang là lượt của tác tử {s.players[g.side]}")
            try:
                g.step(body.action)
            except IllegalMoveError as e:
                raise HTTPException(400, str(e)) from None
            s.stats.append({})
            return state(s)

    @app.post("/games/{game_id}/ai-move")
    async def ai_move(game_id: str) -> dict:
        s = get_session(game_id)
        async with s.lock:
            g = s.game
            if g.is_terminal():
                raise HTTPException(409, "ván đã kết thúc")
            side = g.side
            agent = s.agents[side]
            if agent is None:
                raise HTTPException(409, "đang là lượt của người chơi")
            t0 = time.perf_counter()
            action = await run_in_threadpool(agent.select_action, g.clone(), s.limits)
            elapsed = (time.perf_counter() - t0) * 1000
            try:
                g.step(action)
            except IllegalMoveError as e:         # lỗi của tác tử, không phải của người gọi
                raise HTTPException(500, f"{agent.name} trả nước không hợp lệ: {e}") from None
            stats = {"time_ms": round(elapsed, 1), **plain(agent.last_stats())}
            s.stats.append(stats)
            return {**state(s), "action": action, "agent": agent.name, "last_stats": stats}

    @app.post("/games/{game_id}/undo")
    async def undo(game_id: str, plies: int | None = Query(default=None, ge=1)) -> dict:
        """Lùi nước. Mặc định: ván người–máy lùi về lượt người chơi, ván khác lùi 1 nước."""
        s = get_session(game_id)
        async with s.lock:
            g = s.game
            if plies is None:
                one_human = (s.players[X] == HUMAN) != (s.players[O] == HUMAN)
                plies = 2 if one_human and s.players[g.side] == HUMAN else 1
            if len(g.history) - plies < s.floor:
                raise HTTPException(400, "không còn nước để lùi")
            for _ in range(plies):
                g.undo()
                s.stats.pop()
            return state(s)

    @app.get("/replays")
    def list_replays(
        q: str = Query(default="", description="lọc theo tên tác tử hoặc mã giải"),
        limit: int = Query(default=200, ge=1, le=MAX_REPLAYS),
    ) -> list[dict]:
        out = []
        q = q.lower()
        for r in iter_replays(runs_root):
            label = f"{r.get('tournament_id')} {r.get('agent_x')} {r.get('agent_o')}".lower()
            if q in label:
                out.append({k: r.get(k) for k in ("run_id", "tournament_id", "game_id", "agent_x", "agent_o",
                                                  "opening_id", "result", "termination")}
                           | {"plies": len(r["moves"])})
                if len(out) == limit:
                    break
        return out

    @app.get("/replays/{game_id}")
    def get_replay(game_id: str, run_id: str | None = None, tournament_id: str | None = None) -> dict:
        """Một ván từ log. game_id chỉ duy nhất trong một giải, nên có thể cần thêm run_id / tournament_id."""
        found = [
            r for r in iter_replays(runs_root)
            if str(r.get("game_id")) == game_id
            and (run_id is None or str(r.get("run_id")) == run_id)
            and (tournament_id is None or str(r.get("tournament_id")) == tournament_id)
        ]
        if not found:
            raise HTTPException(404, f"không có ván '{game_id}' trong log")
        if len(found) > 1:
            raise HTTPException(409, f"{len(found)} ván trùng game_id '{game_id}': cần run_id hoặc tournament_id")
        record = found[0]
        try:
            game = CaroGame.from_moves(record["moves"])
        except IllegalMoveError as e:
            raise HTTPException(422, f"log có nước không hợp lệ: {e}") from None
        return {**record, "winning_line": winning_cells(game.bb[game.winner]) if game.winner is not None else []}

    @app.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
