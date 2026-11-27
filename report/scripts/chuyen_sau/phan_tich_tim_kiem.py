"""Đo hành vi tìm kiếm và tốc độ từng thành phần. Chạy khi máy rảnh (số đo thời gian).

    .venv/bin/python report/scripts/chuyen_sau/phan_tich_tim_kiem.py

Ghi chuyen-sau/data/phan-tich-tim-kiem.json.
"""

from __future__ import annotations

import json
import random
import statistics
import time
import timeit

import chung
import torch
import vcf

from caro.agents import make_agent
from caro.agents.base import Limits
from caro.agents.mcts import rollout_fast
from caro.config import load_config
from caro.engine import CaroGame
from caro.engine.bitboard import candidates, evaluate, has_five, iter_bits, winning_squares
from caro.engine.zobrist import full_key
from caro.nn.features import encode
from caro.nn.model import CaroResNet

HEAT_IDS = ("C3-001", "C5-001")


def agents_by_id() -> dict:
    return {a["id"]: a for a in load_config("experiments/rr8_heldout")["agents"]}


def build(cfg: dict, params: dict | None = None, limits: dict | None = None):
    cfg = {**cfg, "params": {**cfg.get("params", {}), **(params or {})},
           "limits": {**cfg["limits"], **(limits or {})}}
    return make_agent(cfg), Limits.from_dict(cfg["limits"])


def midgame_positions() -> list[list[int]]:
    data = json.loads((chung.ROOT / "runs/openings/midgame_positions_v1.json").read_text(encoding="utf-8"))
    positions = [m for m in data["positions"] if not CaroGame.from_moves(m).is_terminal()]
    return random.Random(11).sample(positions, min(50, len(positions)))


def heatmaps(by_id: dict) -> list[dict]:
    suite = {p["id"]: p for p in chung.load_json("bo-chien-thuat")["positions"]}
    setups = [("L4 light", "L4-MCTS", {"rollout": "light"}, (100, 800, 6400)),
              ("L7", "L7-PUCT", {}, (50, 800, 3200)),
              ("L8", "L8-AlphaZero", {}, (50, 800, 3200))]
    out = []
    for pid in HEAT_IDS:
        position = suite[pid]
        maps = []
        for label, name, params, budgets in setups:
            for sims in budgets:
                agent, limits = build(by_id[name], params, {"max_sims": sims})
                agent.reset(chung.SEED)
                move = agent.select_action(CaroGame.from_moves(position["moves"]), limits)
                stats = agent.last_stats()
                visits = {int(k): int(v) for k, v in stats["visits"].items()}
                total = sum(visits.values()) or 1
                maps.append({"label": label, "sims": sims, "move": int(move), "visits": visits,
                             "answer_share": sum(v for k, v in visits.items() if k in position["answers"]) / total,
                             "score": stats.get("score"), "pv": [int(m) for m in stats.get("pv", [])][:8]})
        out.append({"position": position, "maps": maps})
    return out


def alphabeta_depths(by_id: dict, positions: list[list[int]]) -> list[dict]:
    out = []
    for depth in range(1, 7):
        agent, limits = build(by_id["L3-AlphaBeta"], {"backend": "native"}, {"max_depth": depth})
        rows = []
        for moves in positions:
            agent.reset(0)
            start = time.perf_counter()
            move = agent.select_action(CaroGame.from_moves(moves), limits)
            elapsed = (time.perf_counter() - start) * 1000
            s = agent.last_stats()
            rows.append({"move": int(move), "nodes": s.get("nodes", 0), "time_ms": elapsed,
                         "score": s.get("score"), "depth": s.get("depth")})
        out.append({"depth": depth, "rows": rows})
        print("alphabeta depth", depth, statistics.median(r["time_ms"] for r in rows), flush=True)
    return out


def backends(by_id: dict, positions: list[list[int]]) -> list[dict]:
    out = []
    for backend in ("python", "native"):
        for depth in (2, 3, 4):
            agent, limits = build(by_id["L3-AlphaBeta"], {"backend": backend}, {"max_depth": depth})
            times, nodes, moves_chosen = [], [], []
            for moves in positions[:20]:
                agent.reset(0)
                start = time.perf_counter()
                moves_chosen.append(int(agent.select_action(CaroGame.from_moves(moves), limits)))
                times.append((time.perf_counter() - start) * 1000)
                nodes.append(agent.last_stats().get("nodes", 0))
            out.append({"kind": "alphabeta", "backend": backend, "depth": depth, "time_ms": times,
                        "nodes": nodes, "moves": moves_chosen})
    for name in ("L7-PUCT", "L8-AlphaZero"):
        for backend in ("python", "native"):
            for batch in (1, 4, 8, 16, 32):
                agent, limits = build(by_id[name], {"backend": backend, "batch_leaves": batch})
                times, moves_chosen = [], []
                for moves in positions[:20]:
                    agent.reset(0)
                    start = time.perf_counter()
                    moves_chosen.append(int(agent.select_action(CaroGame.from_moves(moves), limits)))
                    times.append((time.perf_counter() - start) * 1000)
                out.append({"kind": "puct", "agent": name, "backend": backend, "batch": batch,
                            "time_ms": times, "moves": moves_chosen})
        print("puct backends", name, flush=True)
    for rollout in ("random", "light"):
        for sims in (100, 400, 800, 1600):
            agent, limits = build(by_id["L4-MCTS"], {"rollout": rollout}, {"max_sims": sims})
            times = []
            for moves in positions[:10]:
                agent.reset(0)
                start = time.perf_counter()
                agent.select_action(CaroGame.from_moves(moves), limits)
                times.append((time.perf_counter() - start) * 1000)
            out.append({"kind": "mcts", "rollout": rollout, "sims": sims, "time_ms": times})
        print("mcts", rollout, flush=True)
    return out


def micro(positions: list[list[int]]) -> list[dict]:
    games = [CaroGame.from_moves(m) for m in positions]
    pairs = [g.me_opp() for g in games]
    empties = [next(iter_bits(g.empty)) for g in games]
    rng = random.Random(0)

    def step_undo():
        for g, a in zip(games, empties, strict=True):
            g.step(a)
            g.undo()

    cases = [
        ("has\\_five", lambda: [has_five(me) for me, _ in pairs], 200),
        ("candidates (bán kính 2)", lambda: [candidates(me, opp, 2) for me, opp in pairs], 200),
        ("winning\\_squares", lambda: [winning_squares(me, opp) for me, opp in pairs], 100),
        ("evaluate", lambda: [evaluate(me, opp) for me, opp in pairs], 100),
        ("step + undo", step_undo, 100),
        ("clone", lambda: [g.clone() for g in games], 200),
        ("full\\_key (Zobrist từ đầu)", lambda: [full_key(g.bb[0], g.bb[1], g.side) for g in games], 100),
        ("encode (đặc trưng mạng)", lambda: [encode(g) for g in games], 100),
        ("rollout random", lambda: [rollout_fast(g.bb[0], g.bb[1], g.side, rng, "random", 1) for g in games], 5),
        ("rollout light", lambda: [rollout_fast(g.bb[0], g.bb[1], g.side, rng, "light", 1) for g in games], 3),
        ("vcf.solve", lambda: [vcf.solve(me, opp) for me, opp in pairs], 5),
    ]
    out = []
    for name, function, number in cases:
        best = min(timeit.repeat(function, number=number, repeat=5))
        out.append({"name": name, "us": best / number / len(games) * 1e6})
        print(name, round(out[-1]["us"], 2), "µs", flush=True)
    return out


@torch.no_grad()
def throughput() -> list[dict]:
    net = CaroResNet({"num_channels": 64, "num_blocks": 6}).eval()
    out = []
    for threads in (1, 4, 8):
        torch.set_num_threads(threads)
        for batch in (1, 2, 4, 8, 16, 32, 64, 256):
            x = torch.zeros(batch, 4, 11, 11)
            net(x)
            times = []
            for _ in range(30 if batch < 64 else 10):
                start = time.perf_counter()
                net(x)
                times.append((time.perf_counter() - start) * 1000)
            ms = statistics.median(times)
            out.append({"threads": threads, "batch": batch, "ms": ms, "per_second": batch / ms * 1000})
    return out


def main() -> None:
    by_id = agents_by_id()
    positions = midgame_positions()
    default_threads = torch.get_num_threads()
    result = {"positions": len(positions), "torch_threads": default_threads,
              "micro": micro(positions), "backends": backends(by_id, positions),
              "alphabeta": alphabeta_depths(by_id, positions), "heatmaps": heatmaps(by_id),
              "throughput": throughput()}
    chung.save_json("phan-tich-tim-kiem", result)


if __name__ == "__main__":
    main()
