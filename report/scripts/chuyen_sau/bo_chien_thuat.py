"""Bộ kiểm tra chiến thuật: thế cờ có đáp án kiểm được bằng bộ giải VCF, chấm tám tác tử.

    .venv/bin/python report/scripts/chuyen_sau/bo_chien_thuat.py build
    .venv/bin/python report/scripts/chuyen_sau/bo_chien_thuat.py run base scale variants
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time

import chung
import vcf

from caro.agents import make_agent
from caro.agents.base import Limits
from caro.config import load_config
from caro.engine import CaroGame
from caro.engine.bitboard import FULL, candidates, iter_bits, popcount, winning_squares
from caro.engine.symmetry import transform_bb

PER_CATEGORY = 100
SOURCES = ("runs/eval/tournament/rr8_heldout/20261126-194315", "runs/eval/tournament/e3_heldout/latest")
CATEGORIES = {
    "C1": "Thắng ngay",
    "C2": "Chặn ô thắng duy nhất",
    "C3": "Tạo đòn kép (VCF 2 nước)",
    "C4": "Chuỗi tứ dài (VCF $\\ge$ 3 nước)",
    "C5": "Phòng thủ trước VCF",
}


def canonical(me: int, opp: int) -> tuple[int, int]:
    return min((transform_bb(me, k), transform_bb(opp, k)) for k in range(8))


def classify(me: int, opp: int) -> tuple[str, int] | None:
    """(loại, mặt nạ đáp án) hoặc None nếu thế không thuộc loại nào."""
    wins = winning_squares(me, opp)
    if wins:
        return "C1", wins
    threats = winning_squares(opp, me)
    if threats:
        return ("C2", threats) if popcount(threats) == 1 else None
    line = vcf.solve(me, opp)
    if line is not None:
        answers = vcf.winning_first_moves(me, opp)
        return ("C3" if vcf.attacks_in(line) == 2 else "C4"), answers
    if vcf.has_vcf(opp, me):                      # đối thủ có VCF nếu ta bỏ lượt
        safe = vcf.safe_moves(me, opp)
        empty = ~(me | opp) & FULL
        if safe and popcount(safe) < popcount(empty):
            return "C5", safe
    return None


def build() -> None:
    pool = []
    for source in SOURCES:
        for record in chung.games(source):
            moves = record["moves"]
            for ply in range(record["opening_plies"], len(moves)):
                pool.append((source, record["game_id"], ply, moves[:ply]))
    random.Random(chung.SEED).shuffle(pool)

    chosen: dict[str, list[dict]] = {name: [] for name in CATEGORIES}
    seen: set[tuple[int, int]] = set()
    used_games: set[tuple[str, str, int]] = set()
    scanned = 0
    for source, game_id, ply, moves in pool:
        if all(len(v) >= PER_CATEGORY for v in chosen.values()):
            break
        scanned += 1
        game = CaroGame.from_moves(moves)
        me, opp = game.me_opp()
        key = canonical(me, opp)
        if key in seen:
            continue
        found = classify(me, opp)
        if found is None:
            continue
        category, answers = found
        # Mỗi ván góp tối đa một thế cho mỗi loại, để một ván dài không chiếm cả tập.
        if len(chosen[category]) >= PER_CATEGORY or (category, source, game_id) in used_games:
            continue
        seen.add(key)
        used_games.add((category, source, game_id))
        line = vcf.solve(me, opp) if category != "C5" else vcf.solve(opp, me)
        chosen[category].append({
            "id": f"{category}-{len(chosen[category]) + 1:03d}", "category": category,
            "source": source, "game_id": game_id, "ply": ply, "moves": moves,
            "answers": list(iter_bits(answers)),
            "candidates": popcount(candidates(me, opp, 2)),
            "vcf_attacks": vcf.attacks_in(line) if line else 0,
            "line": list(line or ()),
        })
    positions = [p for v in chosen.values() for p in v]
    chung.save_json("bo-chien-thuat", {"seed": chung.SEED, "scanned": scanned, "pool": len(pool),
                                       "positions": positions})
    print({k: len(v) for k, v in chosen.items()}, "quét", scanned, "/", len(pool))


# ---------------------------------------------------------------------------
# Chấm tác tử
# ---------------------------------------------------------------------------

def base_agents() -> list[dict]:
    return load_config("experiments/rr8_heldout")["agents"]


def variants(group: str) -> list[tuple[str, dict]]:
    """(nhãn, config tác tử) cho từng nhóm thí nghiệm."""
    by_id = {a["id"]: a for a in base_agents()}

    def make(base: str, params: dict | None = None, limits: dict | None = None) -> dict:
        cfg = by_id[base]
        return {**cfg, "params": {**cfg.get("params", {}), **(params or {})},
                "limits": {**cfg["limits"], **(limits or {})}}

    if group == "base":
        return [(name, by_id[name]) for name in chung.AGENT_ORDER]
    if group == "scale":
        out = [(f"L3|depth={d}", make("L3-AlphaBeta", limits={"max_depth": d})) for d in (1, 2, 3, 4, 5, 6)]
        for rollout in ("random", "light"):
            out += [(f"L4-{rollout}|sims={s}", make("L4-MCTS", {"rollout": rollout}, {"max_sims": s}))
                    for s in (100, 400, 800, 1600, 6400)]
        out += [(f"L4-tactics|sims={s}",
                 make("L4-MCTS", {"rollout": "light", "root_tactics": True}, {"max_sims": s}))
                for s in (100, 800, 6400)]
        for name in ("L7-PUCT", "L8-AlphaZero"):
            out += [(f"{name[:2]}|sims={s}", make(name, limits={"max_sims": s}))
                    for s in (1, 50, 200, 800, 3200)]
        return out
    if group == "variants":
        out = []
        for name in ("L7-PUCT", "L8-AlphaZero"):
            tag = name[:2]
            out += [(f"{tag}|c_puct={c}", make(name, {"c_puct": c})) for c in (0.5, 1.0, 1.5, 3.0, 5.0)]
            out += [(f"{tag}|batch={b}", make(name, {"batch_leaves": b})) for b in (1, 4, 8, 16, 32)]
        out += [(f"L7|mix={m}", make("L7-PUCT", {"leaf_eval": "mix", "mix_lambda": m})) for m in (0.0, 0.5, 1.0)]
        out += [(f"L4|c_uct={c}", make("L4-MCTS", {"rollout": "light", "c_uct": c}))
                for c in (0.25, 0.7, 1.4, 2.8)]
        priorities = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
        out += [(f"L2|-{p}", make("L2-Rule", {"priorities": [q for q in priorities if q != p]}))
                for p in priorities[:6]]
        out += [("L3|no_forced", make("L3-AlphaBeta", {"forced_moves": False})),
                ("L3|no_defense", make("L3-AlphaBeta", {"eval_weights": "no_defense"})),
                ("L3|radius=1", make("L3-AlphaBeta", {"radius": 1}))]
        return out
    raise ValueError(group)


def solve_positions(job: tuple[str, dict, list[dict]]) -> tuple[str, list[dict]]:
    import torch

    torch.set_num_threads(1)
    label, cfg, positions = job
    agent = make_agent(cfg)
    results = []
    for index, position in enumerate(positions):
        game = CaroGame.from_moves(position["moves"])
        agent.reset(chung.SEED * 10_000 + index)
        limits = Limits.from_dict(cfg["limits"])
        start = time.perf_counter()
        move = agent.select_action(game.clone(), limits)
        elapsed = (time.perf_counter() - start) * 1000
        results.append({"id": position["id"], "move": int(move), "ok": int(move) in position["answers"],
                        "time_ms": round(elapsed, 2)})
    return label, results


def run(groups: list[str], workers: int) -> None:
    positions = chung.load_json("bo-chien-thuat")["positions"]
    context = mp.get_context("spawn")
    for group in groups:
        jobs = [(label, cfg, positions) for label, cfg in variants(group)]
        start = time.perf_counter()
        results = {}
        with context.Pool(min(workers, len(jobs))) as pool:
            for label, rows in pool.imap_unordered(solve_positions, jobs):
                results[label] = rows
                print(f"{label:<24} {sum(r['ok'] for r in rows):>4}/{len(rows)}  "
                      f"{time.perf_counter() - start:7.1f}s", flush=True)
        ordered = {label: results[label] for label, *_ in jobs}
        chung.save_json(f"chien-thuat-{group}", ordered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "run"])
    parser.add_argument("groups", nargs="*", default=["base"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "build":
        build()
    else:
        run(args.groups, args.workers)
