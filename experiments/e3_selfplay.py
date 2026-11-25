"""E3 — mỗi cấp tự đấu với chính nó trên cùng bộ khai cuộc, đo lợi thế đi trước.

    uv run python experiments/e3_selfplay.py [--openings 50] [--workers 4]

Mỗi cấp chơi 50 khai cuộc × 2 màu = 100 ván. Hai bên dùng hai thực thể tác tử độc lập
(cùng trọng số, khác RNG/bảng chuyển vị) để phép đo không bị nhiễu bởi trạng thái dùng chung.
Kết quả ghi theo schema chuẩn của arena vào runs/eval/tournament/e3_selfplay/<timestamp>/.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import time
from pathlib import Path

from caro.agents import make_agent
from caro.arena.match import play_match
from caro.config import load_config, project_path
from caro.runlog import Run

L8_BEST = "runs/train/alphazero/20261124-160615_gpu-native/checkpoints/best.pt"

SPECS: dict[str, object] = {
    "L1-Random": "agents/l1_random",
    "L2-Rule": "agents/l2_rule",
    "L3-AlphaBeta": {"from": "agents/l3_alphabeta", "budget": {"max_depth": 4}, "params": {"backend": "native"}},
    "L4-MCTS": {"from": "agents/l4_mcts", "budget": {"max_sims": 400}},
    "L5-PolicySL": "agents/l5_policy_sl",
    "L6-PolicyRL": "agents/l6_policy_rl",
    "L7-PUCT": {"from": "agents/l7_puct", "budget": {"max_sims": 200}, "params": {"backend": "native"}},
    "L8-AlphaZero": {
        "from": "agents/l8_alphazero",
        "id": "L8-AlphaZero",
        "budget": {"max_sims": 200},
        "params": {"checkpoint": L8_BEST, "backend": "native"},
    },
}


def play_one(task: tuple[str, int, int, list[int], int, int]) -> dict:
    """Một ván tự đấu. `first_x` True: bản A cầm X, bản B cầm O."""
    level, game_id, opening_id, moves, seed, first_x = task
    a = make_agent(SPECS[level])
    b = make_agent(SPECS[level])
    x, o = (a, b) if first_x else (b, a)
    t0 = time.perf_counter()
    record = play_match(x, o, moves, seed, game_id)
    elapsed = time.perf_counter() - t0
    record.update({
        "game_id": game_id,
        "tournament_id": "e3_selfplay",
        "agent_x": level,
        "agent_o": level,
        "opening_id": opening_id,
        "selfplay": True,
        "wall_s": round(elapsed, 3),
    })
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="experiments/e3_selfplay")
    ap.add_argument("--openings", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--levels", nargs="*", choices=list(SPECS), default=None,
                    help="chỉ chạy các cấp này (mặc định tất cả)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_openings = args.openings or int(cfg["arena"].get("num_openings", 50))
    workers = args.workers or int(cfg["arena"].get("workers", 4))
    openings = json.loads(project_path("runs/openings/openings_v1.json").read_text(encoding="utf-8"))
    openings = openings[:n_openings]
    seed = int(cfg.get("seed", 2026))
    levels = args.levels or list(SPECS)

    tasks = []
    game_id = 0
    for level in levels:
        for op_id, moves in enumerate(openings):
            for first_x in (True, False):
                tasks.append((level, game_id, op_id, moves, seed, first_x))
                game_id += 1

    print(f"E3: {len(tasks)} ván, {len(levels)} cấp, {n_openings} khai cuộc, {workers} workers", flush=True)
    t0 = time.time()
    with Run(cfg) as run:
        results: list[dict] = []
        with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            for k, rec in enumerate(ex.map(play_one, tasks, chunksize=1), 1):
                run.log_game(rec)
                results.append(rec)
                if k % 50 == 0:
                    print(f"{k}/{len(tasks)} ván, {time.time() - t0:.0f}s", flush=True)

        per_level = {}
        for level in levels:
            rows = [r for r in results if r["agent_x"] == level]
            xw = sum(r["result"] == 1 for r in rows)
            ow = sum(r["result"] == -1 for r in rows)
            dr = sum(r["result"] == 0 for r in rows)
            n = len(rows)
            per_level[level] = {
                "n": n, "x_wins": xw, "o_wins": ow, "draws": dr,
                "x_rate": (xw + 0.5 * dr) / n if n else None,
                "o_rate": (ow + 0.5 * dr) / n if n else None,
                "advantage": (xw - ow) / n if n else None,
                "plies_mean": sum(len(r["moves"]) for r in rows) / n if n else None,
            }
            run.log.info("E3 %s: X %d / O %d / hoà %d trên %d ván", level, xw, ow, dr, n)
        run.finish("ok", elapsed_s=round(time.time() - t0, 1), per_level=per_level)
    out = Path("/tmp/caro-eval/e3_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(per_level, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(per_level, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
