"""Benchmark tác tử trên bộ thế cờ cố định — không đấu, chỉ đo số nút / thời gian (mục 05.3, 10.3).

Chạy:
    python -m caro.arena.bench experiments/a1_ab_pruning
    python -m caro.arena.bench experiments/a2_ordering_tt
    python -m caro.arena.bench experiments/a7_eval_weights
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

from caro.agents import make_agent
from caro.config import load_config, project_path
from caro.engine import CaroGame
from caro.runlog import Run, derive_seed


# ---------------------------------------------------------------------------
# Sinh / nạp thế cờ giữa ván
# ---------------------------------------------------------------------------

def _load_positions(source: str, count: int, seed: int) -> list[list[int]]:
    """Nạp thế cờ từ file JSON hoặc sinh ngẫu nhiên nếu file chưa có.

    Mỗi thế là một list nước đi (moves) tái lập CaroGame.
    """
    path = project_path(source)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        positions = data if isinstance(data, list) else data.get("positions", [])
        return positions[:count]

    # Sinh ngẫu nhiên: chơi ván ngẫu nhiên, lấy thế sau 10-20 nước
    import random
    rng = random.Random(seed)
    positions = []
    while len(positions) < count:
        game = CaroGame()
        n_moves = rng.randint(10, 20)
        for _ in range(n_moves):
            actions = game.legal_actions()
            if not actions or game.is_terminal():
                break
            game.step(rng.choice(actions))
        if not game.is_terminal() and len(game.history) >= 6:
            positions.append(list(game.history))
    return positions[:count]


def _save_positions(positions: list[list[int]], path: str) -> None:
    p = project_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"positions": positions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Benchmark một tác tử trên một thế cờ
# ---------------------------------------------------------------------------

def _bench_one(agent: Any, moves: list[int], limits: Any, repeat: int) -> dict:
    """Chạy agent trên thế cờ `moves`, lặp `repeat` lần, trả thống kê."""
    nodes_list: list[int] = []
    time_list: list[float] = []
    best_move_list: list[int] = []
    score_list: list[int] = []

    for _ in range(repeat):
        game = CaroGame.from_moves(moves)
        if game.is_terminal():
            continue
        agent.reset(0)
        action = agent.select_action(game, limits)
        stats = agent.last_stats()
        nodes_list.append(stats.get("nodes", 0))
        time_list.append(stats.get("time_ms", 0.0))
        best_move_list.append(action)
        score_list.append(stats.get("score", 0))

    if not nodes_list:
        return {}

    return {
        "nodes_median": statistics.median(nodes_list),
        "nodes_mean":   round(statistics.mean(nodes_list)),
        "time_ms_median": round(statistics.median(time_list), 1),
        "time_ms_p95":    round(sorted(time_list)[int(len(time_list) * 0.95)], 1),
        "best_move":      statistics.mode(best_move_list),
        "root_value":     statistics.median(score_list),
        "repeat":         len(nodes_list),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bench(cfg: dict) -> None:
    bench_cfg = cfg.get("bench", {})
    pos_cfg   = bench_cfg.get("positions", {})
    repeat    = bench_cfg.get("repeat", 3)

    source = pos_cfg.get("source", "runs/openings/midgame_positions_v1.json")
    count  = pos_cfg.get("count", 50)
    seed   = pos_cfg.get("seed", cfg.get("seed", 2026))

    positions = _load_positions(source, count, seed)

    # Lưu lại nếu vừa sinh (để commit)
    if not project_path(source).is_file():
        _save_positions(positions, source)

    agents_cfg = cfg.get("agents", [])
    budget     = cfg.get("budget", {})

    with Run(cfg) as run:
        run.log.info("bench %d thế × %d lần × %d tác tử",
                     len(positions), repeat, len(agents_cfg))

        # Kết quả tổng hợp: agent_id → list kết quả mỗi thế
        summary: dict[str, list[dict]] = {a["id"]: [] for a in agents_cfg}

        for pos_idx, moves in enumerate(positions):
            game = CaroGame.from_moves(moves)
            if game.is_terminal():
                run.log.warning("thế %d đã kết thúc, bỏ qua", pos_idx)
                continue

            row_results: dict[str, dict] = {}

            for agent_cfg in agents_cfg:
                from caro.agents.base import Limits
                limits = Limits.from_dict(agent_cfg.get("limits") or budget)
                agent  = make_agent(agent_cfg)
                result = _bench_one(agent, moves, limits, repeat)
                if not result:
                    continue
                result["pos_idx"] = pos_idx
                result["agent"]   = agent_cfg["id"]
                summary[agent_cfg["id"]].append(result)
                row_results[agent_cfg["id"]] = result

                run.log.info(
                    "  thế %3d | %-15s | nodes_med=%7d | time_med=%6.1f ms | move=%d | score=%d",
                    pos_idx, agent_cfg["id"],
                    result["nodes_median"], result["time_ms_median"],
                    result["best_move"],    result["root_value"],
                )

            # Ghi metrics mỗi thế
            metric_row: dict[str, Any] = {"pos_idx": pos_idx, "n_moves": len(moves)}
            for agent_cfg in agents_cfg:
                aid = agent_cfg["id"]
                if aid in row_results:
                    metric_row[f"{aid}_nodes"] = row_results[aid]["nodes_median"]
                    metric_row[f"{aid}_time_ms"] = row_results[aid]["time_ms_median"]
                    metric_row[f"{aid}_score"] = row_results[aid]["root_value"]
            run.log_metric(pos_idx, **metric_row)

        # Bảng tổng hợp cuối
        run.log.info("\n%s", "=" * 70)
        run.log.info("%-15s %10s %10s %10s %8s", "Agent", "nodes_med", "time_med", "time_p95", "%same_move")

        agg: dict[str, dict] = {}
        for agent_cfg in agents_cfg:
            aid = agent_cfg["id"]
            rows = summary[aid]
            if not rows:
                continue
            nodes_all = [r["nodes_median"] for r in rows]
            time_all  = [r["time_ms_median"] for r in rows]
            time_all_p95 = sorted(time_all)[int(len(time_all) * 0.95)]
            agg[aid] = {
                "nodes_median": round(statistics.median(nodes_all)),
                "time_ms_median": round(statistics.median(time_all), 1),
                "time_ms_p95": round(time_all_p95, 1),
            }
            run.log.info("%-15s %10d %9.1f ms %9.1f ms",
                         aid, agg[aid]["nodes_median"],
                         agg[aid]["time_ms_median"], agg[aid]["time_ms_p95"])

        # Kiểm tra nhất quán giá trị gốc (A1): pruning vs no-pruning phải cùng score
        ids = list(agg.keys())
        if len(ids) == 2:
            scores_a = [r["root_value"] for r in summary[ids[0]]]
            scores_b = [r["root_value"] for r in summary[ids[1]]]
            same = sum(a == b for a, b in zip(scores_a, scores_b))
            pct  = same / len(scores_a) * 100 if scores_a else 0
            run.log.info("\nGiá trị gốc trùng nhau: %d/%d (%.1f%%)", same, len(scores_a), pct)
            agg["consistency_pct"] = pct
            if pct < 100:
                run.log.warning("⚠️  Giá trị gốc KHÁC NHAU → có thể là bug!")
            else:
                run.log.info("✅ Giá trị gốc trùng 100%% — nhất quán")

        run.finish("ok", agents=list(agg.keys()), n_positions=len(positions), **agg)


def main() -> None:
    if len(sys.argv) < 2:
        print("Dùng: python -m caro.arena.bench <config>  [key=value ...]")
        sys.exit(1)
    name      = sys.argv[1]
    overrides = sys.argv[2:]
    cfg = load_config(name, overrides)
    run_bench(cfg)


if __name__ == "__main__":
    main()