
import argparse
import concurrent.futures
import json
import os

import pandas as pd

from caro.agents import make_agent
from caro.arena.match import play_match
from caro.arena.openings import load_openings
from caro.arena.schedule import build_match_schedule
from caro.arena.stats import (
    calculate_performance,
    detailed_record_by_turn,
    elo,
    generate_matrix,
    summarize_self_play,
)
from caro.config import load_config
from caro.runlog import Run


def worker_play(match_info, max_plies):
    """Chạy độc lập một ván đấu trong worker."""

    game_id = match_info["game_id"]
    agent_x_cfg = match_info["agent_x_cfg"]
    agent_o_cfg = match_info["agent_o_cfg"]
    opening_id = match_info["opening_id"]
    opening_moves = match_info["opening_moves"]
    seed = match_info["seed"]

    peak_ram_mb = 0.0

    try:
        import resource
        ram_before = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
    except ImportError:
        ram_before = 0

    agent_x = make_agent(agent_x_cfg)
    agent_o = make_agent(agent_o_cfg)

    result = play_match(
        agent_x=agent_x,
        agent_o=agent_o,
        opening_moves=opening_moves,
        seed=seed,
        game_id=game_id,
        max_plies=max_plies,
    )

    try:
        import resource
        ram_after = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        peak_ram_mb = (ram_after - ram_before) / 1024.0
    except ImportError:
        peak_ram_mb = 0.0

    result.update({
        "game_id": game_id,
        "agent_x": agent_x.name,
        "agent_o": agent_o.name,
        "opening_id": opening_id,
        "opening_plies": len(opening_moves),
        "peak_ram_mb": round(peak_ram_mb, 2),
    })

    return result


def generate_tournament_report(
    game_records,
    pairs_mode,
    report_dir,
    arena_cfg,
):
    """Tạo các báo cáo thống kê sau khi giải đấu kết thúc."""

    report_dir.mkdir(parents=True, exist_ok=True)

    games_df = pd.DataFrame(game_records)

    report = {
        "detailed_record_by_turn": detailed_record_by_turn(games_df),
        "performance": calculate_performance(games_df),
    }

    stats_cfg = arena_cfg.get("stats", {})

    if pairs_mode == "self_play":
        report["self_play"] = summarize_self_play(
            games_df,
            n_boot=stats_cfg.get("bootstrap", 1000),
            confidence=stats_cfg.get("confidence", 0.95),
            seed=stats_cfg.get("seed", 42),
        )
    else:
        generate_matrix(games_df, report_dir)

        if stats_cfg.get("elo", True):
            elo_matches = [
                (
                    row["game_id"],
                    row["agent_x"],
                    row["agent_o"],
                    row["result"],
                )
                for row in game_records
            ]
            report["elo"] = elo(elo_matches)

    with (report_dir / "stats.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Chạy giải đấu Arena"
    )
    parser.add_argument(
        "config",
        help="Đường dẫn file config, ví dụ experiments/smoke",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Các tham số override dạng key=value",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    arena_cfg = cfg.get("arena", {})
    agents_cfg = cfg.get("agents", [])
    baselines_cfg = cfg.get("baselines", [])

    openings_cfg = arena_cfg.get("openings", {})
    openings = load_openings(openings_cfg)

    pairs_mode = arena_cfg.get(
        "pairs",
        arena_cfg.get("format", "round_robin"),
    )

    games_per_pair = arena_cfg.get("games_per_pair", 200)
    swap_colors = arena_cfg.get("swap_colors", True)
    seed = cfg.get("seed", 42)
    max_plies = arena_cfg.get("max_plies", 121)

    matches = build_match_schedule(
        agents=agents_cfg,
        baselines=baselines_cfg,
        openings=openings,
        pairs_mode=pairs_mode,
        games_per_pair=games_per_pair,
        swap_colors=swap_colors,
        seed=seed,
    )

    workers = arena_cfg.get(
        "workers",
        max(1, (os.cpu_count() or 2) - 1),
    )

    if not isinstance(workers, int) or workers <= 0:
        raise ValueError(
            "arena.workers phải là số nguyên dương."
        )

    scores = {}
    game_records = []

    with Run(cfg) as run:
        report_dir = run.dir / "reports"

        run.log.info(
            f"Bắt đầu giải đấu: {len(matches)} ván cờ, "
            f"{workers} workers, chế độ {pairs_mode}."
        )

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = {
                executor.submit(
                    worker_play,
                    match,
                    max_plies,
                ): match
                for match in matches
            }

            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                res["tournament_id"] = run.id

                agent_x = res["agent_x"]
                agent_o = res["agent_o"]

                scores.setdefault(agent_x, 0.0)
                scores.setdefault(agent_o, 0.0)

                if res["result"] == 1:
                    scores[agent_x] += 1.0
                elif res["result"] == -1:
                    scores[agent_o] += 1.0
                elif res["result"] == 0:
                    scores[agent_x] += 0.5
                    scores[agent_o] += 0.5

                game_records.append(dict(res))
                run.log_game(res)

        run.log.info(
            f"Bảng điểm chung cuộc: {scores}"
        )

        report = generate_tournament_report(
            game_records=game_records,
            pairs_mode=pairs_mode,
            report_dir=report_dir,
            arena_cfg=arena_cfg,
        )

        run.log.info(
            f"Đã lưu báo cáo thống kê tại: {report_dir}"
        )

        run.finish(
            status="ok",
            score=scores,
        )


if __name__ == "__main__":
    main()
