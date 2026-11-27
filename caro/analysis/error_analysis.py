"""Phân tích lỗi chiến thuật của mọi tác tử từ games.jsonl (issue #12).

Mỗi nước sau chuỗi khai cuộc được xét bằng bitboard, không dùng hàm lượng giá hay ngưỡng điểm. Theo thứ tự:

- thang_ngay    : bên đi có ô thắng ngay. Lỗi nếu không đi vào một ô đó.
- chan_nam      : đối thủ có đúng một ô thắng ngay. Lỗi nếu không chặn.
- chan_don_kep  : đối thủ có nước tạo từ hai ô thắng (tứ hở, đòn kép) và bên đi còn nước đỡ được.
                  Lỗi nếu sau nước đã đi, đối thủ vẫn còn nước như vậy (thua ép trong 4 ply).
- da_thua       : đối thủ có từ hai ô thắng, hoặc đe doạ kép không còn nước đỡ. Không tính lỗi ở nước này.

Mỗi loại đếm cả số cơ hội và số lần sai. Các chuỗi ép dài hơn 4 ply (VCF, VCT) không được xét.

    uv run python -m caro.analysis.error_analysis                 # sáu lượt giải chính thức
    uv run python -m caro.analysis.error_analysis <thư mục giải hoặc lượt chạy> ...

Kết quả ghi vào runs/analysis/error_analysis/<thời điểm>/: summary.json, errors.jsonl, meta.json, run.log.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from caro.config import PROJECT_ROOT, load_config, project_path
from caro.engine.bitboard import (
    FULL,
    LINES,
    WIN,
    exactly,
    iter_bits,
    popcount,
    window_counts,
    winning_squares,
)
from caro.engine.game import CaroGame
from caro.runlog import JsonlWriter, Run

CATEGORIES = ("thang_ngay", "chan_nam", "chan_don_kep")
LABELS = {"thang_ngay": "bỏ sót thắng ngay", "chan_nam": "bỏ sót chặn năm",
          "chan_don_kep": "bỏ sót chặn đòn kép"}
LOST = "da_thua"

# Lượt chạy giải chính thức L1–L8 ở hai chế độ ngân sách.
DEFAULT_RUNS = tuple(f"runs/tournament/main_mode_{mode}_{part}"
                     for mode in "ab" for part in ("l1_l6", "l7_additional", "l8_additional"))


def four_squares(me: int, opp: int) -> int:
    """Các ô trống mà 'me' đặt vào sẽ tạo thêm một cửa sổ 4 quân còn sống (tức tạo tứ)."""
    empty = ~(me | opp) & FULL
    res = 0
    for d, start in LINES:
        w3 = exactly(3, *window_counts(me, opp, d, start))
        for j in range(WIN):
            res |= (w3 << (j * d)) & empty
    return res


def double_threats(attacker: int, defender: int) -> int:
    """Các ô mà attacker đi vào thì thắng ép: defender đến lượt không có ô thắng và attacker có ≥ 2 ô thắng.

    Chỉ gọi khi attacker chưa có ô thắng ngay: lúc đó mọi nước tạo hai ô thắng đều là nước tạo tứ.
    """
    res = 0
    for s in iter_bits(four_squares(attacker, defender)):
        after = attacker | 1 << s
        if popcount(winning_squares(after, defender)) >= 2 and not winning_squares(defender, after):
            res |= 1 << s
    return res


def safe_squares(me: int, opp: int, threats: int) -> int:
    """Các nước của 'me' khiến 'opp' không còn đe doạ kép nào.

    Một nước chỉ phá được đe doạ s khi chiếm s, chiếm một ô thắng mà s tạo ra, hoặc tự tạo tứ để giành
    quyền đi trước; nên chỉ cần thử các ô đó.
    """
    options = threats | four_squares(me, opp)
    for s in iter_bits(threats):
        options |= winning_squares(opp | 1 << s, me)
    res = 0
    for m in iter_bits(options):
        if not double_threats(opp, me | 1 << m):
            res |= 1 << m
    return res


def classify_move(me: int, opp: int, move: int) -> tuple[str, bool, int] | None:
    """Xét nước `move` của 'me'. Trả (loại, có sai không, các ô đúng) hoặc None nếu thế không có đe doạ."""
    wins = winning_squares(me, opp)
    if wins:
        return "thang_ngay", not wins >> move & 1, wins
    blocks = winning_squares(opp, me)
    if popcount(blocks) >= 2:
        return LOST, False, 0
    if blocks:
        return "chan_nam", not blocks >> move & 1, blocks
    threats = double_threats(opp, me)
    if not threats:
        return None
    safe = safe_squares(me, opp, threats)
    if not safe:
        return LOST, False, 0
    return "chan_don_kep", not safe >> move & 1, safe


def opening_length(record: dict) -> int:
    if "opening_plies" in record:
        return record["opening_plies"]
    return len(record["moves"]) - len(record["time_ms"]) if record.get("time_ms") else 0


def analyze_game(record: dict) -> tuple[dict[str, Counter], list[dict]]:
    """Trả (bộ đếm theo tác tử, danh sách lỗi) của một ván; bỏ qua các nước khai cuộc."""
    counts: dict[str, Counter] = {}
    errors: list[dict] = []
    first = opening_length(record)
    names = (record["agent_x"], record["agent_o"])
    game = CaroGame()
    for ply, move in enumerate(record["moves"]):
        side = game.side
        if ply >= first:
            me, opp = game.me_opp()
            tally = counts.setdefault(names[side], Counter())
            tally["moves"] += 1
            found = classify_move(me, opp, move)
            if found:
                category, wrong, correct = found
                tally[category] += 1
                if wrong:
                    tally[category + "_sai"] += 1
                    result = record["result"] if side == 0 else -record["result"]
                    errors.append({
                        "game_id": record["game_id"], "opening_id": record.get("opening_id"), "ply": ply,
                        "side": side, "agent": names[side], "opponent": names[side ^ 1], "move": move,
                        "category": category, "correct": list(iter_bits(correct)), "result_for_agent": result,
                    })
        game.step(move)
    return counts, errors


def find_games(path: Path) -> Path:
    """games.jsonl của lượt chạy: chính thư mục đó, hoặc lượt mới nhất có log ván bên trong."""
    if (path / "games.jsonl").is_file():
        return path / "games.jsonl"
    found = sorted(p for p in path.glob("*/games.jsonl")
                   if not p.parent.is_symlink() and p.parent.name != "latest" and p.stat().st_size)
    if not found:
        raise FileNotFoundError(f"không tìm thấy games.jsonl trong {path}")
    return found[-1]


def summarise(counts: dict[str, Counter]) -> dict:
    out = {}
    for agent, c in sorted(counts.items()):
        row = {"moves": c["moves"], LOST: c[LOST]}
        for category in CATEGORIES:
            row[category] = {"chances": c[category], "errors": c[category + "_sai"]}
        out[agent] = row
    return out


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def run_analysis(paths: list[str], workers: int = 1, overrides: list[str] | tuple[str, ...] = ()) -> Path:
    cfg = load_config("base", ["experiment.kind=analysis", "experiment.name=error_analysis", *overrides])
    sources = [find_games(project_path(p)) for p in paths]
    with Run(cfg) as run:
        writer = JsonlWriter(run.dir / "errors.jsonl", flush_every=100)
        total: dict[str, Counter] = {}
        per_run = []
        with ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext() as pool:
            for source in sources:
                records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
                           if line.strip()]
                counts: dict[str, Counter] = {}
                found = 0
                results = pool.map(analyze_game, records, chunksize=20) if pool else map(analyze_game, records)
                for game_counts, errors in results:
                    for agent, c in game_counts.items():
                        counts.setdefault(agent, Counter()).update(c)
                    for error in errors:
                        writer.write({"source": relative(source.parent), **error})
                    found += len(errors)
                for agent, c in counts.items():
                    total.setdefault(agent, Counter()).update(c)
                per_run.append({"source": relative(source.parent), "games": len(records), "errors": found,
                                "by_agent": summarise(counts)})
                run.log.info("%s: %d ván, %d lỗi", relative(source.parent), len(records), found)
        writer.close()
        run.finish(total_games=sum(r["games"] for r in per_run), total_errors=writer.count,
                   by_agent=summarise(total), runs=per_run)
    return run.dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Phân tích lỗi chiến thuật từ games.jsonl")
    ap.add_argument("paths", nargs="*", help="thư mục giải hoặc lượt chạy (mặc định: sáu lượt giải chính thức)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--set", action="append", default=[], dest="overrides", metavar="KHOÁ=GIÁ_TRỊ",
                    help="ghi đè config, ví dụ run.root=runs/scratch")
    args = ap.parse_args()
    run_analysis(args.paths or list(DEFAULT_RUNS), args.workers, args.overrides)


if __name__ == "__main__":
    main()
