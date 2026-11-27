"""Phân tích sâu vài ván: nước quyết định, vì sao thắng, sai ở đâu. Ghi chuyen-sau/data/phan-tich-van-sau.json.

    .venv/bin/python report/scripts/chuyen_sau/phan_tich_van_sau.py [--workers 14]

Chỉ phân tích bốn ván báo cáo đã dùng (CHOSEN); không chơi ván mới.

Hai trọng tài:
  * vct.solve (độ sâu 2, mở rộng 3 khi lùi từ điểm quyết định): đường thắng ép có chứng minh;
  * KataGomo: tỷ lệ thắng theo góc nhìn bên đang đi và nước engine đề nghị (ước lượng, không phải chứng minh).
Trường "mistake" là nước của bên thua mà sau đó bên thắng có đường thắng ép, trong khi còn nước khác không để lộ
đường thắng ép trong cùng độ sâu (kiểm trên mọi ô trống). Đó mới là ứng viên: báo cáo chỉ gọi là sai lầm quyết định
khi một nước thay thế vượt qua kiem_tra_thay_the.py (xem bang_van_sau.saving_moves).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import time

import chung
import vct

from caro.agents.base import Limits
from caro.agents.katagomo import KataGomoAgent
from caro.engine import CaroGame
from caro.engine.bitboard import DIRECTIONS, FULL, N, has_five, iter_bits, popcount, winning_squares

RR8_LOG = chung.RUNS["rr8"] + "/games.jsonl"
DUEL_LOG = "runs/train/alphazero/20261124-160615_gpu-native/games.jsonl"
L3_LOG = "runs/eval/tournament/l8_vs_l3/20261125-134954_round37/games.jsonl"
# Bốn ván báo cáo đã dùng: ván 13 và 26 của mục ván minh hoạ, ván 1202 và 2726 của phụ lục ván đại diện.
CHOSEN = ((DUEL_LOG, 13), (L3_LOG, 26), (RR8_LOG, 1202), (RR8_LOG, 2726))
DEPTH, DEEP = 2, 3
VISITS = 1600
DIRECTION_NAMES = ("ngang", "dọc", "chéo xuôi", "chéo ngược")


def short(name: str) -> str:
    return "L3" if "AlphaBeta" in name else name.split("-")[0]


def select() -> list[dict]:
    chosen = []
    for log, game_id in CHOSEN:
        # Log huấn luyện L8 chứa cả ván tự đấu và ván cổng cùng mã; ván minh hoạ là ván đánh giá gặp L7.
        found = [r for r in chung.games(log) if r["game_id"] == game_id
                 and (log != DUEL_LOG or "PUCT" in str(r["tournament_id"]))]
        assert len(found) == 1, (log, game_id, len(found))
        record = found[0]
        x, o = short(record["agent_x"]), short(record["agent_o"])
        chosen.append({"log": log, "game_id": game_id, "x": x, "o": o, "moves": record["moves"],
                       "result": record["result"], "winner": x if record["result"] == 1 else o,
                       "opening_plies": len(record["moves"]) - len(record["time_ms"])})
    return chosen


def position(moves: list[int], ply: int) -> tuple[int, int]:
    """(bên đi, đối thủ) trước nước thứ ply (đếm từ 0)."""
    x, o = chung.replay(moves, ply)
    return (x, o) if ply % 2 == 0 else (o, x)


# ---------------------------------------------------------------- hình thế của một nước
def windows_through(cell: int, direction: tuple[int, int]) -> list[list[int]]:
    r, c = divmod(cell, N)
    dr, dc = direction
    out = []
    for j in range(5):
        cells = [(r + (k - j) * dr, c + (k - j) * dc) for k in range(5)]
        if all(0 <= a < N and 0 <= b < N for a, b in cells):
            out.append([a * N + b for a, b in cells])
    return out


def fours_along(me: int, opp: int, windows: list[list[int]]) -> set[int]:
    """Ô thắng sinh bởi các cửa sổ sống có đúng 4 quân."""
    squares = set()
    for w in windows:
        if not any(opp >> i & 1 for i in w) and sum(me >> i & 1 for i in w) == 4:
            squares |= {i for i in w if not me >> i & 1}
    return squares


def shapes(me: int, opp: int, move: int) -> list[dict]:
    """Hình mà nước `move` tạo theo từng hướng: five, open_four, four, open_three."""
    after = me | 1 << move
    found = []
    for name, direction in zip(DIRECTION_NAMES, DIRECTIONS, strict=True):
        windows = windows_through(move, direction)
        live = [w for w in windows if not any(opp >> i & 1 for i in w)]
        if any(all(after >> i & 1 for i in w) for w in live):
            found.append({"dir": name, "kind": "five", "cells": []})
            continue
        squares = fours_along(after, opp, windows)
        if squares:
            found.append({"dir": name, "kind": "open_four" if len(squares) > 1 else "four", "cells": sorted(squares)})
            continue
        makers = sorted({e for w in live if sum(after >> i & 1 for i in w) == 3 for e in w
                         if not after >> e & 1 and len(fours_along(after | 1 << e, opp, windows)) > 1})
        if makers:
            found.append({"dir": name, "kind": "open_three", "cells": makers})
    return found


def blocked(me: int, opp: int, move: int) -> str | None:
    """Nước này đỡ đe doạ nào của đối thủ: ô thắng ngay, hoặc ô tạo đòn kép."""
    if winning_squares(opp, me) >> move & 1:
        return "five"
    if popcount(winning_squares(opp | 1 << move, me)) >= 2:
        return "double"
    return None


# ---------------------------------------------------------------- bộ giải chạy song song
def prove(job: tuple[int, int, int]) -> dict | None:
    me, opp, depth = job
    tree = vct.solve(me, opp, depth)
    if tree is None:
        return None
    line = vct.main_line(tree)
    return {"move": tree["move"], "depth": tree["depth"], "line": line,
            "kind": "VCF" if "vcf" in tree else "VCT",
            "replies": sorted(tree.get("replies", {}))}


def safe_reply(job: tuple[int, int, int, int]) -> tuple[int, bool]:
    """Bên `me` đi `move`; trả True nếu sau đó đối thủ KHÔNG có đường thắng ép trong độ sâu."""
    me, opp, move, depth = job
    after = me | 1 << move
    if has_five(after):
        return move, True
    return move, vct.solve(opp, after, depth) is None


def safe_moves(pool, me: int, opp: int, depth: int) -> list[int]:
    jobs = [(me, opp, m, depth) for m in iter_bits(~(me | opp) & FULL)]
    return sorted(m for m, ok in pool.map(safe_reply, jobs, chunksize=1) if ok)


# ---------------------------------------------------------------- KataGomo
def katago(games: list[dict]) -> None:
    engine = KataGomoAgent(max_visits=VISITS, threads=8)
    limits = Limits(max_sims=VISITS)
    pattern = re.compile(r"Visits (\d+).*?Winrate ([\d.]+)%(?:.*?Drawrate ([\d.]+)%)?")
    try:
        for g in games:
            engine.reset(chung.SEED)
            g["engine"] = []
            for ply in range(g["opening_plies"], len(g["moves"])):
                state = CaroGame.from_moves(g["moves"][:ply])
                best = engine.select_action(state, limits)
                stats = engine.last_stats()
                match = pattern.search(engine._last_chat)
                g["engine"].append({"ply": ply, "best": int(best), "winrate": stats.get("value"),
                                    "draw": float(match.group(3)) / 100 if match and match.group(3) else None,
                                    "visits": stats.get("visits"), "pv": [int(m) for m in stats.get("pv", [])][:6]})
            print("KataGomo xong ván", g["game_id"], flush=True)
    finally:
        engine.close()


# ---------------------------------------------------------------- phân tích
def analyse(pool, g: dict) -> None:
    moves, start = g["moves"], g["opening_plies"]
    winner = 0 if g["result"] == 1 else 1
    plies = list(range(start, len(moves)))
    print("ván", g["game_id"], "chứng minh từng nước", flush=True)
    proofs = pool.map(prove, [(*position(moves, p), DEPTH) for p in plies], chunksize=1)
    g["plies"] = []
    for ply, proof in zip(plies, proofs, strict=True):
        me, opp = position(moves, ply)
        move = moves[ply]
        g["plies"].append({"ply": ply, "side": ply % 2, "move": move, "proof": proof,
                           "shapes": shapes(me, opp, move), "blocked": blocked(me, opp, move)})
    by_ply = {p["ply"]: p for p in g["plies"]}

    winner_turns = [p for p in plies if p % 2 == winner]
    k = winner_turns[-1]
    for ply in reversed(winner_turns):                    # điểm đầu tiên mà từ đó bên thắng luôn giữ đường thắng ép
        if by_ply[ply]["proof"] is None:
            break
        k = ply
    while k - 2 >= start:                                 # lùi thêm bằng độ sâu lớn hơn
        print("  lùi tới nước", k - 1, "độ sâu", DEEP, flush=True)
        deeper = pool.apply(prove, ((*position(moves, k - 2), DEEP),))
        if deeper is None:
            break
        by_ply[k - 2]["proof"] = deeper
        k -= 2

    decisive = None
    forced_chain = []
    j = k - 1
    depth = max(DEPTH, by_ply[k]["proof"]["depth"])
    while j >= start:                                     # nước gần nhất của bên thua còn đường tránh
        me, opp = position(moves, j)
        print("  tìm nước an toàn ở nước", j + 1, "độ sâu", depth, flush=True)
        safe = safe_moves(pool, me, opp, depth)
        if safe and moves[j] not in safe:
            decisive = {"ply": j, "played": moves[j], "safe": safe, "depth": depth,
                        "empty": 121 - popcount(me | opp)}
            break
        forced_chain.append({"ply": j, "safe": safe})
        if safe:                                          # nước đã đi nằm trong tập an toàn: bộ giải không kết luận
            break
        j -= 2
    g["turning"] = {"first_proven": k, "proof": by_ply[k]["proof"], "mistake": decisive, "no_escape": forced_chain}

    missed = []
    for p in g["plies"]:
        after = by_ply.get(p["ply"] + 2)
        if p["proof"] and after is not None and after["proof"] is None:
            missed.append(p["ply"])
    g["missed_wins"] = missed
    print(f"ván {g['game_id']}: thắng ép từ nước {k + 1}; sai lầm "
          f"{decisive['ply'] + 1 if decisive else None}; số nước an toàn "
          f"{len(decisive['safe']) if decisive else 0}; bỏ lỡ {[m + 1 for m in missed]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    games = select()
    print([(g["game_id"], g["x"], g["o"], g["winner"], len(g["moves"]), g["opening_plies"]) for g in games],
          flush=True)
    start = time.perf_counter()
    katago(games)
    print(f"KataGomo {time.perf_counter() - start:.0f}s", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for g in games:
            analyse(pool, g)
            print(f"  đã chạy {time.perf_counter() - start:.0f}s", flush=True)
            chung.save_json("phan-tich-van-sau", {"visits": VISITS, "depth": DEPTH, "deep": DEEP, "games": games})


if __name__ == "__main__":
    main()
