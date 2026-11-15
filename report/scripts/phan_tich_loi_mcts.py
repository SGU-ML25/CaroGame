"""Phân tích lỗi của MCTS trong A3/A4: với mỗi ván MCTS thua, phân loại lỗi tại hai nước cuối của MCTS.

    uv run python report/scripts/phan_tich_loi_mcts.py [thư_mục_run ...]

Mọi phân loại dựa trên engine (winning_squares), không dựa vào đoán tay:
  - nước cuối của MCTS: đối thủ có k ô thắng ngay.  k=1 mà không chặn -> "bỏ sót chặn năm";
    k>=2 -> đã thua (tứ hở/đòn kép), tìm lỗi ở nước trước đó.
  - nước kế cuối (đối thủ chưa có ô thắng): có nước của đối thủ tạo >=2 ô thắng ("đe doạ kép")
    mà MCTS không ngăn -> "bỏ sót chặn ba hở/đòn kép"; ngược lại "khác".
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from caro.engine.bitboard import iter_bits, winning_squares
from caro.engine.game import CaroGame

GOC = Path(__file__).resolve().parents[2]
RUNS = [GOC / "runs/tournament/a3_mcts_sims/latest", GOC / "runs/tournament/a4_rollout/latest"]


def doi_thu_co_don_kep(g: CaroGame) -> bool:
    """g: lượt của MCTS. Đối thủ (đi sau) có nước tạo >=2 ô thắng ngay không (nếu MCTS bỏ qua)?"""
    opp, me = g.bb[g.side ^ 1], g.bb[g.side]
    for m in iter_bits(~(me | opp) & ((1 << 121) - 1)):
        if winning_squares(opp | 1 << m, me).bit_count() >= 2:
            return True
    return False


def phan_loai(moves: list[int], mcts_side: int) -> tuple[str, str]:
    g = CaroGame()
    states = []  # (side to move, bb copy) trước mỗi nước
    for m in moves:
        states.append((g.side, g.bb[0], g.bb[1], m))
        g.step(m)
    mc = [i for i, s in enumerate(states) if s[0] == mcts_side]
    last = states[mc[-1]]
    side, bx, bo, m = last
    me, opp = (bx, bo) if side == 0 else (bo, bx)
    k = winning_squares(opp, me).bit_count()
    if k == 1:
        return "bỏ sót chặn năm (nước cuối)", "k=1"
    if k == 0:
        return "khác", "k=0"
    # k>=2: xét nước kế cuối của MCTS
    if len(mc) < 2:
        return "thua sớm (<2 nước)", ""
    g2 = CaroGame.from_moves(moves[: mc[-2]])
    if winning_squares(g2.bb[g2.side ^ 1], g2.bb[g2.side]).bit_count() >= 1:
        return "bỏ sót chặn năm (nước kế cuối)", ""
    if doi_thu_co_don_kep(g2):
        return "bỏ sót chặn ba hở / đòn kép", ""
    return "khác", ""


def main() -> None:
    runs = [Path(a) for a in sys.argv[1:]] or RUNS
    tong = defaultdict(Counter)
    do_dai = defaultdict(list)
    for r in runs:
        with open(r / "games.jsonl") as f:
            games = [json.loads(line) for line in f]
        for gm in games:
            ten = {0: gm["agent_x"], 1: gm["agent_o"]}
            for s in (0, 1):
                if not ten[s].startswith("MCTS"):
                    continue
                thua = gm["result"] == (-1 if s == 0 else 1)
                if thua:
                    loai, _ = phan_loai(gm["moves"], s)
                    tong[(r.parent.name, ten[s])][loai] += 1
                    do_dai[(r.parent.name, ten[s])].append(len(gm["moves"]))
    for k, c in sorted(tong.items()):
        n = sum(c.values())
        print(k, n, "TB nước:", round(sum(do_dai[k]) / n, 1))
        for loai, v in c.most_common():
            print(f"   {loai:42s} {v:4d}  {v / n:6.1%}")


if __name__ == "__main__":
    main()
