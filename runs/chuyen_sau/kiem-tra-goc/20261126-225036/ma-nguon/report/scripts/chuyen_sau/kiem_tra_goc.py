"""Kiểm tra nước tốt nhất của độ sâu trước có được duyệt đầu tiên ở gốc không, và nếu có thì tiết kiệm bao nhiêu nút.

    .venv/bin/python report/scripts/chuyen_sau/kiem_tra_goc.py

select_action đưa nước tốt nhất lên đầu danh sách `moves`, nhưng _negamax_root lại sắp xếp danh sách đó bằng
_order_moves. Script đo trên 50 thế trung cuộc (backend Python), so mã hiện tại với một lớp con chỉ đổi đúng một
điểm: ở gốc, phần tử đầu của `moves` (nước tốt nhất trước đó) được duyệt trước. Mã trong caro/ không bị sửa.
"""

from __future__ import annotations

import chung
from phan_tich_tim_kiem import midgame_positions

from caro.agents.alphabeta import AlphaBetaAgent
from caro.agents.base import Limits
from caro.engine import CaroGame

DEPTHS = (3, 4)


class Current(AlphaBetaAgent):
    def _order_moves(self, moves, me, opp, key, ply):
        out = super()._order_moves(moves, me, opp, key, ply)
        if ply == 0:
            self.root_log.append(out.index(moves[0]))
        return out


class PreviousFirst(AlphaBetaAgent):
    def _order_moves(self, moves, me, opp, key, ply):
        out = super()._order_moves(moves, me, opp, key, ply)
        if ply == 0 and self.root_log:                    # từ độ sâu 2
            out.remove(moves[0])
            out.insert(0, moves[0])
        if ply == 0:
            self.root_log.append(0)
        return out


def measure(cls, positions: list[list[int]], depth: int) -> list[dict]:
    rows = []
    for moves in positions:
        agent = cls(backend="python")
        agent.reset(0)
        agent.root_log = []
        move = agent.select_action(CaroGame.from_moves(moves), Limits(max_depth=depth))
        stats = agent.last_stats()
        rows.append({"move": int(move), "nodes": stats["nodes"], "score": stats["score"],
                     "time_ms": stats["time_ms"], "ranks": agent.root_log[1:]})
        print(".", end="", flush=True)
    print(f" {cls.__name__} d={depth}", flush=True)
    return rows


def main() -> None:
    positions = midgame_positions()
    result = []
    for depth in DEPTHS:
        current = measure(Current, positions, depth)
        fixed = measure(PreviousFirst, positions, depth)
        ranks = sorted(r for row in current for r in row["ranks"])
        item = {"depth": depth, "passes": len(ranks), "previous_first": sum(r == 0 for r in ranks),
                "rank_median": ranks[len(ranks) // 2], "rank_max": ranks[-1],
                "nodes_current": sum(r["nodes"] for r in current), "nodes_fixed": sum(r["nodes"] for r in fixed),
                "time_current": sum(r["time_ms"] for r in current), "time_fixed": sum(r["time_ms"] for r in fixed),
                "same_score": sum(a["score"] == b["score"] for a, b in zip(current, fixed, strict=True)),
                "same_move": sum(a["move"] == b["move"] for a, b in zip(current, fixed, strict=True)),
                "positions": len(positions), "current": current, "fixed": fixed}
        result.append(item)
        print({k: v for k, v in item.items() if k not in ("current", "fixed")}, flush=True)
    chung.save_json("kiem-tra-goc", result)


if __name__ == "__main__":
    main()
