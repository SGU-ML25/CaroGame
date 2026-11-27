"""Vì sao MCTS rollout light chọn sai ở thế phải chặn (loại C2 của bộ chiến thuật).

    .venv/bin/python report/scripts/chuyen_sau/kiem_tra_rollout.py

Với mỗi thế C2 (đối thủ có đúng một ô thắng w) script đo: (1) giá trị rollout light trực tiếp sau nước chặn và sau một
nước không chặn; (2) khi MCTS chọn một nước không chặn, nút đó đã mở rộng bao nhiêu đáp trả và có w trong số đó không.
Không sửa mã trong caro/.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import statistics

import chung

from caro.agents.base import Limits
from caro.agents.mcts import MCTSAgent, candidate_moves, rollout_fast
from caro.engine import CaroGame
from caro.engine.bitboard import candidates, iter_bits, winning_squares

BUDGETS = (400, 1600)
ROLLOUTS = 200


def one(job: tuple[int, dict]) -> dict:
    index, position = job
    rng = random.Random(chung.SEED + index)
    game = CaroGame.from_moves(position["moves"])
    side, block = game.side, position["answers"][0]

    def direct(move: int) -> float:
        child = game.clone()
        child.step(move)
        total = 0
        for _ in range(ROLLOUTS):
            z = rollout_fast(child.bb[0], child.bb[1], child.side, rng, "light", 1)
            total += z if side == 0 else -z
        return total / ROLLOUTS

    other = rng.choice([c for c in iter_bits(candidates(*game.me_opp(), 2)) if c != block])
    row = {"id": position["id"], "block_value": direct(block), "other_value": direct(other), "search": {}}
    for sims in BUDGETS:
        agent = MCTSAgent(rollout="light", reuse_tree=True)     # giữ lại nút của nước được chọn để đọc cây
        agent.reset(chung.SEED + index)
        move = agent.select_action(game.clone(), Limits(max_sims=sims))
        node = agent._root
        after = game.clone()
        after.step(move)
        me, opp = after.me_opp()
        threats = winning_squares(me, opp)                      # ô thắng ngay của đối thủ sau nước đã chọn
        expanded = [c.move for c in node.children]
        row["search"][sims] = {
            "blocked": move == block, "score": agent.last_stats().get("score"), "visits": node.N,
            "expanded": len(expanded), "replies": len(candidate_moves(after, 2)),
            "winning_reply_expanded": any(threats >> m & 1 for m in expanded)}
    return row


def main() -> None:
    positions = [p for p in chung.load_json("bo-chien-thuat")["positions"] if p["category"] == "C2"]
    with mp.get_context("spawn").Pool(8) as pool:
        rows = pool.map(one, list(enumerate(positions)))
    summary = {"positions": len(rows), "rollouts": ROLLOUTS,
               "block_value": statistics.mean(r["block_value"] for r in rows),
               "other_value": statistics.mean(r["other_value"] for r in rows), "budgets": {}}
    for sims in BUDGETS:
        runs = [r["search"][sims] for r in rows]
        missed = [r for r in runs if not r["blocked"]]
        summary["budgets"][sims] = {
            "blocked": len(runs) - len(missed), "missed": len(missed),
            "missed_score": statistics.mean(r["score"] for r in missed),
            "missed_visits": statistics.mean(r["visits"] for r in missed),
            "missed_expanded": statistics.mean(r["expanded"] for r in missed),
            "missed_replies": statistics.mean(r["replies"] for r in missed),
            "missed_found_win": sum(r["winning_reply_expanded"] for r in missed)}
    print(summary)
    chung.save_json("kiem-tra-rollout", {"summary": summary, "positions": rows})


if __name__ == "__main__":
    main()
