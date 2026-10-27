"""Smoke test Cấp 3: AlphaBetaAgent đấu RuleAgent nhiều ván, win rate phải rõ rệt.

Hoàn thành khi: khoảng Wilson không chứa 0.5 (thắng rõ rệt hơn Cấp 2).

Chạy:
    python experiments/smoke_ab.py
    python experiments/smoke_ab.py --games 50 --depth 3
    python experiments/smoke_ab.py --verbose
"""

from __future__ import annotations

import argparse
import math
import sys

from caro.agents.alphabeta import AlphaBetaAgent
from caro.agents.base import Limits
from caro.agents.rule_agent import RuleAgent
from caro.engine import CaroGame, X, O

AB_LIMITS   = Limits(mode="fixed", max_depth=3)
RULE_LIMITS = Limits(mode="fixed", max_depth=1)


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng Wilson 95% cho tỉ lệ thắng."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def play_one(ab_side: int, seed: int, depth: int) -> str:
    """Chơi 1 ván. ab_side=0 → AB đi X (trước), 1 → đi O (sau).
    Trả 'win' | 'loss' | 'draw' theo góc nhìn AlphaBetaAgent.
    """
    game = CaroGame()
    ab   = AlphaBetaAgent(limits=AB_LIMITS)
    rule = RuleAgent(limits=RULE_LIMITS)
    ab.reset(seed)
    rule.reset(seed + 1)

    limits_ab   = Limits(mode="fixed", max_depth=depth)
    limits_rule = Limits(mode="fixed", max_depth=1)

    agents = [None, None]
    agents[ab_side]     = ab
    agents[ab_side ^ 1] = rule

    while not game.is_terminal():
        agent = agents[game.side]
        if agent is ab:
            action = agent.select_action(game, limits_ab)
        else:
            action = agent.select_action(game, limits_rule)
        game.step(action)

    result = game.result()
    if result is None or result == 0:
        return "draw"
    winner_side = X if result == 1 else O
    return "win" if winner_side == ab_side else "loss"


def run(n_games: int, seed: int, depth: int, verbose: bool) -> None:
    wins = losses = draws = 0

    for i in range(n_games):
        ab_side = i % 2
        outcome = play_one(ab_side, seed + i, depth)
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            draws += 1

        if verbose:
            side_str = "X(trước)" if ab_side == 0 else "O(sau)"
            print(f"  ván {i+1:>4}: AB={side_str}  → {outcome}")

    total = n_games
    win_rate = wins / total * 100
    lo, hi = wilson_interval(wins, total)

    print(f"\n{'='*50}")
    print(f"  AlphaBeta(depth={depth}) vs RuleAgent — {total} ván")
    print(f"  Thắng : {wins}  ({win_rate:.1f}%)")
    print(f"  Thua  : {losses}")
    print(f"  Hoà   : {draws}")
    print(f"  Wilson 95%: [{lo*100:.1f}%, {hi*100:.1f}%]")
    print(f"{'='*50}")

    # Hoàn thành khi khoảng Wilson không chứa 0.5
    if lo > 0.5:
        print(f"  ✅ PASS — khoảng Wilson [{lo*100:.1f}%, {hi*100:.1f}%] không chứa 50%")
    elif hi < 0.5:
        print(f"  ❌ FAIL — AB thua RuleAgent rõ rệt")
        sys.exit(1)
    else:
        print(f"  ⚠️  CHƯA RÕ — khoảng Wilson chứa 50%, cần thêm ván hoặc tăng depth")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test AlphaBeta vs RuleAgent")
    ap.add_argument("--games",   type=int, default=20,  help="số ván (mặc định 20)")
    ap.add_argument("--seed",    type=int, default=42,  help="seed khởi đầu (mặc định 42)")
    ap.add_argument("--depth",   type=int, default=3,   help="độ sâu AB (mặc định 3)")
    ap.add_argument("--verbose", action="store_true",   help="in kết quả từng ván")
    args = ap.parse_args()

    print(f"AlphaBeta(depth={args.depth}) vs RuleAgent — {args.games} ván, seed={args.seed}")
    run(args.games, args.seed, args.depth, args.verbose)


if __name__ == "__main__":
    main()