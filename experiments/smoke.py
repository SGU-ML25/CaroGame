"""Smoke test Cấp 2: RuleAgent đấu RandomAgent nhiều ván, win rate phải gần tuyệt đối.

Chạy:
    python experiments/smoke.py
    python experiments/smoke.py --games 200 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from caro.agents.base import Limits
from caro.agents.random_agent import RandomAgent
from caro.agents.rule_agent import RuleAgent
from caro.engine import CaroGame, X, O

LIMITS = Limits(mode="fixed", max_depth=1)


def play_one(rule_side: int, seed: int) -> str:
    """Chơi 1 ván. rule_side=0 → RuleAgent đi X (trước), 1 → đi O (sau).
    Trả 'win' | 'loss' | 'draw' theo góc nhìn RuleAgent.
    """
    game = CaroGame()

    rule  = RuleAgent(limits=LIMITS)
    rand  = RandomAgent(limits=LIMITS)
    rule.reset(seed)
    rand.reset(seed + 1)

    agents = [None, None]
    agents[rule_side] = rule
    agents[rule_side ^ 1] = rand

    while not game.is_terminal():
        agent = agents[game.side]
        action = agent.select_action(game, LIMITS)
        game.step(action)

    result = game.result()   # +1 X thắng, -1 O thắng, 0 hoà
    if result is None or result == 0:
        return "draw"
    winner_side = X if result == 1 else O
    return "win" if winner_side == rule_side else "loss"


def run(n_games: int, seed: int, verbose: bool) -> None:
    counts: Counter = Counter()

    for i in range(n_games):
        # Xen kẽ: nửa ván RuleAgent đi trước, nửa đi sau
        rule_side = i % 2
        outcome = play_one(rule_side, seed + i)
        counts[outcome] += 1

        if verbose:
            side_str = "X(trước)" if rule_side == 0 else "O(sau)"
            print(f"  ván {i+1:>4}: RuleAgent={side_str}  → {outcome}")

    total = n_games
    win_rate = counts["win"] / total * 100
    print(f"\n{'='*45}")
    print(f"  Tổng ván : {total}")
    print(f"  Thắng    : {counts['win']}  ({win_rate:.1f}%)")
    print(f"  Thua     : {counts['loss']}")
    print(f"  Hoà      : {counts['draw']}")
    print(f"{'='*45}")

    # Tiêu chí hoàn thành: thắng gần tuyệt đối (≥ 95%)
    threshold = 95.0
    if win_rate >= threshold:
        print(f"  ✅ PASS — win rate {win_rate:.1f}% ≥ {threshold}%")
    else:
        print(f"  ❌ FAIL — win rate {win_rate:.1f}% < {threshold}%")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test RuleAgent vs RandomAgent")
    ap.add_argument("--games", type=int, default=100, help="số ván (mặc định 100)")
    ap.add_argument("--seed",  type=int, default=42,  help="seed khởi đầu (mặc định 42)")
    ap.add_argument("--verbose", action="store_true",  help="in kết quả từng ván")
    args = ap.parse_args()

    print(f"RuleAgent vs RandomAgent — {args.games} ván, seed={args.seed}")
    run(args.games, args.seed, args.verbose)


if __name__ == "__main__":
    main()