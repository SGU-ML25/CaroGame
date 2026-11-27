"""Đo đỉnh RSS của một tiến trình chỉ chạy một tác tử trên 10 thế giữa ván, ngân sách như RR8.

    uv run python experiments/do_bo_nho.py        # ghi runs/bench/ram_rss.json
"""

import json
import resource
import subprocess
import sys
from pathlib import Path

SPECS = {
    "L1-Random": "agents/l1_random", "L2-Rule": "agents/l2_rule",
    "L3-AlphaBeta": {"from": "agents/l3_alphabeta", "budget": {"max_depth": 4}, "params": {"backend": "native"}},
    "L4-MCTS": {"from": "agents/l4_mcts", "budget": {"max_sims": 400}},
    "L5-PolicySL": "agents/l5_policy_sl", "L6-PolicyRL": "agents/l6_policy_rl",
    "L7-PUCT": {"from": "agents/l7_puct", "budget": {"max_sims": 200}, "params": {"backend": "native"}},
    "L8-AlphaZero": {"from": "agents/l8_alphazero", "budget": {"max_sims": 200}, "params": {"backend": "native"}},
}
if len(sys.argv) > 1:
    from caro.agents import make_agent
    from caro.agents.base import Limits
    from caro.config import load_config
    from caro.engine import CaroGame
    cfg = load_config("experiments/rr8_heldout", ["agents=" + json.dumps([SPECS[sys.argv[1]]])])
    spec = cfg["agents"][0]
    agent = make_agent(spec)
    for moves in json.loads(Path("runs/openings/midgame_positions_v1.json").read_text(encoding="utf-8"))["positions"][:10]:
        agent.reset(0)
        agent.select_action(CaroGame.from_moves(moves), Limits.from_dict(spec["limits"]))
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
else:
    out = {}
    for name in SPECS:
        r = subprocess.run([sys.executable, __file__, name], capture_output=True, text=True, check=True)
        out[name] = round(float(r.stdout.split()[-1]))
        print(name, out[name], "MB", flush=True)
    Path("runs/bench/ram_rss.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
