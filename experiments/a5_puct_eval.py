"""A5 — Policy-only (cấp 5/6) vs policy + PUCT (cấp 7), cùng mạng. Đấu cặp có ngân sách tường minh.

    uv run python experiments/a5_puct_eval.py [--workers 6] [--out runs/eval/a5_puct]

Mỗi cặp: 100 khai cuộc chung (runs/openings/openings_v1.json) × 2 màu = 200 ván. Điểm = (thắng + 0,5·hoà)/tổng của
tác tử đầu cặp, kèm khoảng Wilson 95%. L3 dùng ngân sách mặc định của agent (độ sâu 6).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path

from caro.agents import make_agent
from caro.arena.match import play_match
from caro.arena.stats import wilson
from caro.runlog import derive_seed

SEED = 42


def puct(ck: str, sims: int) -> dict:
    return {"id": f"L7-PUCT[{ck.split(':')[-1]},{sims}]", "level": 7, "kind": "puct",
            "limits": {"mode": "fixed", "max_sims": sims}, "params": {"checkpoint": ck}}


L2, L3, L5, L6 = "agents/l2_rule", "agents/l3_alphabeta", "agents/l5_policy_sl", "agents/l6_policy_rl"
PAIRS = [
    (L5, L2), (L5, L3), (L6, L2), (L6, L3),
    (puct("model:l5", 100), L3), (puct("model:l5", 200), L3), (puct("model:l5", 400), L3), (puct("model:l5", 800), L3),
    (puct("model:l5", 200), L2), (puct("model:l6", 200), L3),
    (puct("model:l5", 200), L5),            # cùng mạng: có tìm kiếm so với không
]
_cache: dict[str, object] = {}


def _agent(spec):
    key = json.dumps(spec, sort_keys=True)
    if key not in _cache:
        _cache[key] = make_agent(spec)
    return _cache[key]


def _label(spec) -> str:
    return spec["id"] if isinstance(spec, dict) else spec.split("/")[-1]


def play(task):
    pi, a_is_x, op_id, moves = task
    a, b = (_agent(s) for s in PAIRS[pi])
    gid = pi * 1000 + op_id * 2 + (0 if a_is_x else 1)
    r = play_match(a if a_is_x else b, b if a_is_x else a, moves, SEED, gid)
    res = r["result"]                       # +1 X thắng, −1 O thắng, 0 hoà
    mine = 1 if a_is_x else -1
    return pi, (0.5 if res == 0 else (1.0 if res == mine else 0.0)), sum(r.get("time_ms_list", []) or [0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="runs/eval/a5_puct")
    ap.add_argument("--openings", type=int, default=100)
    args = ap.parse_args()
    openings = json.loads(Path("runs/openings/openings_v1.json").read_text(encoding="utf-8"))[:args.openings]
    tasks = [(pi, a_x, i, mv) for pi in range(len(PAIRS)) for i, mv in enumerate(openings) for a_x in (True, False)]
    acc = {pi: [] for pi in range(len(PAIRS))}
    t0 = time.time()
    with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        for k, (pi, score, _) in enumerate(ex.map(play, tasks, chunksize=4), 1):
            acc[pi].append(score)
            if k % 200 == 0:
                print(f"{k}/{len(tasks)} ván, {time.time() - t0:.0f}s", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pi, (a, b) in enumerate(PAIRS):
        s = acc[pi]
        n, w, d = len(s), sum(x == 1.0 for x in s), sum(x == 0.5 for x in s)
        lo, hi = wilson(sum(s), n)
        rows.append({"agent": _label(a), "opponent": _label(b), "n": n, "wins": w, "draws": d, "losses": n - w - d,
                     "score": round(sum(s) / n, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(f"{r['agent']:<22} vs {r['opponent']:<22} {r['wins']}W {r['draws']}D {r['losses']}L  "
              f"score {r['score']:.3f} [{r['ci_low']:.3f}; {r['ci_high']:.3f}]")
    print(f"tổng {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
