"""Phân tích thống kê bổ sung trên log RR8 giữ riêng. Ghi chuyen-sau/data/phan-tich-thong-ke.json.

    .venv/bin/python report/scripts/chuyen_sau/phan_tich_thong_ke.py
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections import defaultdict
from statistics import NormalDist, median

import chung
import numpy as np

from caro.arena.stats import bradley_terry, elo, wilson

BOOT = 10_000


def pair_scores(records: list[dict]) -> dict[tuple[str, str], dict[int, list[tuple[int, float]]]]:
    """(a, b) theo thứ tự cấp → opening_id → [(màu của a, điểm của a)]."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        a, b = sorted((r["agent_x"], r["agent_o"]), key=chung.AGENT_ORDER.index)
        a_is_x = r["agent_x"] == a
        score = (r["result"] * (1 if a_is_x else -1) + 1) / 2
        out[a, b][r["opening_id"]].append((0 if a_is_x else 1, score))
    return out


def sign_test(wins: int, losses: int) -> float:
    """Kiểm định dấu hai phía chính xác (bỏ các khai cuộc hoà điểm)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(max(wins, losses), n + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def pairs_table(records: list[dict], rng: np.random.Generator) -> list[dict]:
    rows = []
    lengths = defaultdict(list)
    for r in records:
        lengths[frozenset((r["agent_x"], r["agent_o"]))].append(len(r["moves"]))
    for (a, b), openings in sorted(pair_scores(records).items(),
                                   key=lambda kv: [chung.AGENT_ORDER.index(n) for n in kv[0]]):
        flat = [g for games in openings.values() for g in games]
        per_opening = np.array([sum(s for _, s in games) / len(games) for _, games in sorted(openings.items())])
        n = len(flat)
        score = sum(s for _, s in flat) / n
        boot = per_opening[rng.integers(0, len(per_opening), (BOOT, len(per_opening)))].mean(axis=1)
        better, worse = int((per_opening > 0.5).sum()), int((per_opening < 0.5).sum())
        as_x = [s for c, s in flat if c == 0]
        as_o = [s for c, s in flat if c == 1]
        rows.append({"a": a, "b": b, "n": n, "w": sum(s == 1 for _, s in flat), "d": sum(s == 0.5 for _, s in flat),
                     "l": sum(s == 0 for _, s in flat), "score": score, "wilson": wilson(score * n, n),
                     "boot": [float(q) for q in np.quantile(boot, [0.025, 0.975])],
                     "as_x": sum(as_x) / len(as_x), "as_o": sum(as_o) / len(as_o),
                     "openings_better": better, "openings_worse": worse,
                     "openings_even": len(per_opening) - better - worse, "p": sign_test(better, worse),
                     "length_median": median(lengths[frozenset((a, b))]),
                     "length_min": min(lengths[frozenset((a, b))]), "length_max": max(lengths[frozenset((a, b))])})
    for row, adjusted in zip(rows, holm([r["p"] for r in rows]), strict=True):
        row["p_holm"] = adjusted
    return rows


def ratings(records: list[dict], rng: np.random.Generator) -> dict:
    matches = [(r["agent_x"], r["agent_o"], r["result"]) for r in records]
    priors = {str(p): bradley_terry(matches, prior_draws=p) for p in (0.1, 0.5, 1.0, 2.0, 5.0)}
    base = priors["1.0"]

    openings = sorted({r["opening_id"] for r in records})
    by_opening = defaultdict(list)
    for r in records:
        by_opening[r["opening_id"]].append((r["agent_x"], r["agent_o"], r["result"]))
    convergence = []
    for k in (5, 10, 20, 30, 40, 50):
        draws = []
        for _ in range(400):
            chosen = rng.choice(len(openings), size=k, replace=True)
            draws.append(bradley_terry([m for i in chosen for m in by_opening[openings[i]]]))
        convergence.append({"openings": k, "agents": {
            name: [float(np.quantile([d[name] for d in draws], q)) for q in (0.025, 0.5, 0.975)]
            for name in chung.AGENT_ORDER}})

    shuffled = {}
    shuffler = random.Random(chung.SEED)
    for k_factor in (16, 32):
        runs = []
        for _ in range(200):
            order = matches[:]
            shuffler.shuffle(order)
            runs.append(elo(order, k_factor=k_factor))
        shuffled[str(k_factor)] = {name: {"mean": float(np.mean([r[name] for r in runs])),
                                          "std": float(np.std([r[name] for r in runs])),
                                          "min": float(min(r[name] for r in runs)),
                                          "max": float(max(r[name] for r in runs))}
                                   for name in chung.AGENT_ORDER}
    sequential = elo([(r["game_id"], r["agent_x"], r["agent_o"], r["result"]) for r in records])
    return {"priors": priors, "convergence": convergence, "elo_shuffled": shuffled,
            "elo_sequential": sequential, "base": base}


def model_fit(pairs: list[dict], rating: dict[str, float]) -> dict:
    rows = []
    for p in pairs:
        predicted = 1 / (1 + 10 ** ((rating[p["b"]] - rating[p["a"]]) / 400))
        rows.append({"a": p["a"], "b": p["b"], "observed": p["score"], "predicted": predicted,
                     "residual": p["score"] - predicted})
    score = {(p["a"], p["b"]): p["score"] for p in pairs}

    def beats(a, b):
        return score[a, b] > 0.5 if (a, b) in score else score[b, a] < 0.5

    cycles = [list(t) for t in itertools.combinations(chung.AGENT_ORDER, 3)
              if beats(t[0], t[1]) and beats(t[1], t[2]) and beats(t[2], t[0])
              or beats(t[1], t[0]) and beats(t[2], t[1]) and beats(t[0], t[2])]
    return {"pairs": rows, "rmse": float(np.sqrt(np.mean([r["residual"] ** 2 for r in rows]))), "cycles": cycles}


def board_usage(records: list[dict]) -> dict:
    first = {name: [0] * 121 for name in chung.AGENT_ORDER}
    every = {name: [0] * 121 for name in chung.AGENT_ORDER}
    by_ply: dict = {name: defaultdict(lambda: {"time": [], "distance": []}) for name in chung.AGENT_ORDER}
    for r in records:
        seen = set()
        for k, move in enumerate(r["moves"][r["opening_plies"]:]):
            ply = r["opening_plies"] + k
            name = r["agent_x"] if ply % 2 == 0 else r["agent_o"]
            every[name][move] += 1
            if name not in seen:
                first[name][move] += 1
                seen.add(name)
            bucket = by_ply[name][min(ply // 4 * 4, 40)]
            bucket["time"].append(r["time_ms"][k])
            previous = r["moves"][ply - 1]
            bucket["distance"].append(max(abs(move // 11 - previous // 11), abs(move % 11 - previous % 11)))
    return {"first": first, "every": every, "by_ply": {
        name: [{"ply": ply, "n": len(v["time"]), "time_median": float(np.median(v["time"])),
                "time_p90": float(np.percentile(v["time"], 90)), "distance": float(np.mean(v["distance"])),
                "adjacent": float(np.mean(np.array(v["distance"]) <= 1))}
               for ply, v in sorted(buckets.items())] for name, buckets in by_ply.items()}}


def openings_effect(records: list[dict]) -> list[dict]:
    openings = json.loads((chung.ROOT / "runs/openings/openings_heldout_v2.json").read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for r in records:
        grouped[r["opening_id"]].append(r)
    rows = []
    for opening_id, games in sorted(grouped.items()):
        moves = openings[opening_id]
        decisive = [g["result"] for g in games]
        rows.append({"id": opening_id, "moves": moves, "n": len(games),
                     "x_score": float(np.mean([(v + 1) / 2 for v in decisive])),
                     "draws": sum(v == 0 for v in decisive),
                     "length": float(np.mean([len(g["moves"]) for g in games]))})
    return rows


def power() -> list[dict]:
    z = NormalDist().inv_cdf
    rows = []
    for p1 in (0.52, 0.55, 0.60, 0.65, 0.75):
        row = {"p1": p1}
        for target in (0.8, 0.9):
            n = ((z(0.975) * 0.5 + z(target) * math.sqrt(p1 * (1 - p1))) / (p1 - 0.5)) ** 2
            row[str(target)] = math.ceil(n)
        rows.append(row)
    return rows


def main() -> None:
    records = chung.games(chung.RUNS["rr8"])
    rng = np.random.default_rng(chung.SEED)
    pairs = pairs_table(records, rng)
    rating = ratings(records, rng)
    chung.save_json("phan-tich-thong-ke", {
        "games": len(records), "pairs": pairs, "ratings": rating, "fit": model_fit(pairs, rating["base"]),
        "board": board_usage(records), "openings": openings_effect(records), "power": power()})
    print("cặp:", len(pairs), "· có ý nghĩa sau Holm:", sum(p["p_holm"] < 0.05 for p in pairs))


if __name__ == "__main__":
    main()
