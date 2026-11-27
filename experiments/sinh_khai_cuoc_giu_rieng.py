"""Sinh bộ khai cuộc giữ riêng: không trùng dữ liệu huấn luyện và bộ đánh giá cũ.

    uv run python experiments/sinh_khai_cuoc_giu_rieng.py

Một khai cuộc bị loại nếu thế cờ của nó (tính cả 8 phép đối xứng) đã có trong `openings_v1.json`,
đã xuất hiện trong bất kỳ mẫu huấn luyện nào của expert_v1/expert_v2, hoặc đã được chọn trước đó.
Phân phối đề xuất giữ như bộ cũ: 2–4 nước ngẫu nhiên trong vùng 5×5 giữa bàn.
"""

from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np

from caro.config import project_path
from caro.engine import CaroGame
from caro.engine.symmetry import canonical_key

V1 = "runs/openings/openings_v1.json"
OUT = "runs/openings/openings_heldout_v2.json"
DATASETS = ("runs/data/expert_v1/expert_dataset.npz", "runs/data/expert_v2/expert_dataset.npz")
COUNT, PLIES, REGION, SEED = 100, (2, 4), 5, 20261126


def key_of(moves: list[int]) -> int:
    game = CaroGame.from_moves(moves)
    return canonical_key(*game.bb, game.side)


def training_keys(path: str) -> set[int]:
    """Khoá chuẩn hoá của mọi mẫu có không quá PLIES[1] quân (kênh 0 mình, 1 đối thủ, 2 lượt X)."""
    states = np.load(project_path(path))["states"]
    stones = states[:, :2].sum(axis=(1, 2, 3))
    keys = set()
    for s in states[stones <= PLIES[1]]:
        me, opp = (sum(1 << int(i) for i in np.flatnonzero(s[c].reshape(-1))) for c in (0, 1))
        x_turn = bool(s[2, 0, 0] == 1)
        x, o = (me, opp) if x_turn else (opp, me)
        keys.add(canonical_key(x, o, 0 if x_turn else 1))
    return keys


def main() -> None:
    used = {key_of(m) for m in json.loads(project_path(V1).read_text(encoding="utf-8"))}
    for path in DATASETS:
        if not project_path(path).is_file():
            raise SystemExit(f"thiếu {path}: không kiểm được trùng với dữ liệu huấn luyện")
        found = training_keys(path)
        print(f"{path}: {len(found)} thế ≤ {PLIES[1]} quân")
        used |= found

    rng = random.Random(SEED)
    start = (11 - REGION) // 2
    squares = [r * 11 + c for r in range(start, start + REGION) for c in range(start, start + REGION)]
    openings: list[list[int]] = []
    rejected = 0
    while len(openings) < COUNT:
        moves = rng.sample(squares, rng.randint(*PLIES))
        key = key_of(moves)
        if key in used:
            rejected += 1
            continue
        used.add(key)
        openings.append(moves)

    project_path(OUT).write_text(json.dumps(openings, indent=2) + "\n", encoding="utf-8")
    print(f"ghi {OUT}: {len(openings)} khai cuộc, loại {rejected} đề xuất trùng, seed {SEED}")
    print("số nước:", dict(sorted(Counter(map(len, openings)).items())))
    print("trong 50 khai cuộc đầu:", dict(sorted(Counter(map(len, openings[:50])).items())))


if __name__ == "__main__":
    main()
