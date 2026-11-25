"""Đánh giá một checkpoint policy/value bằng đấu với các tác tử cố định.

    uv run python experiments/eval_checkpoint.py \
        --checkpoint runs/train/pg_policy/.../checkpoints/best.pt \
        --vs agents/l1_random agents/l2_rule agents/l3_alphabeta agents/l5_policy_sl \
        --games 100 --label "RL v3 best" --out runs/eval/a5b_from_sl.csv

Mỗi đối thủ chơi `games` ván đổi màu, khai cuộc ngẫu nhiên 2 nước. Điểm = (W + 0.5D)/n.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from caro.training.train_pg import build_model, evaluate, load_init


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="đường dẫn .pt | model:<tên> | hf://…")
    ap.add_argument("--vs", nargs="+", default=[
        "agents/l1_random", "agents/l2_rule", "agents/l3_alphabeta", "agents/l5_policy_sl",
    ])
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default="runs/eval/checkpoint_eval.csv")
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model({"model": {"channels": 64, "blocks": 6}}, device)
    source = load_init(model, args.checkpoint, device)
    label = args.label or Path(str(source)).stem
    print(f"checkpoint: {source}\nlabel: {label}\ndevice: {device}\n")

    rows = []
    for ref in args.vs:
        ev = evaluate(model, ref, args.games, device, args.seed)
        tag = ref["id"] if isinstance(ref, dict) else ref.split("/")[-1]
        row = {"checkpoint": label, "source": str(source), "opponent": tag, **ev}
        rows.append(row)
        print(f"{tag:16s} n={ev['n']:3d}  {ev['wins']}W {ev['draws']}D {ev['losses']}L  "
              f"score={ev['score']:.3f} [{ev['ci_low']:.3f}; {ev['ci_high']:.3f}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nghi {out}")


if __name__ == "__main__":
    main()
