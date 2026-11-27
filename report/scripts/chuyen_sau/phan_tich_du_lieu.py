"""Kiểm tra nhãn policy trong dữ liệu expert (hoà điểm, thế thắng ngay) và ảnh hưởng lên số đo trên tập test.

    .venv/bin/python report/scripts/chuyen_sau/phan_tich_du_lieu.py

Một mẫu là "hoà điểm" khi nhãn có từ hai nước trở lên với xác suất bằng nhau. scored_moves sắp xếp ổn định theo
điểm, nên khi mọi nước cùng điểm, top-k là k ứng viên có chỉ số ô nhỏ nhất. Script đếm các mẫu này, tái hiện lại
bằng chính tác tử L3 trên một số thế, và tách số đo của bốn mạng theo hai nhóm mẫu.
"""

from __future__ import annotations

import chung
import numpy as np
import torch
from phan_tich_mang import MODELS, infer, load

from caro.agents import make_agent
from caro.agents.base import Limits
from caro.config import load_config
from caro.engine import CaroGame
from caro.engine.bitboard import iter_bits, popcount, winning_squares
from caro.training.train_sl import mover_relative_values, split_groups

REPRODUCE = 200


def bitboards(states: np.ndarray) -> list[tuple[int, int]]:
    packed = np.packbits(states[:, :2].reshape(len(states), 2, 121), axis=2, bitorder="little")
    return [(int.from_bytes(a.tobytes(), "little"), int.from_bytes(b.tobytes(), "little")) for a, b in packed]


def groups(states, policies):
    support = policies > 1e-6
    spread = np.where(support, policies, -np.inf).max(axis=1) - np.where(support, policies, np.inf).min(axis=1)
    tie = (support.sum(axis=1) > 1) & (spread < 1e-6)
    boards = bitboards(states)
    lost = np.array([not winning_squares(me, opp) and popcount(winning_squares(opp, me)) >= 2
                     for me, opp in boards])
    return tie, lost, boards


def dataset_summary(version: str) -> tuple[dict, tuple]:
    data = np.load(chung.ROOT / f"runs/data/expert_{version}/expert_dataset.npz")
    states, policies, values, ids = (data[k] for k in ("states", "policies", "values", "game_ids"))
    tie, lost, boards = groups(states, policies)
    top = policies.argmax(axis=1)
    stones = states[:, :2].reshape(len(states), -1).sum(axis=1)
    per_game = np.bincount(ids, minlength=20000)

    def share(mask, of):
        return float(mask[of].mean()) if of.any() else 0.0

    everything = np.ones(len(tie), bool)
    summary = {
        "samples": len(tie), "tie": int(tie.sum()), "lost": int(lost.sum()),
        "tie_given_lost": share(tie, lost), "row1_all": share(top < 11, everything),
        "row1_tie": share(top < 11, tie), "row1_clean": share(top < 11, ~tie),
        "row1_lost": share(top < 11, lost), "tie_given_row1": share(tie, top < 11),
        "games_total": len(per_game), "games_empty": int((per_game == 0).sum()),
        "games_le2": int((per_game <= 2).sum()), "games_median": float(np.median(per_game)),
        "by_stones": [{"lo": lo, "hi": hi, "n": int(((stones >= lo) & (stones < hi)).sum()),
                       "tie": share(tie, (stones >= lo) & (stones < hi)),
                       "lost": share(lost, (stones >= lo) & (stones < hi)),
                       "draw": share(values == 0, (stones >= lo) & (stones < hi))}
                      for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 50), (50, 122))],
        "heat_tie": (policies[tie].mean(axis=0)).tolist(), "heat_clean": (policies[~tie].mean(axis=0)).tolist(),
    }
    return summary, (states, policies, values, ids, tie, lost, boards)


def reproduce(states, policies, lost, boards) -> dict:
    """Gọi lại L3 (độ sâu 3, search đủ từng nước ở gốc) trên các thế đã thua."""
    agent = make_agent("agents/l3_alphabeta")
    rng = np.random.default_rng(chung.SEED)
    chosen = rng.choice(np.flatnonzero(lost), size=min(REPRODUCE, int(lost.sum())), replace=False)
    all_equal = matches = 0
    example = None
    for i in chosen:
        me, opp = boards[i]
        x_to_move = states[i, 2, 0, 0] == 1
        game = CaroGame()
        game.bb = [me, opp] if x_to_move else [opp, me]
        game.side = 0 if x_to_move else 1
        agent.reset(0)
        scored = agent.scored_moves(game, Limits(max_depth=3), top_k=121, prune_root=False)
        equal = len({s for _, s in scored}) == 1
        stored = sorted(np.flatnonzero(policies[i] > 1e-6).tolist())
        same = sorted(m for m, _ in scored[:len(stored)]) == stored
        all_equal += equal
        matches += same
        if example is None and equal and same and stored[0] < 11:
            example = {"index": int(i), "x": int(game.bb[0]), "o": int(game.bb[1]), "side": game.side,
                       "score": int(scored[0][1]), "candidates": len(scored), "stored": stored}
    return {"n": len(chosen), "all_equal": all_equal, "same_support": matches, "example": example}


def immediate_wins(states, policies, boards) -> tuple[np.ndarray, np.ndarray, dict, list]:
    """Mẫu mà bên đi có ô thắng ngay: nhãn đặt bao nhiêu xác suất lên các ô đó, và vì sao."""
    masks = np.zeros((len(boards), 121), bool)
    for i, (me, opp) in enumerate(boards):
        masks[i, list(iter_bits(winning_squares(me, opp)))] = True
    has = masks.any(axis=1)
    mass = (policies * masks).sum(axis=1)
    summary = {"n": int(has.sum()), "share": float(has.mean()), "label_mass": float(mass[has].mean()),
               "label_argmax_wins": float(masks[has, policies[has].argmax(axis=1)].mean()),
               "mass_below_half": float((mass[has] < 0.5).mean()),
               "single_winner": float((masks[has].sum(axis=1) == 1).mean())}
    agent = make_agent("agents/l3_alphabeta")
    examples = []
    for i in np.flatnonzero(has)[:3]:
        me, opp = boards[i]
        x_to_move = states[i, 2, 0, 0] == 1
        game = CaroGame()
        game.bb = [me, opp] if x_to_move else [opp, me]
        game.side = 0 if x_to_move else 1
        agent.reset(0)
        scored = agent.scored_moves(game, Limits(max_depth=3), top_k=8, prune_root=False)
        examples.append({"index": int(i), "winning": np.flatnonzero(masks[i]).tolist(),
                         "scored": [[int(m), int(s)] for m, s in scored]})
    return has, masks, summary, examples


def main() -> None:
    torch.set_num_threads(4)
    v2, (states, policies, values, ids, tie, lost, boards) = dataset_summary("v2")
    v1, _ = dataset_summary("v1")
    result = {"expert_v1": v1, "expert_v2": v2, "reproduce": reproduce(states, policies, lost, boards)}
    print({k: v for k, v in result["reproduce"].items() if k != "example"}, flush=True)

    has_win, win_masks, wins, examples = immediate_wins(states, policies, boards)
    result["immediate_win"] = {**wins, "examples": examples}
    print(wins, flush=True)

    test = split_groups(ids, load_config("training/sl_v3")["seed"]) == 2
    relative = mover_relative_values(states, values)
    top = policies.argmax(axis=1)
    models = {}
    for name, ref in MODELS.items():
        probs, v = infer(load(ref), states[test])
        order = np.argsort(-probs, axis=1)
        rank = (order == top[test][:, None]).argmax(axis=1)
        row1 = probs[:, :11].sum(axis=1)
        out = {}
        for label, mask in (("all", np.ones(test.sum(), bool)), ("clean", ~tie[test]), ("tie", tie[test]),
                            ("lost", lost[test])):
            support = policies[test][mask] > 1e-6
            out[label] = {"n": int(mask.sum()), "top1": float((rank[mask] < 1).mean()),
                          "top3": float((rank[mask] < 3).mean()),
                          "in_support": float(support[np.arange(mask.sum()), order[mask, 0]].mean()),
                          "row1_mass": float(row1[mask].mean()),
                          "mse": float(((v[mask] - relative[test][mask]) ** 2).mean())}
        m = has_win[test]
        winners = win_masks[test][m]
        out["immediate_win"] = {"n": int(m.sum()), "mass": float((probs[m] * winners).sum(axis=1).mean()),
                                "top1": float(winners[np.arange(m.sum()), order[m, 0]].mean())}
        models[name] = out
        print(name, {k: (round(o["top1"], 3), round(o.get("row1_mass", o.get("mass")), 3)) for k, o in out.items()},
              flush=True)
    result["models"] = models
    chung.save_json("phan-tich-du-lieu", result)
    for version in ("expert_v1", "expert_v2"):
        print(version, {k: v for k, v in result[version].items() if not k.startswith("heat") and k != "by_stones"})


if __name__ == "__main__":
    main()
