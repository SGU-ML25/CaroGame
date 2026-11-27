"""Đo mạng policy–value trên tập test, đối xứng D4 và bộ thế chiến thuật. Chạy CPU.

    .venv/bin/python report/scripts/chuyen_sau/phan_tich_mang.py

Ghi chuyen-sau/data/phan-tich-mang.json.
"""

from __future__ import annotations

import itertools

import chung
import numpy as np
import torch
import vcf

from caro.config import load_config
from caro.engine import CaroGame
from caro.engine.symmetry import PERMS
from caro.hub import resolve_checkpoint
from caro.nn.features import encode, legal_mask
from caro.nn.model import CaroResNet
from caro.training.train_sl import mover_relative_values, split_groups

MODELS = {
    "SL v3 (L5, L7)": "model:l5",
    "SL v3 không augment": "runs/train/a6_noaug/best.pt",
    "REINFORCE (L6)": "model:l6",
    "AlphaZero vòng 37 (L8)": "model:l8",
}
BUCKETS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 122))
SYMMETRY_SAMPLES = 4000


def load(ref: str) -> CaroResNet:
    net = CaroResNet({"num_channels": 64, "num_blocks": 6})
    state = torch.load(resolve_checkpoint(ref), map_location="cpu")
    net.load_state_dict(state.get("model_state_dict", state))
    return net.eval()


@torch.no_grad()
def infer(net: CaroResNet, states: np.ndarray, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """(xác suất policy đã che ô có quân, value)."""
    probs, values = [], []
    for i in range(0, len(states), batch):
        x = torch.from_numpy(states[i:i + batch].astype(np.float32))
        mask = ~(x[:, :2].sum(dim=1) > 0).reshape(len(x), 121)
        logits, v = net(x, mask)
        probs.append(torch.softmax(logits, dim=1).numpy())
        values.append(v.squeeze(1).numpy())
    return np.concatenate(probs), np.concatenate(values)


def transform(states: np.ndarray, k: int) -> np.ndarray:
    flat = states.reshape(len(states), 4, 121)
    out = np.empty_like(flat)
    out[:, :, list(PERMS[k])] = flat
    return out.reshape(states.shape)


def policy_metrics(probs, values, target, z, stones) -> dict:
    arg = target.argmax(axis=1)
    order = np.argsort(-probs, axis=1)
    rank = (order == arg[:, None]).argmax(axis=1)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(axis=1)

    def summary(m):
        return {"n": int(m.sum()), "top1": float((rank[m] < 1).mean()), "top3": float((rank[m] < 3).mean()),
                "top5": float((rank[m] < 5).mean()), "entropy": float(entropy[m].mean()),
                "mse": float(((values[m] - z[m]) ** 2).mean()),
                "sign": float((np.sign(values[m]) == np.sign(z[m]))[z[m] != 0].mean()),
                "mean_v": float(values[m].mean()), "mean_z": float(z[m].mean())}

    edges = np.linspace(-1, 1, 11)
    calibration = []
    for lo, hi in itertools.pairwise(edges):
        m = (values >= lo) & (values < hi if hi < 1 else values <= hi)
        calibration.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                            "v": float(values[m].mean()) if m.any() else None,
                            "z": float(z[m].mean()) if m.any() else None})
    return {"all": summary(np.ones(len(z), bool)),
            "by_stones": [{"lo": lo, "hi": hi, **summary((stones >= lo) & (stones < hi))} for lo, hi in BUCKETS],
            "rank_cdf": [float((rank < k).mean()) for k in range(1, 21)],
            "calibration": calibration}


def symmetry_metrics(net, states, target) -> dict:
    base, base_v = infer(net, states)
    tv, agree, values = [], [], [base_v]
    average = base.copy()
    for k in range(1, 8):
        probs, v = infer(net, transform(states, k))
        back = probs[:, list(PERMS[k])]                    # xác suất ô i = xác suất ô ảnh của i
        tv.append(0.5 * np.abs(back - base).sum(axis=1))
        agree.append(back.argmax(axis=1) == base.argmax(axis=1))
        values.append(v)
        average += back
    average /= 8
    values = np.stack(values)
    arg = target.argmax(axis=1)
    return {"n": len(states), "tv": float(np.mean(tv)), "top1_agree": float(np.mean(agree)),
            "value_std": float(values.std(axis=0).mean()),
            "value_range": float((values.max(axis=0) - values.min(axis=0)).mean()),
            "top1_single": float((base.argmax(axis=1) == arg).mean()),
            "top1_ensemble": float((average.argmax(axis=1) == arg).mean())}


def tactical_metrics(net) -> dict:
    positions = chung.load_json("bo-chien-thuat")["positions"]
    games = [CaroGame.from_moves(p["moves"]) for p in positions]
    probs, values = infer(net, np.stack([encode(g) for g in games]))
    out = {}
    for category in sorted({p["category"] for p in positions}):
        rows = [i for i, p in enumerate(positions) if p["category"] == category]
        mass = [probs[i, positions[i]["answers"]].sum() for i in rows]
        top1 = [int(probs[i].argmax()) in positions[i]["answers"] for i in rows]
        top5 = [bool(set(np.argsort(-probs[i])[:5].tolist()) & set(positions[i]["answers"])) for i in rows]
        out[category] = {"n": len(rows), "mass": float(np.mean(mass)), "top1": float(np.mean(top1)),
                         "top5": float(np.mean(top5)), "value": float(values[rows].mean()),
                         "value_positive": float((values[rows] > 0).mean())}
        assert all(legal_mask(games[i])[positions[i]["answers"]].all() for i in rows)
    return out


def trajectories(nets: dict) -> list[dict]:
    """Giá trị (góc nhìn X) theo từng nước của vài ván L7 – L8, kèm nước đầu tiên một bên có VCF."""
    records = [r for r in chung.games(chung.RUNS["rr8"])
               if {r["agent_x"], r["agent_o"]} == {"L7-PUCT", "L8-AlphaZero"}]
    records.sort(key=lambda r: (len(r["moves"]), r["game_id"]))
    chosen = [records[int(q * (len(records) - 1))] for q in (0.25, 0.5, 0.75, 0.95)]
    out = []
    for record in chosen:
        states, first = [], None
        for ply in range(record["opening_plies"], len(record["moves"])):
            game = CaroGame.from_moves(record["moves"][:ply])
            states.append(encode(game))
            if first is None and vcf.has_vcf(*game.me_opp()):
                first = {"ply": ply, "side": ply % 2}
        sign = np.array([1 if ply % 2 == 0 else -1
                         for ply in range(record["opening_plies"], len(record["moves"]))])
        item = {k: record[k] for k in ("game_id", "agent_x", "agent_o", "result", "opening_plies", "moves")}
        item["first_vcf"] = first
        item["values"] = {name: (infer(net, np.stack(states))[1] * sign).tolist()
                          for name, net in nets.items()}
        out.append(item)
    return out


def main() -> None:
    torch.set_num_threads(4)
    data = np.load(chung.ROOT / "runs/data/expert_v2/expert_dataset.npz")
    states, policies, values, game_ids = (data[k] for k in ("states", "policies", "values", "game_ids"))
    values = mover_relative_values(states, values)
    group = split_groups(game_ids, load_config("training/sl_v3")["seed"])
    stones = states[:, :2].reshape(len(states), -1).sum(axis=1)

    dataset = {"samples": len(values), "games": len(np.unique(game_ids)),
               "split": {name: int((group == k).sum()) for k, name in enumerate(("train", "val", "test"))},
               "stones_hist": np.bincount(stones, minlength=1).tolist(),
               "value_counts": {str(int(v)): int((values == v).sum()) for v in (-1, 0, 1)},
               "x_to_move": int((states[:, 2, 0, 0] == 1).sum()),
               "game_length_hist": np.bincount(np.bincount(game_ids)).tolist(),
               "target_entropy_mean": float(-(policies * np.log(np.clip(policies, 1e-12, 1))).sum(axis=1).mean()),
               "target_max_mean": float(policies.max(axis=1).mean()),
               "target_support_mean": float((policies > 1e-6).sum(axis=1).mean()),
               "policy_heat": (policies.sum(axis=0) / len(policies)).tolist()}

    test = group == 2
    rng = np.random.default_rng(chung.SEED)
    sample = rng.choice(np.flatnonzero(test), size=SYMMETRY_SAMPLES, replace=False)
    nets = {name: load(ref) for name, ref in MODELS.items()}
    models = {}
    for name, net in nets.items():
        probs, pred = infer(net, states[test])
        models[name] = {**policy_metrics(probs, pred, policies[test], values[test], stones[test]),
                        "symmetry": symmetry_metrics(net, states[sample], policies[sample]),
                        "tactics": tactical_metrics(net)}
        print(name, models[name]["all"], models[name]["symmetry"], flush=True)

    first = next(iter(nets.values()))
    layers = [{"name": n, "shape": list(p.shape), "params": p.numel()} for n, p in first.named_parameters()]
    chosen = {n: nets[n] for n in ("SL v3 (L5, L7)", "AlphaZero vòng 37 (L8)")}
    positions = chung.load_json("bo-chien-thuat")["positions"]
    heat = []
    for p in [next(p for p in positions if p["category"] == c) for c in ("C1", "C2", "C3", "C4", "C5")]:
        x = np.stack([encode(CaroGame.from_moves(p["moves"]))])
        heat.append({"id": p["id"], "maps": {n: {"policy": infer(net, x)[0][0].tolist(),
                                                 "value": float(infer(net, x)[1][0])}
                                             for n, net in nets.items()}})
    chung.save_json("phan-tich-mang", {"dataset": dataset, "models": models, "layers": layers,
                                       "trajectories": trajectories(chosen), "heat": heat})


if __name__ == "__main__":
    main()
