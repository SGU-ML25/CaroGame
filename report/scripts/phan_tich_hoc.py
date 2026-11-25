"""Phân tích nhẹ các checkpoint học được: top-1/top-3, calibration value, entropy policy.

Chạy từ gốc repo (một tiến trình, CPU, không ảnh hưởng máy đang dùng):

    uv run python report/scripts/phan_tich_hoc.py

Kết quả ghi JSON vào report/ban-mo-rong/so-lieu-hoc-moi.json để script sinh LaTeX đọc lại.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from caro.nn.model import CaroResNet

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/ban-mo-rong/so-lieu-hoc-moi.json"
CKPTS = {
    "SL v1": "runs/train/sl_policy/latest.pt",
    "SL v2": "runs/train/sl_policy_v2/latest.pt",
    "SL v3": "runs/train/sl_policy_v3/latest.pt",
}


def split_by_game(states, game_ids, seed=42):
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(game_ids))
    n_train, n_val = int(len(unique) * 0.8), int(len(unique) * 0.9)
    test_ids = unique[n_val:]
    return np.isin(game_ids, test_ids)


def stats(pred, target):
    err = pred - target
    return {
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
        "corr": float(np.corrcoef(pred, target)[0, 1]) if np.std(pred) > 0 else 0.0,
        "sign_acc": float(np.mean(np.sign(pred) == np.sign(target))),
        "bias": float(np.mean(err)),
    }


def main():
    torch.set_num_threads(2)
    data = np.load(ROOT / "runs/data/expert_v2/expert_dataset.npz")
    states, policies, values, game_ids = data["states"], data["policies"], data["values"], data["game_ids"]
    test = split_by_game(states, game_ids)
    xs = torch.from_numpy(states[test].astype(np.float32))
    pol = policies[test]
    val_abs = values[test]
    x_turn = states[test][:, 2, 0, 0] == 1
    val_rel = val_abs * np.where(x_turn, 1.0, -1.0)
    # chỉ số nước trong ván: mẫu của mỗi ván được lưu liên tiếp theo thứ tự nước
    order = np.zeros(len(game_ids), dtype=np.int32)
    lengths = np.zeros(len(game_ids), dtype=np.int32)
    for g in np.unique(game_ids):
        m = game_ids == g
        order[m] = np.arange(m.sum())
        lengths[m] = m.sum()
    phase = order / np.maximum(1, lengths - 1)
    phase = phase[test]

    report = {"n_test": int(test.sum()), "n_games_test": int(len(np.unique(game_ids[test]))), "models": {}}
    for name, rel in CKPTS.items():
        net = CaroResNet({"num_channels": 64, "num_blocks": 6})
        state = torch.load(ROOT / rel, map_location="cpu")
        net.load_state_dict(state.get("model_state_dict", state) if isinstance(state, dict) else state)
        net.eval()
        logits, v = [], []
        with torch.no_grad():
            for i in range(0, len(xs), 512):
                lg, vv = net(xs[i:i + 512])
                logits.append(lg)
                v.append(vv.squeeze(1))
        logits = torch.cat(logits).numpy()
        pred_v = torch.cat(v).numpy()
        # top-k so với policy target (target có 0 ở nước ngoài top-k)
        top1 = np.argmax(logits, axis=1)
        top3 = np.argsort(-logits, axis=1)[:, :3]
        target_arg = np.argmax(pol, axis=1)
        target_support = pol > 1e-8
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        ce = -np.sum(pol * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        ent_pred = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
        ent_true = -np.sum(pol * np.log(np.clip(pol, 1e-12, 1.0)), axis=1)
        report["models"][name] = {
            "top1": float(np.mean(top1 == target_arg)),
            "top3": float(np.mean([target_arg[i] in top3[i] for i in range(len(top1))])),
            "top1_in_support": float(np.mean([target_support[i, top1[i]] for i in range(len(top1))])),
            "policy_ce": float(np.mean(ce)),
            "policy_entropy_pred": float(np.mean(ent_pred)),
            "policy_entropy_target": float(np.mean(ent_true)),
            "value_relative": stats(pred_v, val_rel),
            "value_absolute": stats(pred_v, val_abs),
            "value_relative_X": stats(pred_v[x_turn], val_rel[x_turn]),
            "value_relative_O": stats(pred_v[~x_turn], val_rel[~x_turn]),
            "value_by_phase": [
                stats(pred_v[m], val_rel[m])
                for m in (phase < 1 / 3, (phase >= 1 / 3) & (phase < 2 / 3), phase >= 2 / 3)
            ],
        }
        print(name, {k: round(v, 4) for k, v in report["models"][name]["value_relative"].items()},
              "top1", round(report["models"][name]["top1"], 4), "top3", round(report["models"][name]["top3"], 4), flush=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ghi", OUT)


if __name__ == "__main__":
    main()
