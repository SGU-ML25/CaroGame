"""Cấp 6 (REINFORCE): đặc trưng nhanh khớp board_to_tensor, một vòng huấn luyện chạy được trên CPU."""

import json
import random

import pytest

torch = pytest.importorskip("torch")

from caro.config import load_config
from caro.engine.game import CaroGame
from caro.nn.features import board_to_tensor
from caro.nn.model import CaroResNet
from caro.training import train_pg


def _random_game(seed: int, plies: int) -> CaroGame:
    rng = random.Random(seed)
    g = CaroGame()
    for _ in range(plies):
        if g.is_terminal():
            break
        g.step(rng.choice(g.legal_actions()))
    return g


@pytest.mark.parametrize("plies", [0, 1, 7, 30])
def test_encode_matches_board_to_tensor(plies):
    g = _random_game(plies, plies)
    assert torch.equal(torch.from_numpy(train_pg.encode(g)), board_to_tensor(g))


def test_legal_mask_is_empty_cells():
    g = _random_game(3, 10)
    mask = train_pg.legal_mask(g)
    assert mask.sum() == 121 - 10
    assert sorted(g.legal_actions()) == [int(i) for i in mask.nonzero()[0]]


def test_update_changes_weights_and_reports_losses():
    net = CaroResNet({"num_channels": 8, "num_blocks": 1})
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    rng = random.Random(0)
    batch = train_pg.play_iteration(net, None, 4, 1.0, 1.0, 2, torch.device("cpu"), rng)
    assert batch["x"].shape[1:] == (4, 11, 11) and len(batch["z"]) == len(batch["action"])
    assert set(batch["z"].tolist()) <= {-1.0, 0.0, 1.0}
    before = [p.detach().clone() for p in net.parameters()]
    stats = train_pg.update(net, opt, batch, 1.0, 0.01, 1.0, 1.0, torch.device("cpu"))
    assert all(k in stats for k in ("loss", "policy_loss", "value_loss", "entropy"))
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters(), strict=True))


def test_train_end_to_end_tiny(tmp_path):
    cfg = load_config("training/pg", [
        "train.episodes=4", "train.batch_games=2", "train.eval_every=4", "train.eval_games=2",
        "train.init=random", "train.eval_vs=[agents/l1_random]", "model.channels=8", "model.blocks=1",
        f"run.root={tmp_path}", "run.index=null", "log.console=false", "device=cpu",
    ])
    summary = train_pg.train(cfg)
    assert summary["episodes"] == 4
    run_dir = tmp_path / "train" / "pg_policy" / "latest"
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert any(m.get("eval") for m in metrics) and any("policy_loss" in m for m in metrics)
    assert (run_dir / "checkpoints" / "best.pt").is_file() and (run_dir / "checkpoints" / "last.pt").is_file()
