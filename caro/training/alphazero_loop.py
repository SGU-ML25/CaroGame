"""L8: tự đấu bằng PUCT, học policy từ lượt thăm và value từ kết quả ván."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from caro.agents.base import Limits
from caro.agents.puct import PUCTAgent
from caro.arena.match import play_match
from caro.config import load_config, project_path
from caro.engine import CaroGame, N
from caro.nn.dataset import ArrayDataset
from caro.nn.features import encode
from caro.nn.model import CaroResNet
from caro.runlog import Run, derive_seed
from caro.training.train_pg import build_model, load_init, save_checkpoint

GameSamples = tuple[np.ndarray, np.ndarray, np.ndarray]


def selfplay_game(agent: PUCTAgent, seed: int, temperature_moves: int) -> tuple[GameSamples, dict]:
    game = CaroGame()
    agent.reset(seed)
    states, policies, movers = [], [], []
    moves, times, sims = [], [], []
    while not game.is_terminal():
        agent.temperature = float(len(moves) < temperature_moves)
        move = agent.select_action(game, agent.limits)
        states.append(encode(game).astype(np.uint8))
        policies.append(agent.last_pi)
        movers.append(game.side)
        stats = agent.last_stats()
        moves.append(move)
        times.append(stats["time_ms"])
        sims.append(stats["sims"])
        game.step(move)
    result = game.result()
    values = np.asarray([result if side == 0 else -result for side in movers], dtype=np.int8)
    return (np.stack(states), np.stack(policies), values), {
        "seed": seed, "moves": moves, "result": result, "termination": game.termination(),
        "time_ms": times, "sims": sims,
    }


def train_replay(model, optimizer, replay, settings, device, rng, augment) -> dict:
    """Lấy mẫu đều theo thế cờ trong cửa sổ replay, chỉ ghép mảng của minibatch."""
    games = tuple(replay)
    ends = np.cumsum([len(game[2]) for game in games])
    starts = np.r_[0, ends[:-1]]
    sums = dict.fromkeys(("loss", "policy_loss", "value_loss"), 0.0)
    model.train()
    for _ in range(settings["steps_per_iter"]):
        indices = rng.integers(ends[-1], size=settings["batch_size"])
        game_ids = np.searchsorted(ends, indices, side="right")
        positions = indices - starts[game_ids]
        samples = [tuple(arr[pos] for arr in games[i]) for i, pos in zip(game_ids, positions, strict=True)]
        data = ArrayDataset(*(np.stack(column) for column in zip(*samples, strict=True)), augment=augment)
        states, policy, values = [
            torch.stack(column).to(device) for column in zip(*(data[i] for i in range(len(data))), strict=True)
        ]
        legal = ~states[:, :2].bool().any(dim=1).reshape(-1, N * N)
        logits, pred = model(states, legal)
        log_probs = F.log_softmax(logits, dim=1).masked_fill(~legal, 0.0)
        policy_loss = -(policy * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(pred.squeeze(1), values)
        loss = policy_loss + value_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Loss L8 không hữu hạn.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for name, value in zip(sums, (loss, policy_loss, value_loss), strict=True):
            sums[name] += value.item()
    return {name: value / settings["steps_per_iter"] for name, value in sums.items()}


def evaluate(
    run: Run, candidate, reference, label: str, settings: dict, seed: int, iteration: int, backend: str = "auto",
) -> dict:
    """Đo điểm theo góc nhìn ứng viên ở ngân sách mô phỏng cố định."""
    count = settings["games"]
    swap = settings.get("swap_colors", True)
    openings = json.loads(project_path("runs/openings/openings_v1.json").read_text(encoding="utf-8"))
    opening_ids = random.Random(seed).sample(range(len(openings)), count // 2 if swap else count)
    limits = Limits(max_sims=settings["sims_per_move"])
    a = PUCTAgent(name="L8", limits=limits, model=candidate, backend=backend)
    b = PUCTAgent(name=label, limits=limits, model=reference, backend=backend)
    wins = draws = losses = 0
    for k in range(count):
        opening_id = opening_ids[k // 2 if swap else k]
        a_is_x = k % 2 == 0
        x, o = (a, b) if a_is_x else (b, a)
        game = play_match(x, o, openings[opening_id], seed, k)
        result = game["result"] * (1 if a_is_x else -1)
        wins += result == 1
        draws += result == 0
        losses += result == -1
        run.log_game({
            **game, "game_id": k, "opening_id": opening_id,
            "tournament_id": f"{label}-{iteration}", "agent_x": x.name, "agent_o": o.name,
        })
    return {"n": count, "wins": wins, "draws": draws, "losses": losses, "score": (wins + 0.5 * draws) / count}


def validate(cfg: dict) -> None:
    limits = {
        "selfplay": {"games_per_iter": 1, "sims_per_move": 1, "temperature_moves": 0, "torch_threads": 1},
        "buffer": {"window_games": 1},
        "train": {"batch_size": 1, "steps_per_iter": 1, "iterations": 1},
        "gating": {"games": 0, "sims_per_move": 1},
        "evaluation": {"games": 0, "sims_per_move": 1},
    }
    openings = json.loads(project_path("runs/openings/openings_v1.json").read_text(encoding="utf-8"))
    for section, keys in limits.items():
        for key, minimum in keys.items():
            value = cfg[section][key]
            if type(value) is not int or value < minimum:
                raise ValueError(f"{section}.{key} phải là số nguyên >= {minimum}.")
    for section in ("gating", "evaluation"):
        settings = cfg[section]
        games = settings["games"]
        swap = settings.get("swap_colors", True)
        if swap and games % 2:
            raise ValueError(f"{section}.games phải chẵn khi đổi màu.")
        if games > len(openings) * (2 if swap else 1):
            raise ValueError(f"{section}.games vượt số khai cuộc có sẵn.")
    if not 0 <= cfg["gating"]["threshold"] <= 1:
        raise ValueError("gating.threshold phải nằm trong [0, 1].")
    if cfg["train"]["optimizer"] != "adam":
        raise ValueError("L8 hiện hỗ trợ train.optimizer=adam.")
    noise = cfg["selfplay"]["dirichlet"]
    if noise and (not noise["alpha"] > 0 or not 0 <= noise["epsilon"] <= 1):
        raise ValueError("Dirichlet cần alpha > 0 và epsilon trong [0, 1].")


def train(cfg: dict) -> dict:
    validate(cfg)
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    rng = np.random.default_rng(seed)
    device_name = cfg["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.set_num_threads(cfg["selfplay"]["torch_threads"])

    sp, t, gate, final = (cfg[key] for key in ("selfplay", "train", "gating", "evaluation"))
    replay = deque(maxlen=cfg["buffer"]["window_games"])
    with Run(cfg) as run:
        model = build_model(cfg, device)
        source = load_init(model, t["init"], device)
        incumbent = copy.deepcopy(model)
        agent = PUCTAgent(limits=Limits(max_sims=sp["sims_per_move"]), model=model,
                          c_puct=sp["c_puct"], dirichlet=sp["dirichlet"], backend=sp.get("backend", "auto"))
        optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
        accepted = best_iteration = played = 0
        last_gate = None
        run.log.info("L8 init=%s, device=%s, PUCT backend=%s", source, device, agent.backend)
        save_checkpoint(run.checkpoint_path("best.pt"), incumbent, cfg, 0, {"iteration": 0})
        for iteration in range(1, t["iterations"] + 1):
            model.eval()
            run.log.info("L8 vòng %d: tự đấu %d ván", iteration, sp["games_per_iter"])
            for _ in range(sp["games_per_iter"]):
                samples, game = selfplay_game(agent, derive_seed(seed, "selfplay", played), sp["temperature_moves"])
                replay.append(samples)
                run.log_game({
                    **game, "game_id": played, "opening_id": None, "tournament_id": "selfplay",
                    "agent_x": "L8-selfplay", "agent_o": "L8-selfplay",
                })
                played += 1
            optimizer_before = copy.deepcopy(optimizer.state_dict()) if gate["games"] else None
            losses = train_replay(model, optimizer, replay, t, device, rng, cfg["buffer"]["augment_x8"])
            save_checkpoint(run.checkpoint_path("last.pt"), model, cfg, played, {"iteration": iteration})
            last_gate = None
            if gate["games"]:
                last_gate = evaluate(run, model, incumbent, "L8-incumbent", gate,
                                     derive_seed(seed, "gating", iteration), iteration, agent.backend)
            promote = last_gate is None or last_gate["score"] >= gate["threshold"]
            if promote:
                incumbent.load_state_dict(model.state_dict())
                accepted += 1
                best_iteration = iteration
                save_checkpoint(run.checkpoint_path("best.pt"), incumbent, cfg, played, {"iteration": iteration})
            else:
                model.load_state_dict(incumbent.state_dict())
                optimizer.load_state_dict(optimizer_before)
            run.log_metric(iteration, **losses, accepted=promote, gating=last_gate,
                           selfplay_games=played, replay_positions=sum(len(game[2]) for game in replay))
            run.log.info("L8 vòng %d: loss %.4f · duyệt=%s · điểm đối chứng=%s",
                         iteration, losses["loss"], promote, last_gate and last_gate["score"])

        against_l7 = None
        if final["games"]:
            l7 = CaroResNet().to(device)
            load_init(l7, "model:l5", device)
            against_l7 = evaluate(run, incumbent, l7, "L7-PUCT[l5]", final,
                                  derive_seed(seed, "l7"), best_iteration, agent.backend)
            run.log_metric(t["iterations"], evaluation=against_l7, best_iteration=best_iteration)
            run.log.info("L8 best.pt gặp L7: %s", against_l7)
        summary = {"selfplay_games": played, "accepted_iterations": accepted, "best_iteration": best_iteration,
                   "init_source": source, "last_gating": last_gate, "vs_l7": against_l7}
        run.finish("ok", **summary)
        return {**summary, "run_dir": str(run.dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Ví dụ: training/alphazero")
    parser.add_argument("overrides", nargs="*", help="Ví dụ: train.iterations=1")
    args = parser.parse_args(argv)
    print(train(load_config(args.config, args.overrides)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
