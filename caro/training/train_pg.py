"""Cấp 6 — REINFORCE có baseline (policy gradient tự chơi), issue #21, config `training/pg`.

    uv run python -m caro.training.train_pg training/pg                     # khởi tạo từ model:l5
    uv run python -m caro.training.train_pg training/pg train.init=random   # A5b: từ đầu
    uv run python -m caro.training.train_pg experiments/a5b_rl_init         # A5b: từ cấp 5

Mỗi vòng lặp chơi `batch_games` ván song song (suy luận theo lô cho mọi ván đang chơi) rồi cập nhật một bước:

    L = -mean[(z - v(s)) * log π(a|s)]  +  c_v * mean[(v(s) - z)^2]  -  β * H(π(.|s))

z ∈ {+1, 0, -1} là kết quả ván theo góc nhìn bên vừa đi nước đó (gamma = 1, thưởng chỉ ở cuối ván), v(s) là value head
dùng làm baseline (không truyền gradient của policy vào v), β = entropy_coef giảm dần theo `entropy_decay`.

Đối thủ mỗi ván: bản mới nhất (tự chơi, cả hai bên đều được học) hoặc một checkpoint cũ trong pool (đóng băng, chỉ học
nước của bên học). Mỗi `eval_every` episode: đóng băng mạng, đấu `eval_games` ván đổi màu (argmax) với từng tác tử trong
`eval_vs` rồi ghi điểm + khoảng Wilson vào metrics.jsonl (learning curve).

Đặc trưng đầu vào trùng `caro.nn.features.board_to_tensor` (kiểm bằng tests/test_train_pg.py) nhưng tính nhanh bằng numpy.
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from caro.agents import make_agent
from caro.agents.base import BaseAgent, Limits
from caro.arena.stats import wilson
from caro.config import load_config
from caro.engine.game import CaroGame, N
from caro.hub import resolve_checkpoint
from caro.nn.features import bits, encode, legal_mask  # noqa: F401  (tests/test_train_pg import từ đây)
from caro.nn.model import CaroResNet
from caro.runlog import Run, derive_seed

OPENING_REGION = range(3, 8)      # khai cuộc ngẫu nhiên trong vùng 5×5 giữa bàn


# ---------------------------------------------------------------------------
# Đặc trưng và chọn nước theo lô
# ---------------------------------------------------------------------------

@torch.no_grad()
def policy_step(
    net: CaroResNet, games: list[CaroGame], temperature: float, device: torch.device, greedy: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Chọn một nước cho mỗi ván trong lô. Trả (đặc trưng, mask hợp lệ, nước đi)."""
    xs = np.stack([encode(g) for g in games])
    ms = np.stack([legal_mask(g) for g in games])
    logits, _ = net(torch.from_numpy(xs).to(device), torch.from_numpy(ms).to(device))
    if greedy:
        acts = logits.argmax(dim=1)
    else:
        acts = torch.multinomial(torch.softmax(logits / temperature, dim=1), 1).squeeze(1)
    return xs, ms, acts.tolist()


def random_opening(game: CaroGame, plies: int, rng: random.Random) -> None:
    for _ in range(plies):
        cells = [r * N + c for r in OPENING_REGION for c in OPENING_REGION if not (game.occupied >> (r * N + c)) & 1]
        if not cells or game.is_terminal():
            return
        game.step(rng.choice(cells))


# ---------------------------------------------------------------------------
# Tự chơi một vòng lặp và cập nhật
# ---------------------------------------------------------------------------

def play_iteration(
    model: CaroResNet, opp_model: CaroResNet | None, n_games: int, p_latest: float, temperature: float,
    opening_plies: int, device: torch.device, rng: random.Random,
) -> dict:
    """Chơi n_games ván song song. Trả các mẫu (x, mask, action, z) của những nước được học + thống kê."""
    games = [CaroGame() for _ in range(n_games)]
    self_play = [opp_model is None or rng.random() < p_latest for _ in range(n_games)]
    learner = [i % 2 for i in range(n_games)]            # bên học xen kẽ X/O
    for g in games:
        random_opening(g, opening_plies, rng)
    traj: list[list[tuple]] = [[] for _ in games]
    active = [i for i, g in enumerate(games) if not g.is_terminal()]

    model.eval()
    if opp_model is not None:
        opp_model.eval()
    while active:
        mine = [i for i in active if self_play[i] or games[i].side == learner[i]]
        mine_set = set(mine)
        theirs = [i for i in active if i not in mine_set]
        for idxs, net, record in ((mine, model, True), (theirs, opp_model, False)):
            if not idxs:
                continue
            xs, ms, acts = policy_step(net, [games[i] for i in idxs], temperature, device)
            for k, i in enumerate(idxs):
                if record:
                    traj[i].append((xs[k], ms[k], acts[k], games[i].side))
                games[i].step(acts[k])
        active = [i for i in active if not games[i].is_terminal()]

    xs, ms, acts, zs = [], [], [], []
    wins = draws = losses = 0                              # bên học trong các ván với checkpoint cũ
    for i, g in enumerate(games):
        for x, m, a, mover in traj[i]:
            xs.append(x)
            ms.append(m)
            acts.append(a)
            zs.append(0.0 if g.winner is None else (1.0 if g.winner == mover else -1.0))
        if not self_play[i]:
            if g.winner is None:
                draws += 1
            elif g.winner == learner[i]:
                wins += 1
            else:
                losses += 1
    return {
        "x": np.stack(xs), "mask": np.stack(ms), "action": np.asarray(acts, dtype=np.int64),
        "z": np.asarray(zs, dtype=np.float32),
        "n_self": sum(self_play), "wins": wins, "draws": draws, "losses": losses,
        "plies": sum(len(g.history) for g in games),
    }


def update(
    model: CaroResNet, optimizer: torch.optim.Optimizer, batch: dict, temperature: float, entropy_coef: float,
    value_coef: float, grad_clip: float | None, device: torch.device, chunk: int = 1024,
) -> dict:
    """Một bước cập nhật trên toàn bộ mẫu của vòng lặp; chia lô con `chunk` mẫu (cộng dồn gradient) để vừa VRAM."""
    model.train()
    n = len(batch["z"])
    optimizer.zero_grad()
    sums = dict.fromkeys(("loss", "policy_loss", "value_loss", "entropy"), 0.0)
    for lo in range(0, n, chunk):
        sl = slice(lo, lo + chunk)
        x = torch.from_numpy(batch["x"][sl]).to(device)
        mask = torch.from_numpy(batch["mask"][sl]).to(device)
        action = torch.from_numpy(batch["action"][sl]).to(device)
        z = torch.from_numpy(batch["z"][sl]).to(device)

        logits, v = model(x, mask)
        v = v.squeeze(1)
        logp = F.log_softmax(logits / temperature, dim=1)
        chosen = logp.gather(1, action[:, None]).squeeze(1)
        advantage = z - v.detach()                          # baseline: value head, không chảy gradient vào policy
        policy_loss = -(advantage * chosen).mean()
        value_loss = F.mse_loss(v, z)
        entropy = -(logp.exp() * logp.masked_fill(~mask, 0.0)).sum(dim=1).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        w = len(z) / n                                      # trung bình theo mẫu, không theo lô con
        (loss * w).backward()
        for k, val in (("loss", loss), ("policy_loss", policy_loss), ("value_loss", value_loss), ("entropy", entropy)):
            sums[k] += val.item() * w
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip if grad_clip else float("inf")))
    optimizer.step()
    return {**sums, "grad_norm": grad_norm, "mean_z": float(batch["z"].mean())}


# ---------------------------------------------------------------------------
# Đánh giá (đóng băng mạng, argmax)
# ---------------------------------------------------------------------------

class NetAgent(BaseAgent):
    """Bọc mạng policy thành tác tử (argmax, không tìm kiếm) để đấu trong vòng đánh giá."""

    def __init__(self, net: CaroResNet, device: torch.device, name: str = "L6-PolicyRL") -> None:
        super().__init__(name=name)
        self.net = net
        self.device = device

    def select_action(self, game: CaroGame, limits: Limits) -> int:
        self.net.eval()
        return policy_step(self.net, [game], 1.0, self.device, greedy=True)[2][0]


def evaluate(net: CaroResNet, opponent_ref: str | dict, games: int, device: torch.device, seed: int) -> dict:
    """`games` ván đổi màu với cùng một bộ khai cuộc ngẫu nhiên. Điểm = (thắng + 0,5·hoà) / tổng."""
    agent = NetAgent(net, device)
    opp = make_agent(opponent_ref)
    wins = draws = losses = 0
    for k in range(games):
        rng = random.Random(derive_seed(seed, k // 2))      # hai ván của một cặp dùng chung khai cuộc
        game = CaroGame()
        random_opening(game, 2, rng)
        learner = k % 2
        agent.reset(derive_seed(seed, k, "net"))
        opp.reset(derive_seed(seed, k, "opp"))
        while not game.is_terminal():
            who, lim = (agent, agent.limits) if game.side == learner else (opp, opp.limits)
            game.step(who.select_action(game, lim))
        if game.winner is None:
            draws += 1
        elif game.winner == learner:
            wins += 1
        else:
            losses += 1
    n = wins + draws + losses
    score = (wins + 0.5 * draws) / n if n else float("nan")
    low, high = wilson(wins + 0.5 * draws, n) if n else (float("nan"), float("nan"))
    return {"n": n, "wins": wins, "draws": draws, "losses": losses, "score": score, "ci_low": low, "ci_high": high}


# ---------------------------------------------------------------------------
# Vòng huấn luyện
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device) -> CaroResNet:
    m = cfg.get("model") or {}
    return CaroResNet({"num_channels": m.get("channels", 64), "num_blocks": m.get("blocks", 6)}).to(device)


def load_init(model: CaroResNet, init: str, device: torch.device) -> str:
    if init == "random":
        return "random"
    path = resolve_checkpoint(init)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)
    return str(path)


def save_checkpoint(path: Path, model: CaroResNet, cfg: dict, episode: int, extra: dict | None = None) -> None:
    m = cfg.get("model") or {}
    torch.save({
        "model_state_dict": model.state_dict(), "episode": episode,
        "config": {"num_channels": m.get("channels", 64), "num_blocks": m.get("blocks", 6)}, **(extra or {}),
    }, path)


def train(cfg: dict) -> dict:
    t = cfg["train"]
    seed = cfg["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    batch_games = t["batch_games"]
    iterations = max(1, t["episodes"] // batch_games)
    eval_every = t["eval_every"]
    pool_cfg = t.get("opponent_pool") or {"latest": 1.0}
    p_latest = pool_cfg.get("latest", 0.5)
    pool: deque = deque(maxlen=max(1, cfg["checkpoint"].get("keep_last", 5)))
    snapshot_every = t.get("snapshot_every", 5)            # vòng lặp; mỗi lần đánh giá cũng thêm một bản

    with Run(cfg) as run:
        model = build_model(cfg, device)
        src = load_init(model, t["init"], device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
        entropy_coef = t["entropy_coef"]
        pool.append(copy.deepcopy(model.state_dict()))
        opp_model = build_model(cfg, device)
        run.log.info("cấp 6 REINFORCE: init=%s (%s), device=%s, %d vòng × %d ván, tham số=%d",
                     t["init"], src, device, iterations, batch_games, sum(p.numel() for p in model.parameters()))

        episodes = 0
        total_plies = 0
        next_eval = eval_every
        best = -1.0
        last_eval: dict = {}
        t0 = time.monotonic()
        for it in range(1, iterations + 1):
            use_pool = p_latest < 1.0 and len(pool) > 0
            if use_pool:
                opp_model.load_state_dict(random.choice(list(pool)))
            batch = play_iteration(
                model, opp_model if use_pool else None, batch_games, p_latest, t["temperature"],
                t.get("random_opening_plies", 2), device, rng)
            stats = update(model, optimizer, batch, t["temperature"], entropy_coef, t.get("value_coef", 1.0),
                           t.get("grad_clip", 1.0), device, t.get("update_chunk", 1024))
            entropy_coef *= t.get("entropy_decay", 1.0)
            episodes += batch_games
            total_plies += batch["plies"]
            n_pool = batch["wins"] + batch["draws"] + batch["losses"]
            run.log_metric(
                episodes, **stats, entropy_coef=entropy_coef, iteration=it, samples=len(batch["z"]),
                train_win_vs_pool=(batch["wins"] + 0.5 * batch["draws"]) / n_pool if n_pool else None)
            if it % snapshot_every == 0:
                pool.append(copy.deepcopy(model.state_dict()))

            if episodes >= next_eval or it == iterations:
                next_eval += eval_every
                pool.append(copy.deepcopy(model.state_dict()))
                scores = []
                values = {}
                for ref in t["eval_vs"]:
                    ev = evaluate(model, ref, t["eval_games"], device, derive_seed(seed, "eval", episodes))
                    tag = ref["id"] if isinstance(ref, dict) else ref.split("/")[-1]
                    values.update({f"{tag}_{k}": v for k, v in ev.items()})
                    scores.append(ev["score"])
                    run.log.info("episode %d vs %s: %dW %dD %dL, score %.3f [%.3f; %.3f]", episodes, tag,
                                 ev["wins"], ev["draws"], ev["losses"], ev["score"], ev["ci_low"], ev["ci_high"])
                mean_score = float(np.mean(scores)) if scores else float("nan")
                run.log_metric(episodes, eval=True, score_mean=mean_score, **values)
                last_eval = values
                save_checkpoint(run.checkpoint_path("last.pt"), model, cfg, episodes)
                if scores and mean_score > best:
                    best = mean_score
                    save_checkpoint(run.checkpoint_path("best.pt"), model, cfg, episodes, {"score_mean": mean_score})
        if not run.checkpoint_path("best.pt").exists():
            save_checkpoint(run.checkpoint_path("best.pt"), model, cfg, episodes)
        elapsed = time.monotonic() - t0
        summary = {"episodes": episodes, "plies": total_plies, "init": t["init"], "best_score_mean": best,
                   "last_eval": last_eval, "train_seconds": round(elapsed, 1), "device": str(device)}
        run.finish("ok", **summary)
        summary["run_dir"] = str(run.dir)
        return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cấp 6 — REINFORCE có baseline")
    ap.add_argument("config", help="vd: training/pg hoặc experiments/a5b_rl_init")
    ap.add_argument("overrides", nargs="*", help="a.b=c, vd: train.init=random run.tag=from_random")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    summary = train(cfg)
    print({k: v for k, v in summary.items() if k != "last_eval"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
