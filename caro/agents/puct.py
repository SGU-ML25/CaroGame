"""Cấp 7 — PUCT: MCTS dùng policy của mạng làm xác suất tiên nghiệm và value của mạng thay cho rollout.

Cùng kiểu AlphaGo/AlphaZero (mục 09): nút lá không rollout ngẫu nhiên mà hỏi mạng một lần, mạng trả
(policy trên 121 ô, v ∈ [-1, 1]). Chọn con theo

    score(a) = Q(a) + c_puct · P(a) · sqrt(N_cha) / (1 + N(a))

Quy ước dấu: mọi W/Q trong cây theo góc nhìn BÊN ĐANG ĐI tại nút đó (khớp value head của SL v3 / train_pg).
Giá trị của con theo góc nhìn cha = −W_con / N_con; khi lan ngược lên, đổi dấu ở mỗi tầng.

Gom lá: mỗi lượt chọn `batch_leaves` lá (virtual loss để các lá khác nhau) rồi suy luận mạng một lần theo lô.

Tham số (params trong configs/agents/l7_puct.yaml):
    checkpoint    -- `model:<tên>` | đường dẫn .pt | `hf://…` (như PolicyAgent); None = mạng khởi tạo ngẫu nhiên
    c_puct        -- hệ số khám phá (mặc định 1.5)
    leaf_eval     -- 'value' (v của mạng) | 'mix' ((1−λ)·v + λ·z_rollout, kiểu AlphaGo)
    mix_lambda    -- λ của 'mix'
    dirichlet     -- None, hoặc {alpha, epsilon}: nhiễu ở gốc (dùng khi tự chơi, KHÔNG dùng khi thi đấu)
    batch_leaves  -- số lá gom mỗi lần suy luận
    temperature   -- 0: chọn nước nhiều lượt thăm nhất; >0: lấy mẫu ∝ N^(1/τ) (tự chơi cấp 8)
    backend       -- 'python' | 'native' (Cython) | 'auto' (ưu tiên Cython nếu đã build)

Không có tiêu chí dừng nào (Limits rỗng) thì dừng sau DEFAULT_MAX_SIMS mô phỏng.
"""

from __future__ import annotations

import math
import time
from collections.abc import Generator, Sequence
from typing import Any

import numpy as np
import torch

from caro.agents.base import BaseAgent, Limits
from caro.agents.mcts import rollout_fast
from caro.engine import CaroGame
from caro.hub import resolve_checkpoint
from caro.nn.features import encode, legal_mask
from caro.nn.model import CaroResNet

try:
    from caro.engine import _puct_native
except ImportError:
    _puct_native = None

DEFAULT_MAX_SIMS = 200
EvalRequest = tuple[list[np.ndarray], list[np.ndarray]]
EvalResult = tuple[np.ndarray, np.ndarray]


class PUCTNode:
    """W, N theo góc nhìn bên đang đi tại nút; `vl` là số virtual loss đang treo (chỉ tồn tại trong một lượt gom lô)."""

    __slots__ = ("N", "W", "children", "move", "prior", "vl")

    def __init__(self, move: int | None, prior: float) -> None:
        self.move = move
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.vl = 0
        self.children: list[PUCTNode] = []

    @property
    def expanded(self) -> bool:
        return bool(self.children)


def terminal_value(game: CaroGame) -> float:
    """Giá trị nút kết thúc theo góc nhìn bên đang đi: có người thắng thì đó là bên vừa đi (side đã đổi) → −1; hoà → 0."""
    return -1.0 if game.winner is not None else 0.0


def select_child(node: PUCTNode, c_puct: float) -> PUCTNode:
    sqrt_n = math.sqrt(max(1, node.N + node.vl))
    best, best_score = node.children[0], -math.inf
    for ch in node.children:
        n = ch.N + ch.vl
        # virtual loss = bên chọn (cha) thua → bên đi ở con thắng: cộng +1 mỗi lượt vào W của con
        q = -(ch.W + ch.vl) / n if n else 0.0
        score = q + c_puct * ch.prior * sqrt_n / (1 + n)
        if score > best_score:
            best, best_score = ch, score
    return best


class PUCTAgent(BaseAgent):
    def __init__(
        self,
        name: str | None = None,
        limits: Limits | None = None,
        checkpoint: str | None = None,
        c_puct: float = 1.5,
        leaf_eval: str = "value",
        mix_lambda: float = 0.5,
        dirichlet: dict | None = None,
        batch_leaves: int = 8,
        temperature: float = 0.0,
        channels: int = 64,
        blocks: int = 6,
        model: CaroResNet | None = None,
        backend: str = "auto",
    ) -> None:
        super().__init__(name, limits)
        if leaf_eval not in ("value", "mix"):
            raise ValueError(f"leaf_eval phải là 'value' hoặc 'mix', nhận {leaf_eval!r}")
        if batch_leaves < 1:
            raise ValueError("batch_leaves phải ≥ 1")
        if backend not in ("python", "native", "auto"):
            raise ValueError("backend phải là python | native | auto")
        if backend == "native" and _puct_native is None:
            raise ValueError("backend='native' cần build bằng python setup_native.py build_ext --inplace")
        self.backend = "native" if backend != "python" and _puct_native is not None else "python"
        native = _puct_native if self.backend == "native" else None
        self._node = native.PUCTNode if native else PUCTNode
        self._select_child = native.select_child if native else select_child
        self._make_children = native.make_children if native else None
        self.c_puct = float(c_puct)
        self.leaf_eval = leaf_eval
        self.mix_lambda = float(mix_lambda)
        self.dirichlet = dirichlet
        self.batch_leaves = int(batch_leaves)
        self.temperature = float(temperature)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is not None:                      # dùng chung mạng đang huấn luyện (vòng AlphaZero)
            self.model = model
            self.device = next(model.parameters()).device
        else:
            self.model = CaroResNet({"num_channels": channels, "num_blocks": blocks}).to(self.device)
            if checkpoint:
                state = torch.load(resolve_checkpoint(checkpoint), map_location=self.device)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                self.model.load_state_dict(state)
        self.model.eval()
        self.last_pi: np.ndarray | None = None     # phân phối thăm ở gốc (121,) — mẫu huấn luyện cấp 8

    # ---- mạng -------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, xs: list[np.ndarray], masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Lô thế cờ → (xác suất policy (B,121) đã mask, value (B,) theo góc nhìn bên đi)."""
        x = torch.from_numpy(np.stack(xs)).to(self.device)
        m = torch.from_numpy(np.stack(masks)).to(self.device)
        logits, v = self.model(x, m)
        return torch.softmax(logits, dim=1).cpu().numpy(), v.squeeze(1).cpu().numpy()

    def _expand(self, node: PUCTNode, legal: np.ndarray, probs: np.ndarray) -> None:
        cells = np.flatnonzero(legal)
        p = probs[cells]
        total = float(p.sum())
        p = p / total if total > 0 else np.full(len(cells), 1.0 / len(cells))
        if self._make_children is not None:
            node.children = self._make_children(cells, p.astype(np.float64, copy=False))
        else:
            node.children = [PUCTNode(int(c), float(pc)) for c, pc in zip(cells, p, strict=True)]

    def _leaf_value(self, game: CaroGame | None, v: float) -> float:
        if game is None:
            return v
        z = rollout_fast(game.bb[0], game.bb[1], game.side, self.rng, "light", 1)   # tuyệt đối: +1 nếu X thắng
        z_me = z if game.side == 0 else -z
        return (1 - self.mix_lambda) * v + self.mix_lambda * z_me

    # ---- tìm kiếm -----------------------------------------------------------
    def _add_noise(self, root: PUCTNode) -> None:
        alpha, eps = float(self.dirichlet["alpha"]), float(self.dirichlet["epsilon"])
        noise = [self.rng.gammavariate(alpha, 1.0) for _ in root.children]
        total = sum(noise) or 1.0
        for ch, g in zip(root.children, noise, strict=True):
            ch.prior = (1 - eps) * ch.prior + eps * g / total

    def select_action(self, game: CaroGame, limits: Limits) -> int:
        search = self._search(game, limits)
        try:
            request = next(search)
            while True:
                request = search.send(self._evaluate(*request))
        except StopIteration as finished:
            return finished.value

    def _search(self, game: CaroGame, limits: Limits) -> Generator[EvalRequest, EvalResult, int]:
        """Nhường quyền tại mỗi lần suy luận để có thể ghép lá từ nhiều cây."""
        if game.is_terminal():
            return -1
        t0 = time.perf_counter()
        deadline = limits.deadline(margin_ms=5.0)
        max_sims = limits.max_sims if limits.max_sims is not None else self.limits.max_sims
        if max_sims is None and limits.time_ms is None:
            max_sims = DEFAULT_MAX_SIMS

        work = game.clone()
        root = self._node(None, 1.0)
        root_mask = legal_mask(work)
        probs, vals = yield [encode(work)], [root_mask]
        self._expand(root, root_mask, probs[0])
        root_value = float(vals[0])
        priors = {ch.move: ch.prior for ch in root.children}
        if self.dirichlet:
            self._add_noise(root)

        sims = 0
        while True:
            if max_sims is not None and sims >= max_sims:
                break
            if limits.time_ms is not None and deadline.expired() and sims > 0:
                break
            budget = self.batch_leaves if max_sims is None else min(self.batch_leaves, max_sims - sims)
            pending: list[tuple[list[PUCTNode], PUCTNode, np.ndarray, np.ndarray, CaroGame | None]] = []
            for _ in range(budget):
                path = [root]
                node = root
                depth = 0
                while node.expanded:
                    node = self._select_child(node, self.c_puct)
                    path.append(node)
                    work.step(node.move)
                    depth += 1
                if work.is_terminal():
                    self._backup(path, terminal_value(work), release=False)
                    sims += 1
                else:
                    for n in path:                           # virtual loss: các lá trong lô khác nhau
                        n.vl += 1
                    pending.append((path, node, encode(work), legal_mask(work),
                                    work.clone() if self.leaf_eval == "mix" else None))
                for _ in range(depth):
                    work.undo()
            if pending:
                probs, vals = yield [p[2] for p in pending], [p[3] for p in pending]
                for (path, node, _x, mask, game_leaf), pr, v in zip(pending, probs, vals, strict=True):
                    if not node.expanded:                    # lá trùng trong cùng lô chỉ mở rộng một lần
                        self._expand(node, mask, pr)
                    self._backup(path, self._leaf_value(game_leaf, float(v)), release=True)
                    sims += 1

        return self._finish(root, game, t0, sims, root_value, priors)

    @staticmethod
    def _backup(path: list[PUCTNode], value: float, release: bool) -> None:
        """`value` theo góc nhìn bên đi ở lá; đổi dấu mỗi tầng đi lên."""
        for node in reversed(path):
            node.N += 1
            node.W += value
            if release:
                node.vl -= 1
            value = -value

    def _finish(self, root: PUCTNode, game: CaroGame, t0: float, sims: int, root_value: float,
                priors: dict[int, float]) -> int:
        visits = {ch.move: ch.N for ch in root.children if ch.N > 0}
        pi = np.zeros(121, dtype=np.float32)
        for mv, n in visits.items():
            pi[mv] = n
        if self.temperature > 0 and visits:
            w = pi ** (1.0 / self.temperature)
            move = int(self.rng.choices(range(121), weights=w.tolist())[0])
            pi = w / w.sum()
        else:
            best = max(root.children, key=lambda ch: (ch.N, -ch.W / ch.N if ch.N else -math.inf))
            move = best.move
            pi = pi / pi.sum() if pi.sum() > 0 else pi
        self.last_pi = pi

        pv: list[int] = []
        node = next(ch for ch in root.children if ch.move == move)
        pv.append(node.move)
        while node.children and any(c.N for c in node.children):
            node = max(node.children, key=lambda c: c.N)
            pv.append(node.move)
        chosen = next(ch for ch in root.children if ch.move == move)
        stats: dict[str, Any] = {
            "sims": sims, "time_ms": round((time.perf_counter() - t0) * 1000, 2),
            "visits": visits, "pv": pv, "policy": {m: p for m, p in priors.items() if p > 1e-4},
            "value": root_value,
        }
        if chosen.N:
            stats["score"] = round(-chosen.W / chosen.N, 4)       # theo góc nhìn bên đi ở gốc
        self._stats = stats
        return move


def select_actions(agents: Sequence[PUCTAgent], games: Sequence[CaroGame]) -> list[int]:
    """Ghép lượt suy luận giữa các ván dùng cùng mạng và ngân sách cố định."""
    matches = list(zip(agents, games, strict=True))
    if not matches:
        return []
    if len(matches) == 1:
        agent, game = matches[0]
        return [agent.select_action(game, agent.limits)]
    if len(set(agents)) != len(agents):
        raise ValueError("Mỗi ván cần tác tử PUCT riêng để giữ RNG và thống kê.")
    if any(a.model is not agents[0].model or a.limits.time_ms is not None for a in agents):
        raise ValueError("PUCT ghép lô cần dùng chung mạng và không đặt giới hạn thời gian.")
    active = [(i, a._search(g, a.limits), None) for i, (a, g) in enumerate(matches)]
    moves = [-1] * len(matches)
    while active:
        waiting, states, masks = [], [], []
        for i, search, result in active:
            try:
                xs, ms = search.send(result)
            except StopIteration as finished:
                moves[i] = finished.value
                continue
            start = len(states)
            states.extend(xs)
            masks.extend(ms)
            waiting.append((i, search, slice(start, len(states))))
        if not waiting:
            break
        probs, values = agents[0]._evaluate(states, masks)
        active = [(i, search, (probs[span], values[span])) for i, search, span in waiting]
    return moves
