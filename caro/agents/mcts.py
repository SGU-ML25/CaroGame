"""Tác tử cấp 4 — Monte Carlo Tree Search (UCT) cho Caro 11×11 (mục 06.1–06.2, issue #13).

Bốn bước của mỗi lần lặp:
1. Selection: đi xuống theo UCT: W/N + c_uct * sqrt(ln(N_cha) / N)
2. Expansion: mở rộng 1 nước chưa thử trong candidates(candidate_radius)
3. Simulation (Rollout): chơi đến hết ván theo chính sách 'random' hoặc 'light'
4. Backpropagation: cộng kết quả, đổi góc nhìn theo mover từng tầng

Config: configs/agents/l4_mcts.yaml, kind=mcts.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

from caro.agents.base import BaseAgent, Limits
from caro.engine import CaroGame
from caro.engine.bitboard import (
    FULL,
    LINES,
    WIN,
    exactly,
    has_five,
    iter_bits,
    window_counts,
)
from caro.engine.bitboard import (
    candidates as bb_candidates,
)


def pick_random_bit(b: int, count: int, rng: random.Random) -> int:
    """Chọn ngẫu nhiên 1 bit 1 từ bitmask b có đúng count bit 1."""
    n = rng.randrange(count)
    for _ in range(n):
        b &= b - 1
    return (b & -b).bit_length() - 1


def first_winning_square(me: int, opp: int) -> int | None:
    """Trả về một ô thắng ngay nếu có, hoặc None."""
    empty = ~(me | opp) & FULL
    for d, start in LINES:
        w4 = exactly(4, *window_counts(me, opp, d, start))
        if w4:
            for j in range(WIN):
                sq = (w4 << (j * d)) & empty
                if sq:
                    return (sq & -sq).bit_length() - 1
    return None


def rollout_fast(
    bb_x: int,
    bb_o: int,
    side: int,
    rng: random.Random,
    rollout_mode: str,
    rollout_radius: int,
) -> int:
    """Mô phỏng đến hết ván. Trả kết quả theo góc nhìn X: +1 X thắng, -1 O thắng, 0 hoà.

    Kiểm tra kết thúc: chỉ gọi has_five trên bitboard của người vừa đi (mục 06.2).
    """
    bb = [bb_x, bb_o]
    cur_side = side
    while True:
        m_me = bb[cur_side]
        m_opp = bb[cur_side ^ 1]
        empty = ~(m_me | m_opp) & FULL
        if not empty:
            return 0

        move = None
        if rollout_mode == "light":
            # 1. Thắng ngay
            move = first_winning_square(m_me, m_opp)
            if move is None:
                # 2. Chặn thua ngay
                move = first_winning_square(m_opp, m_me)

        if move is None:
            # 3. Ngẫu nhiên trong bán kính rollout_radius
            cands = bb_candidates(m_me, m_opp, rollout_radius)
            cnt = cands.bit_count()
            if not cnt:
                cands = empty
                cnt = empty.bit_count()
            move = pick_random_bit(cands, cnt, rng)

        bb[cur_side] |= 1 << move
        if has_five(bb[cur_side]):
            return 1 if cur_side == 0 else -1
        cur_side ^= 1


def candidate_moves(game: CaroGame, radius: int) -> list[int]:
    """Tập ứng viên mở rộng: ô trống trong bán kính Chebyshev quanh các quân đã đặt."""
    if game.is_terminal():
        return []
    me, opp = game.me_opp()
    cands = bb_candidates(me, opp, radius)
    if cands:
        return list(iter_bits(cands))
    return game.legal_actions()


class MCTSNode:
    """Nút trong cây tìm kiếm MCTS."""

    __slots__ = ("N", "W", "children", "key", "move", "mover", "parent", "untried")

    def __init__(
        self,
        parent: MCTSNode | None,
        move: int | None,
        mover: int | None,
        untried: list[int],
        key: int = 0,
    ) -> None:
        self.parent = parent
        self.move = move
        self.mover = mover
        self.untried = untried
        self.children: list[MCTSNode] = []
        self.N = 0
        self.W = 0.0
        self.key = key

    def add_child(self, move: int, mover: int, untried: list[int], key: int = 0) -> MCTSNode:
        child = MCTSNode(parent=self, move=move, mover=mover, untried=untried, key=key)
        self.children.append(child)
        return child


class MCTSAgent(BaseAgent):
    """Tác tử MCTS (UCT) với rollout ngẫu nhiên / nhẹ và tuỳ chọn tái sử dụng cây.

    Tham số (đọc từ params trong config YAML):
        c_uct            -- hệ số khám phá trong công thức UCT (mặc định 1.4)
        rollout          -- 'random' hoặc 'light' (thắng ngay → chặn → ngẫu nhiên)
        rollout_radius   -- bán kính cho bước ngẫu nhiên trong rollout (mặc định 1)
        candidate_radius -- bán kính cho các ứng viên mở rộng nút (mặc định 2)
        reuse_tree       -- cờ bật/tắt tái sử dụng cây giữa các nước (mặc định False)
    """

    def __init__(
        self,
        name: str | None = None,
        limits: Limits | None = None,
        c_uct: float = 1.4,
        rollout: str = "random",
        rollout_radius: int = 1,
        candidate_radius: int = 2,
        reuse_tree: bool = False,
    ) -> None:
        super().__init__(name=name, limits=limits)
        self.c_uct = c_uct
        if rollout not in ("random", "light"):
            raise ValueError(f"rollout phải là 'random' hoặc 'light', nhận {rollout!r}")
        self.rollout_mode = rollout
        self.rollout_radius = rollout_radius
        self.candidate_radius = candidate_radius
        self.reuse_tree = reuse_tree

        self._root: MCTSNode | None = None
        self._stats: dict[str, Any] = {}

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self._root = None
        self._stats = {}

    def _locate_root(self, game: CaroGame) -> MCTSNode | None:
        """Tìm nút gốc tái sử dụng tương ứng với trạng thái bàn cờ hiện tại."""
        if not self.reuse_tree or self._root is None:
            return None
        if self._root.key == game.key:
            self._root.parent = None
            return self._root
        for ch in self._root.children:
            if ch.key == game.key:
                ch.parent = None
                return ch
            # Tìm sâu thêm 1 tầng đối thủ (nếu agent đi nước rồi đối thủ đáp)
            for gch in ch.children:
                if gch.key == game.key:
                    gch.parent = None
                    return gch
        return None

    def select_action(self, game: CaroGame, limits: Limits) -> int:
        """Chọn nước đi bằng MCTS. Trả chỉ số ô (0–120)."""
        if game.is_terminal():
            return -1

        t0 = time.perf_counter()
        deadline = limits.deadline(margin_ms=5.0)
        mode = limits.mode
        max_sims = limits.max_sims if limits.max_sims is not None else self.limits.max_sims

        root = self._locate_root(game)
        if root is None:
            root = MCTSNode(
                parent=None,
                move=None,
                mover=game.side ^ 1,
                untried=candidate_moves(game, self.candidate_radius),
                key=game.key,
            )

        sims = 0
        c_uct = self.c_uct
        rng = self.rng
        rollout_mode = self.rollout_mode
        rollout_radius = self.rollout_radius
        candidate_radius = self.candidate_radius

        # Vòng lặp mô phỏng
        while True:
            if max_sims is not None and sims >= max_sims:
                break
            if mode == "time" and deadline.expired() and sims > 0:
                break
            if limits.time_ms is not None and deadline.expired() and sims > 0:
                break

            # 1. Selection
            g = game.clone()
            node = root
            while not node.untried and node.children:
                best_score = -float("inf")
                best_ch = None
                log_N = math.log(node.N)
                for ch in node.children:
                    uct = (ch.W / ch.N) + c_uct * math.sqrt(log_N / ch.N)
                    if uct > best_score:
                        best_score = uct
                        best_ch = ch
                node = best_ch
                g.step(node.move)

            # 2. Expansion
            if node.untried and not g.is_terminal():
                idx = rng.randrange(len(node.untried))
                m = node.untried.pop(idx)
                g.step(m)
                child_untried = candidate_moves(g, candidate_radius)
                node = node.add_child(move=m, mover=g.side ^ 1, untried=child_untried, key=g.key)

            # 3. Simulation
            if g.is_terminal():
                z = g.result()
            else:
                z = rollout_fast(g.bb[0], g.bb[1], g.side, rng, rollout_mode, rollout_radius)

            # 4. Backpropagation
            curr = node
            while curr is not None:
                curr.N += 1
                if curr.mover is not None:
                    curr.W += z if curr.mover == 0 else -z
                curr = curr.parent

            sims += 1

        if not root.children:
            if root.untried:
                chosen_move = rng.choice(root.untried)
            else:
                legal = game.legal_actions()
                chosen_move = legal[0] if legal else -1
            best_child = None
        else:
            # Chọn con có N lớn nhất, giải hoà bằng W
            best_child = max(root.children, key=lambda ch: (ch.N, ch.W))
            chosen_move = best_child.move

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Thống kê
        visits = {ch.move: ch.N for ch in root.children}
        pv: list[int] = []
        if best_child is not None:
            curr = best_child
            pv.append(curr.move)
            while curr.children:
                curr = max(curr.children, key=lambda ch: (ch.N, ch.W))
                pv.append(curr.move)

        self._stats = {
            "sims": sims,
            "time_ms": round(elapsed_ms, 2),
            "visits": visits,
            "pv": pv,
        }
        if best_child is not None and best_child.N > 0:
            self._stats["score"] = round(best_child.W / best_child.N, 4)

        # Lưu lại cho lượt kế tiếp nếu tái sử dụng cây
        if self.reuse_tree and best_child is not None:
            self._root = best_child
        else:
            self._root = None

        return chosen_move
