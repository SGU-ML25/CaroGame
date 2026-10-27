"""Tác tử cấp 3 — Negamax Alpha-Beta, iterative deepening, Transposition Table (issue #10).

Mục tiêu kép: tác tử tìm kiếm chính + expert sinh dữ liệu cho cấp 5 (mục 05.1–05.2).
Config: configs/agents/l3_alphabeta.yaml, kind=alphabeta.
"""

from __future__ import annotations

import time
from typing import Any

from caro.agents.base import BaseAgent, Limits
from caro.engine import CaroGame
from caro.engine.bitboard import (
    LINES,
    WEIGHT,
    evaluate,
    exactly,
    has_five,
    iter_bits,
    popcount,
    side_score,
    window_counts,
    winning_squares,
)
from caro.engine.bitboard import (
    candidates as bb_candidates,
)
from caro.engine.zobrist import SIDE_KEY, ZOBRIST

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

INF = 10_000_000
WIN_SCORE = 1_000_000   # điểm thắng; trừ ply để ưu tiên thắng nhanh

# Cờ bảng chuyển vị
EXACT = 0
LOWER = 1   # alpha cut — giá trị thực ≥ value (lower bound)
UPPER = 2   # beta cut  — giá trị thực ≤ value (upper bound)


# ---------------------------------------------------------------------------
# Helper: phát hiện tứ và đòn kép
# ---------------------------------------------------------------------------

def _makes_four(me: int, opp: int) -> bool:
    """me có ít nhất 1 cửa sổ 4 quân sống không?"""
    for d, start in LINES:
        if exactly(4, *window_counts(me, opp, d, start)):
            return True
    return False


def _double_attack(me: int, opp: int) -> bool:
    """me có ≥2 ô thắng ngay (đòn kép) không?"""
    return popcount(winning_squares(me, opp)) >= 2


# ---------------------------------------------------------------------------
# Transposition Table
# ---------------------------------------------------------------------------

class TranspositionTable:
    """Bảng băm Zobrist kích thước cố định (replace-always)."""

    def __init__(self, size_mb: int = 64) -> None:
        self._size = max(1, (size_mb * 1024 * 1024) // 200)
        self._table: dict[int, tuple[int, int, int, int]] = {}

    def get(self, key: int) -> tuple[int, int, int, int] | None:
        """Trả (value, depth, flag, move) hoặc None nếu không có."""
        return self._table.get(key)

    def put(self, key: int, value: int, depth: int, flag: int, move: int) -> None:
        if len(self._table) >= self._size:
            self._table.pop(next(iter(self._table)))
        self._table[key] = (value, depth, flag, move)

    def clear(self) -> None:
        self._table.clear()


# ---------------------------------------------------------------------------
# AlphaBetaAgent
# ---------------------------------------------------------------------------

class AlphaBetaAgent(BaseAgent):
    """Negamax Alpha-Beta với iterative deepening và Transposition Table.

    Tham số (từ params trong config YAML):
        pruning             -- bật/tắt alpha-beta pruning (A1)
        move_ordering       -- bật/tắt sắp xếp nước (A2)
        tt                  -- bật/tắt bảng chuyển vị Zobrist (A2)
        tt_size_mb          -- kích thước TT tính bằng MB (mặc định 64)
        iterative_deepening -- bật/tắt iterative deepening
        forced_moves        -- bật/tắt nước bắt buộc (thắng ngay / chặn thắng)
        radius              -- bán kính candidates() (mặc định 2)
        eval_weights        -- 'default' hoặc 'no_defense' (A7)
    """

    def __init__(
        self,
        name: str | None = None,
        limits: Limits | None = None,
        pruning: bool = True,
        move_ordering: bool = True,
        tt: bool = True,
        tt_size_mb: int = 64,
        iterative_deepening: bool = True,
        forced_moves: bool = True,
        radius: int = 2,
        eval_weights: str | dict = "default",
    ) -> None:
        super().__init__(name=name, limits=limits)
        self.pruning = pruning
        self.move_ordering = move_ordering
        self.use_tt = tt
        self.iterative_deepening = iterative_deepening
        self.forced_moves = forced_moves
        self.radius = radius

        if eval_weights == "no_defense":
            self._weight = WEIGHT
            self._no_defense = True
        elif isinstance(eval_weights, dict):
            base = list(WEIGHT)
            for k, v in eval_weights.items():
                base[int(k)] = v
            self._weight = tuple(base)
            self._no_defense = False
        else:
            self._weight = WEIGHT
            self._no_defense = False

        self._tt = TranspositionTable(tt_size_mb)
        self._killer: list[list[int]] = [[-1, -1] for _ in range(64)]
        self._history: dict[int, int] = {}
        self._stats: dict = {}
        self._nodes: int = 0

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self._tt.clear()
        self._killer = [[-1, -1] for _ in range(64)]
        self._history.clear()

    # ------------------------------------------------------------------
    # Interface bắt buộc
    # ------------------------------------------------------------------

    def select_action(self, game: CaroGame, limits: Limits) -> int:
        me, opp = game.me_opp()
        key = game.key
        side = game.side          # 0=X, 1=O — dùng để tính Zobrist đúng
        deadline = limits.deadline(margin_ms=10.0)

        self._nodes = 0
        self._killer = [[-1, -1] for _ in range(64)]
        self._history.clear()
        t0 = time.perf_counter()

        cand_bb = bb_candidates(me, opp, self.radius)
        moves = list(iter_bits(cand_bb))
        if not moves:
            return -1
        if len(moves) == 1:
            return moves[0]

        best_move = moves[0]
        best_score = -INF
        self._pv: list[int] = [best_move]
        depth_reached = 1

        max_depth = limits.max_depth if limits.max_depth is not None else 6

        if self.iterative_deepening:
            for depth in range(1, max_depth + 1):
                if deadline.expired():
                    break
                score, pv = self._negamax_root(
                    me, opp, key, side, moves, depth, deadline
                )
                if deadline.expired() and depth > 1:
                    break
                best_score = score
                best_move = pv[0] if pv else best_move
                self._pv = pv
                depth_reached = depth
                if best_move in moves:
                    moves.remove(best_move)
                    moves.insert(0, best_move)
                if best_score >= WIN_SCORE - 100:
                    break
        else:
            best_score, pv = self._negamax_root(
                me, opp, key, side, moves, max_depth, deadline
            )
            best_move = pv[0] if pv else best_move
            self._pv = pv
            depth_reached = max_depth

        elapsed = (time.perf_counter() - t0) * 1000
        self._stats = {
            "nodes":   self._nodes,
            "depth":   depth_reached,
            "time_ms": round(elapsed, 1),
            "pv":      self._pv,
            "score":   best_score,
        }
        return best_move

    # ------------------------------------------------------------------
    # Tìm kiếm gốc
    # ------------------------------------------------------------------

    def _negamax_root(
        self, me: int, opp: int, key: int, side: int,
        moves: list[int], depth: int, deadline: Any,
    ) -> tuple[int, list[int]]:
        alpha, beta = -INF, INF
        best_score = -INF
        moves[0]
        best_pv: list[int] = []

        ordered = self._order_moves(moves, me, opp, key, 0)

        for m in ordered:
            new_opp  = me | (1 << m)
            # Cập nhật key đúng side: me đang là side
            new_key  = key ^ ZOBRIST[side][m] ^ SIDE_KEY
            new_side = side ^ 1

            score, pv = self._negamax(
                opp, new_opp, new_key, new_side,
                depth - 1, -beta, -alpha, 1, deadline
            )
            score = -score

            if score > best_score:
                best_score = score
                best_pv = [m] + pv
            alpha = max(alpha, score)
            if self.pruning and alpha >= beta:
                break
            if deadline.expired():
                break

        return best_score, best_pv

    # ------------------------------------------------------------------
    # Negamax đệ quy — me là bên đến lượt, opp là bên vừa đi
    # ------------------------------------------------------------------

    def _negamax(
        self, me: int, opp: int, key: int, side: int,
        depth: int, alpha: int, beta: int, ply: int, deadline: Any,
    ) -> tuple[int, list[int]]:
        self._nodes += 1

        # opp vừa đi — kiểm tra opp thắng
        if has_five(opp):
            return -(WIN_SCORE - ply), []

        # Tra TT
        if self.use_tt:
            entry = self._tt.get(key)
            if entry is not None:
                tt_val, tt_depth, tt_flag, tt_move = entry
                if tt_depth >= depth:
                    if tt_flag == EXACT:
                        return tt_val, [tt_move] if tt_move >= 0 else []
                    elif tt_flag == LOWER:
                        alpha = max(alpha, tt_val)
                    elif tt_flag == UPPER:
                        beta = min(beta, tt_val)
                    if self.pruning and alpha >= beta:
                        return tt_val, [tt_move] if tt_move >= 0 else []

        # Node lá
        if depth == 0 or deadline.expired():
            return self._eval_leaf(me, opp), []

        # Tập ứng viên
        cand_bb = bb_candidates(me, opp, self.radius)
        moves = list(iter_bits(cand_bb))
        if not moves:
            return self._eval_leaf(me, opp), []

        # Nước bắt buộc
        if self.forced_moves:
            forced = self._forced(me, opp, moves)
            if forced:
                moves = forced

        ordered = self._order_moves(moves, me, opp, key, ply)

        orig_alpha = alpha
        best_score = -INF
        best_pv: list[int] = []
        best_m = ordered[0]

        for m in ordered:
            new_opp  = me | (1 << m)
            new_key  = key ^ ZOBRIST[side][m] ^ SIDE_KEY
            new_side = side ^ 1

            score, pv = self._negamax(
                opp, new_opp, new_key, new_side,
                depth - 1, -beta, -alpha, ply + 1, deadline
            )
            score = -score

            if score > best_score:
                best_score = score
                best_m = m
                best_pv = [m] + pv
            alpha = max(alpha, score)
            if self.pruning and alpha >= beta:
                self._update_killer(m, ply)
                self._history[m] = self._history.get(m, 0) + depth * depth
                break
            if deadline.expired():
                break

        # Ghi TT
        if self.use_tt and not deadline.expired():
            flag = EXACT if orig_alpha < best_score < beta else \
                   LOWER if best_score >= beta else UPPER
            self._tt.put(key, best_score, depth, flag, best_m)

        return best_score, best_pv

    # ------------------------------------------------------------------
    # Nước bắt buộc
    # ------------------------------------------------------------------

    def _forced(self, me: int, opp: int, moves: list[int]) -> list[int]:
        win_sqs = winning_squares(me, opp)
        wins = [m for m in moves if (1 << m) & win_sqs]
        if wins:
            return wins
        opp_win = winning_squares(opp, me)
        blocks = [m for m in moves if (1 << m) & opp_win]
        if blocks:
            return blocks
        return []

    # ------------------------------------------------------------------
    # Sắp xếp nước
    # ------------------------------------------------------------------

    def _order_moves(
        self, moves: list[int], me: int, opp: int, key: int, ply: int,
    ) -> list[int]:
        if not self.move_ordering:
            return moves

        tt_move = -1
        if self.use_tt:
            entry = self._tt.get(key)
            if entry is not None:
                tt_move = entry[3]

        scores: list[tuple[int, int]] = []
        for m in moves:
            bit_m = 1 << m
            fm = me | bit_m
            if m == tt_move:
                s = 10_000_000
            elif has_five(fm):
                s = 9_500_000
            elif _double_attack(fm, opp):
                s = 9_000_000
            elif _makes_four(fm, opp):
                s = 8_000_000
            elif _double_attack(opp | bit_m, me):
                s = 7_000_000
            elif ply < len(self._killer) and m == self._killer[ply][0]:
                s = 6_000_000
            elif ply < len(self._killer) and m == self._killer[ply][1]:
                s = 5_000_000
            elif m in self._history:
                s = 1_000_000 + self._history[m]
            else:
                s = side_score(fm, opp, self._weight)
            scores.append((s, m))

        scores.sort(reverse=True)
        return [m for _, m in scores]

    # ------------------------------------------------------------------
    # Đánh giá lá
    # ------------------------------------------------------------------

    def _eval_leaf(self, me: int, opp: int) -> int:
        if self._no_defense:
            return side_score(me, opp, self._weight)
        return evaluate(me, opp, self._weight)

    # ------------------------------------------------------------------
    # Killer moves
    # ------------------------------------------------------------------

    def _update_killer(self, move: int, ply: int) -> None:
        if ply >= len(self._killer):
            return
        if move != self._killer[ply][0]:
            self._killer[ply][1] = self._killer[ply][0]
            self._killer[ply][0] = move

    # ------------------------------------------------------------------
    # Sinh dữ liệu expert (top-k nước gốc kèm điểm)
    # ------------------------------------------------------------------

    def scored_moves(
        self, game: CaroGame, limits: Limits, top_k: int = 5,
    ) -> list[tuple[int, int]]:
        """Trả [(move, score), ...] top-k nước tốt nhất theo góc nhìn bên đang đi.

        Dùng để sinh dữ liệu expert cho cấp 5 (mục 05.1–05.2).
        """
        me, opp = game.me_opp()
        key = game.key
        side = game.side
        deadline = limits.deadline(margin_ms=10.0)

        cand_bb = bb_candidates(me, opp, self.radius)
        moves = list(iter_bits(cand_bb))
        if not moves:
            return []

        self._nodes = 0
        max_depth = limits.max_depth if limits.max_depth is not None else 4
        results: list[tuple[int, int]] = []

        for m in moves:
            new_opp = me | (1 << m)
            new_key = key ^ ZOBRIST[side][m] ^ SIDE_KEY
            new_side = side ^ 1
            score, _ = self._negamax(
                opp, new_opp, new_key, new_side,
                max_depth - 1, -INF, INF, 1, deadline
            )
            results.append((m, -score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def last_stats(self) -> dict:
        return dict(self._stats)