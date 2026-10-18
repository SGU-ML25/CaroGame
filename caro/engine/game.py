"""CaroGame — API thống nhất cho mọi tác tử, arena và UI (mục 02.3).

Nước đi là chỉ số ô 0..120 (i = r·11 + c). Quy ước góc nhìn:
- result() luôn theo X: +1 X thắng, −1 O thắng, 0 hoà, None nếu chưa kết thúc.
- step() trả reward theo góc nhìn người VỪA đi.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable

from caro.engine.bitboard import CELLS, FULL, N, has_five, iter_bits
from caro.engine.zobrist import SIDE_KEY, ZOBRIST

X, O = 0, 1
SYMBOLS = ("X", "O")


class IllegalMoveError(ValueError):
    pass


class CaroGame:
    __slots__ = ("bb", "history", "key", "side", "winner")

    def __init__(self) -> None:
        self.reset()

    @classmethod
    def from_moves(cls, moves: Iterable[int]) -> CaroGame:
        g = cls()
        for a in moves:
            g.step(a)
        return g

    def reset(self) -> CaroGame:
        self.bb = [0, 0]             # bb[X], bb[O]
        self.side = X                # bên đến lượt
        self.history: list[int] = []
        self.key = 0
        self.winner: int | None = None
        return self

    # ---- truy vấn -------------------------------------------------------
    @property
    def occupied(self) -> int:
        return self.bb[X] | self.bb[O]

    @property
    def empty(self) -> int:
        return ~(self.bb[X] | self.bb[O]) & FULL

    @property
    def last_move(self) -> int | None:
        return self.history[-1] if self.history else None

    def me_opp(self) -> tuple[int, int]:
        """(bitboard bên đến lượt, bitboard đối thủ)."""
        return self.bb[self.side], self.bb[self.side ^ 1]

    def is_terminal(self) -> bool:
        return self.winner is not None or (self.bb[X] | self.bb[O]) == FULL

    def legal_mask(self) -> int:
        return 0 if self.is_terminal() else self.empty

    def legal_actions(self) -> list[int]:
        return list(iter_bits(self.legal_mask()))

    def result(self) -> int | None:
        if self.winner is not None:
            return 1 if self.winner == X else -1
        return 0 if self.is_terminal() else None

    def termination(self) -> str | None:
        """'five' | 'full' | None — trường termination trong log ván."""
        if self.winner is not None:
            return "five"
        return "full" if self.is_terminal() else None

    # ---- thay đổi trạng thái -------------------------------------------
    def step(self, action: int) -> tuple[float, bool, dict]:
        """Đi một nước. Trả (reward, done, info); reward theo góc nhìn người vừa đi.

        Nước bất hợp lệ → IllegalMoveError, trạng thái không đổi.
        """
        if self.is_terminal():
            raise IllegalMoveError("ván đã kết thúc")
        if isinstance(action, bool):
            raise IllegalMoveError(f"nước {action!r} không hợp lệ")
        try:
            a = operator.index(action)   # nhận cả int của numpy
        except TypeError:
            raise IllegalMoveError(f"nước {action!r} không phải số nguyên") from None
        if not 0 <= a < CELLS or (self.occupied >> a) & 1:
            raise IllegalMoveError(f"nước {a} không hợp lệ")

        mover = self.side
        self.bb[mover] |= 1 << a
        self.key ^= ZOBRIST[mover][a] ^ SIDE_KEY
        self.history.append(a)
        self.side ^= 1
        if has_five(self.bb[mover]):
            self.winner = mover
        done = self.is_terminal()
        reward = 1.0 if self.winner == mover else 0.0
        return reward, done, {"winner": self.winner}

    def undo(self) -> int:
        """Lùi một nước, trả lại nước vừa lùi."""
        if not self.history:
            raise IllegalMoveError("không còn nước để lùi")
        a = self.history.pop()
        self.side ^= 1
        self.key ^= ZOBRIST[self.side][a] ^ SIDE_KEY
        self.bb[self.side] ^= 1 << a
        self.winner = None           # thế trước đó chắc chắn chưa kết thúc
        return a

    def clone(self) -> CaroGame:
        g = CaroGame.__new__(CaroGame)
        g.bb = self.bb[:]
        g.side, g.key, g.winner = self.side, self.key, self.winner
        g.history = self.history[:]
        return g

    # ---- hiển thị -------------------------------------------------------
    def render(self) -> str:
        x, o = self.bb

        def sym(i: int) -> str:
            return "X" if (x >> i) & 1 else "O" if (o >> i) & 1 else "."

        return "\n".join(" ".join(sym(r * N + c) for c in range(N)) for r in range(N))

    def __repr__(self) -> str:
        state = self.termination() or f"{SYMBOLS[self.side]} đi"
        return f"CaroGame({len(self.history)} nước, {state})"
