"""Interface chung của mọi tác tử cấp 1–8 (mục 03.2).

Arena, training và UI chỉ dùng tác tử qua `Agent`; tác tử chỉ dùng engine (và nn cho cấp 5–8).
Ngân sách mỗi nước được truyền vào dưới dạng `Limits`, dựng từ khoá `limits` của config đã ghép.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, fields
from typing import Any, Protocol, runtime_checkable

from caro.engine import CaroGame

BUDGET_MODES = ("time", "fixed")

# Khoá chuẩn của last_stats() — tác tử chỉ ghi khoá nó có, arena/UI đọc theo đúng tên này.
STATS_KEYS = (
    "nodes",      # Alpha-Beta: số nút đã duyệt
    "sims",       # MCTS / PUCT: số mô phỏng
    "depth",      # độ sâu tìm trọn vẹn gần nhất
    "time_ms",    # thời gian tác tử tự đo
    "pv",         # biến chính: list nước đi
    "score",      # giá trị gốc theo góc nhìn bên đi
    "reason",     # cấp 2: luật đã dùng (P1…P7)
    "visits",     # MCTS / PUCT: {ô: số lượt thăm} — heatmap UI
    "policy",     # cấp 5–8: {ô: xác suất} — heatmap UI
    "value",      # cấp 5–8: v(s) theo góc nhìn bên đi
)


@dataclass(frozen=True)
class Limits:
    """Ngân sách một nước. mode='time' dùng time_ms; mode='fixed' dùng max_depth / max_sims.

    Giá trị None = không giới hạn theo tiêu chí đó (tác tử tự chọn mặc định của mình).
    """

    mode: str = "fixed"
    time_ms: int | None = None
    max_depth: int | None = None
    max_sims: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in BUDGET_MODES:
            raise ValueError(f"budget.mode phải là {BUDGET_MODES}, nhận {self.mode!r}")
        for f in fields(self)[1:]:
            v = getattr(self, f.name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v <= 0):
                raise ValueError(f"budget.{f.name} phải là số nguyên dương hoặc null, nhận {v!r}")
        if self.mode == "time" and self.time_ms is None:
            raise ValueError("budget.mode=time cần time_ms")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Limits:
        data = data or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"khoá budget không hợp lệ: {sorted(unknown)}")
        return cls(**data)

    def deadline(self, margin_ms: float = 0.0) -> Deadline:
        """Đồng hồ cho nước hiện tại; chỉ hết giờ khi có time_ms."""
        return Deadline(self.time_ms, margin_ms)


class Deadline:
    """Đo giờ một nước. Gọi ngay khi bắt đầu select_action, kiểm tra expired() trong vòng lặp tìm kiếm."""

    __slots__ = ("end", "start")

    def __init__(self, time_ms: float | None, margin_ms: float = 0.0) -> None:
        self.start = time.perf_counter()
        self.end = None if time_ms is None else self.start + max(0.0, time_ms - margin_ms) / 1000

    def expired(self) -> bool:
        return self.end is not None and time.perf_counter() >= self.end

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000

    def remaining_ms(self) -> float | None:
        return None if self.end is None else max(0.0, (self.end - time.perf_counter()) * 1000)


@runtime_checkable
class Agent(Protocol):
    name: str
    limits: Limits      # ngân sách mặc định lấy từ config; arena / UI có thể truyền Limits khác

    def reset(self, seed: int) -> None: ...
    def select_action(self, game: CaroGame, limits: Limits) -> int: ...
    def last_stats(self) -> dict: ...


class BaseAgent:
    """Phần chung tiện dùng: tên, ngân sách mặc định, RNG có seed, thống kê nước gần nhất.

    Lớp con nhận tham số qua __init__ (từ `params` của config) và cài select_action.
    """

    def __init__(self, name: str | None = None, limits: Limits | None = None) -> None:
        self.name = name or type(self).__name__
        self.limits = limits or Limits()
        self.rng = random.Random(0)
        self._stats: dict = {}

    def reset(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self._stats = {}

    def select_action(self, game: CaroGame, limits: Limits) -> int:
        raise NotImplementedError

    def last_stats(self) -> dict:
        return dict(self._stats)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"
