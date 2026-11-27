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
