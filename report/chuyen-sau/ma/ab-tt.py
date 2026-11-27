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
