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
    """Giá trị nút kết thúc theo góc nhìn bên đang đi: có người thắng thì đó là bên vừa đi (side đã đổi) -> -1; hoà -> 0."""
    return -1.0 if game.winner is not None else 0.0


def select_child(node: PUCTNode, c_puct: float) -> PUCTNode:
    sqrt_n = math.sqrt(max(1, node.N + node.vl))
    best, best_score = node.children[0], -math.inf
    for ch in node.children:
        n = ch.N + ch.vl
        # virtual loss = bên chọn (cha) thua -> bên đi ở con thắng: cộng +1 mỗi lượt vào W của con
        q = -(ch.W + ch.vl) / n if n else 0.0
        score = q + c_puct * ch.prior * sqrt_n / (1 + n)
        if score > best_score:
            best, best_score = ch, score
    return best
