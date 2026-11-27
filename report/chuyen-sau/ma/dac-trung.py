def bits(x: int) -> np.ndarray:
    """121 bit của bitboard -> mảng uint8 độ dài CELLS (bit i = ô i)."""
    return np.unpackbits(np.frombuffer(x.to_bytes(16, "little"), dtype=np.uint8), bitorder="little")[:CELLS]


def encode(game: CaroGame) -> np.ndarray:
    """(4, N, N): quân mình, quân đối thủ, 1 nếu X đi, 0 - giống board_to_tensor."""
    me, opp = game.me_opp()
    x = np.zeros((4, CELLS), dtype=np.float32)
    x[0] = bits(me)
    x[1] = bits(opp)
    if game.side == X:
        x[2] = 1.0
    return x.reshape(4, N, N)


def legal_mask(game: CaroGame) -> np.ndarray:
    return bits(game.empty).astype(bool)
