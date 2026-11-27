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
