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
