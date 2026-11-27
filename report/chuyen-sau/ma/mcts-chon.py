def select_action(self, game: CaroGame, limits: Limits) -> int:
    """Chọn nước đi bằng MCTS. Trả chỉ số ô (0-120)."""
    if game.is_terminal():
        return -1

    t0 = time.perf_counter()
    restrict: list[int] | None = None
    if self.root_tactics:
        me, opp = game.me_opp()
        forced = first_winning_square(me, opp)
        if forced is None:
            forced = first_winning_square(opp, me)
        if forced is None:
            # Đi tạo >=2 ô thắng (tứ hở / đòn kép) thì đối thủ không chặn nổi -> đi luôn.
            # Ngược lại, nếu đối thủ có nước tạo >=2 ô thắng thì chỉ xét những ô đó (chặn ba hở).
            empty = ~(me | opp) & FULL
            threats = []
            for m in iter_bits(bb_candidates(me, opp, self.candidate_radius) or empty):
                if winning_squares(me | 1 << m, opp).bit_count() >= 2:
                    forced = m
                    break
                if winning_squares(opp | 1 << m, me).bit_count() >= 2:
                    threats.append(m)
            if forced is None and threats:
                restrict = threats
        if forced is not None:
            self._root = None
            self._stats = {"sims": 0, "time_ms": round((time.perf_counter() - t0) * 1000, 2),
                           "visits": {}, "pv": [forced]}
            return forced
    deadline = limits.deadline(margin_ms=5.0)
    mode = limits.mode
    max_sims = limits.max_sims if limits.max_sims is not None else self.limits.max_sims
    if max_sims is None and limits.time_ms is None:
        # Không có tiêu chí dừng nào (vd. Limits() rỗng) - dùng mặc định thay vì lặp vô hạn.
        max_sims = DEFAULT_MAX_SIMS

    root = self._locate_root(game)
    if root is None:
        root = MCTSNode(
            parent=None,
            move=None,
            mover=game.side ^ 1,
            untried=candidate_moves(game, self.candidate_radius),
            key=game.key,
        )
        if restrict is not None:
            root.untried = restrict

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
