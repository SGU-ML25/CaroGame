def bradley_terry(matches, base_rating=1000.0, prior_draws=1.0, max_iter=100, tol=1e-9):
    """Elo ước lượng đồng thời theo mô hình Bradley-Terry (không phụ thuộc thứ tự ván).

    `matches`: dãy (agent_x, agent_o, result) với result  in  {1, 0, -1} theo góc nhìn X; hoà tính nửa điểm.
    Mỗi cặp đã gặp nhau được cộng `prior_draws` ván hoà ảo, nên tác tử thắng hoặc thua tuyệt đối vẫn có
    điểm hữu hạn; điểm của các tác tử đó phụ thuộc mạnh vào số ván ảo. Trung bình các điểm bằng base_rating.
    """
    matches = list(matches)
    agents = sorted({a for m in matches for a in m[:2]})
    if not agents:
        return {}
    index = {a: i for i, a in enumerate(agents)}
    n = len(agents)
    games = np.zeros((n, n))
    score = np.zeros((n, n))
    for agent_x, agent_o, result in matches:
        i, j = index[agent_x], index[agent_o]
        if i == j:
            continue
        s = (result + 1) / 2
        games[i, j] += 1
        games[j, i] += 1
        score[i, j] += s
        score[j, i] += 1 - s
    met = games > 0
    games += prior_draws * met
    score += 0.5 * prior_draws * met

    def log_likelihood(r):
        diff = r[:, None] - r[None, :]
        return float(-(score * np.logaddexp(0.0, -diff)).sum())

    # Newton trên thang log-odds. Số hạng 1/n cố định phương tịnh tiến chung (chỉ hiệu điểm có nghĩa).
    r = np.zeros(n)
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(r[None, :] - r[:, None]))
        grad = (score - games * p).sum(axis=1)
        if np.abs(grad).max() < tol:
            break
        w = games * p * (1 - p)
        hessian = np.diag(w.sum(axis=1)) - w + 1.0 / n
        step = np.linalg.solve(hessian + 1e-12 * np.eye(n), grad)
        scale = 1.0
        while scale > 1e-6 and log_likelihood(r + scale * step) < log_likelihood(r):
            scale /= 2
        r = r + scale * step
        r -= r.mean()
    return {a: float(base_rating + 400 / math.log(10) * r[index[a]]) for a in agents}
