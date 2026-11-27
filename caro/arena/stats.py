
import math
from collections import defaultdict

import numpy as np
import pandas as pd


def wilson(k, n, z=1.96):
    """Khoảng tin cậy Wilson cho tỷ lệ nhị phân."""
    if n <= 0:
        return 0.0, 0.0

    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z / den * math.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n)
    )
    return center - half, center + half


def bootstrap(
    match_pairs,
    n_boot=10000,
    confidence=0.95,
    seed=42,
):
    """Bootstrap theo cặp trận đổi màu, trả về CI cho điểm trung bình."""

    if not isinstance(n_boot, int) or n_boot <= 0:
        raise ValueError("n_boot phải là số nguyên dương.")

    if not 0 < confidence < 1:
        raise ValueError("confidence phải nằm trong khoảng (0, 1).")

    if not match_pairs:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    pairs = np.asarray(match_pairs, dtype=float)

    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Mỗi cặp phải chứa đúng hai điểm trận đấu.")

    n_pairs = len(pairs)
    scores = np.empty(n_boot)

    for i in range(n_boot):
        indices = rng.integers(0, n_pairs, size=n_pairs)
        scores[i] = pairs[indices].mean()

    alpha = 1 - confidence
    low, high = np.quantile(
        scores,
        [alpha / 2, 1 - alpha / 2],
    )
    return float(low), float(high)


def elo(matches, base_rating=1000, k_factor=32):
    """Tính Elo theo thứ tự game_id nếu thông tin này có sẵn."""

    matches = list(matches)

    if matches and len(matches[0]) == 4:
        matches.sort(key=lambda item: item[0])
        matches = [
            (agent_x, agent_o, result)
            for _, agent_x, agent_o, result in matches
        ]

    ratings = defaultdict(lambda: base_rating)

    for agent_x, agent_o, result in matches:
        rx = ratings[agent_x]
        ro = ratings[agent_o]

        ex = 1 / (1 + 10 ** ((ro - rx) / 400))
        eo = 1 / (1 + 10 ** ((rx - ro) / 400))

        sx = 1.0 if result == 1 else (
            0.5 if result == 0 else 0.0
        )
        so = 1.0 - sx

        ratings[agent_x] += k_factor * (sx - ex)
        ratings[agent_o] += k_factor * (so - eo)

    return dict(ratings)


def bradley_terry(matches, base_rating=1000.0, prior_draws=1.0, max_iter=100, tol=1e-9):
    """Elo ước lượng đồng thời theo mô hình Bradley-Terry (không phụ thuộc thứ tự ván).

    `matches`: dãy (agent_x, agent_o, result) với result ∈ {1, 0, -1} theo góc nhìn X; hoà tính nửa điểm.
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


def detailed_record_by_turn(games_df):
    """Thống kê thắng, hòa, thua theo bên X và bên O."""

    records = {}
    agents = sorted(
        pd.unique(games_df[["agent_x", "agent_o"]].values.ravel())
    )

    for agent in agents:
        as_x = games_df[games_df["agent_x"] == agent]
        as_o = games_df[games_df["agent_o"] == agent]

        wins_x = int((as_x["result"] == 1).sum())
        draws_x = int((as_x["result"] == 0).sum())
        losses_x = int((as_x["result"] == -1).sum())

        wins_o = int((as_o["result"] == -1).sum())
        draws_o = int((as_o["result"] == 0).sum())
        losses_o = int((as_o["result"] == 1).sum())

        total_games = len(as_x) + len(as_o)
        total_wins = wins_x + wins_o
        total_draws = draws_x + draws_o
        total_losses = losses_x + losses_o

        score = (
            (total_wins + 0.5 * total_draws) / total_games
            if total_games else 0.0
        )

        records[agent] = {
            "first_hand_X": {
                "wins": wins_x,
                "draws": draws_x,
                "losses": losses_x,
            },
            "second_hand_O": {
                "wins": wins_o,
                "draws": draws_o,
                "losses": losses_o,
            },
            "total": {
                "wins": total_wins,
                "draws": total_draws,
                "losses": total_losses,
                "score": score,
            },
        }

    return records


def summarize_self_play(
    games_df,
    n_boot=1000,
    confidence=0.95,
    seed=42,
):
    """Thống kê lợi thế đi trước và khoảng tin cậy bootstrap."""

    self_play_df = games_df[
        games_df["agent_x"] == games_df["agent_o"]
    ]

    records = {}
    agents = sorted(self_play_df["agent_x"].unique())

    for agent in agents:
        agent_games = self_play_df[
            self_play_df["agent_x"] == agent
        ]

        total = len(agent_games)

        x_wins = int((agent_games["result"] == 1).sum())
        x_draws = int((agent_games["result"] == 0).sum())
        x_losses = int((agent_games["result"] == -1).sum())

        o_wins = x_losses
        o_draws = x_draws
        o_losses = x_wins

        score_x = (
            (x_wins + 0.5 * x_draws) / total
            if total else 0.0
        )
        score_o = (
            (o_wins + 0.5 * o_draws) / total
            if total else 0.0
        )

        if total and "opening_id" not in agent_games.columns:
            raise ValueError(
                "Dữ liệu self-play phải có opening_id "
                "để bootstrap theo cặp opening."
            )

        opening_pairs = []
        first_results = []

        if total:
            if "game_id" not in agent_games.columns:
                raise ValueError(
                    "Dữ liệu self-play phải có game_id "
                    "để xác định thứ tự hai ván."
                )

            sizes = set(agent_games.groupby("opening_id").size())
            if sizes not in ({1}, {2}):
                raise ValueError(
                    "Self-play cần mọi opening có đúng 1 ván, hoặc mọi opening "
                    "có đúng 2 ván, để bootstrap theo opening."
                )

            for _, opening_games in agent_games.groupby(
                "opening_id",
                sort=True,
            ):

                opening_games = opening_games.sort_values("game_id")

                opening_plies = [
                    int(value)
                    for value in opening_games.get(
                        "opening_plies",
                        pd.Series(0, index=opening_games.index),
                    )
                ]

                if len(set(opening_plies)) != 1:
                    raise ValueError(
                        "Hai ván cùng opening phải có số nước "
                        "opening giống nhau."
                    )

                first_side = opening_plies[0] % 2

                results = opening_games["result"].tolist()

                adjusted_results = [
                    float(result) if first_side == 0
                    else -float(result)
                    for result in results
                ]

                first_results.extend(adjusted_results)
                # Một ván mỗi opening (tác tử tất định không sinh hai ván trùng nhau): lặp lại
                # giá trị để đơn vị bootstrap vẫn là opening, trung bình cụm không đổi.
                opening_pairs.append(
                    (adjusted_results[0], adjusted_results[-1])
                )

        first_wins = sum(result == 1 for result in first_results)
        first_draws = sum(result == 0 for result in first_results)
        first_losses = sum(result == -1 for result in first_results)

        score_first = (
            (first_wins + 0.5 * first_draws) / total
            if total else 0.0
        )
        score_second = 1.0 - score_first if total else 0.0

        ci_low, ci_high = bootstrap(
            opening_pairs,
            n_boot=n_boot,
            confidence=confidence,
            seed=seed,
        )

        records[agent] = {
            "games": total,
            "as_X": {
                "wins": x_wins,
                "draws": x_draws,
                "losses": x_losses,
                "score": score_x,
            },
            "as_O": {
                "wins": o_wins,
                "draws": o_draws,
                "losses": o_losses,
                "score": score_o,
            },
            "as_first": {
                "wins": first_wins,
                "draws": first_draws,
                "losses": first_losses,
                "score": score_first,
            },
            "as_second": {
                "score": score_second,
            },
            "first_move_advantage": score_first - score_second,
            "first_move_advantage_ci": {
                "low": ci_low,
                "high": ci_high,
                "confidence": confidence,
                "n_boot": n_boot,
                "opening_pairs": len(opening_pairs),
            },
        }

    return records


def calculate_performance(games_df):
    """Thống kê hiệu năng theo đúng tác tử thực hiện từng lượt."""

    stats = {}
    agents = sorted(
        pd.unique(games_df[["agent_x", "agent_o"]].values.ravel())
    )

    collected = {
        agent: {"times": [], "nodes": [], "sims": []}
        for agent in agents
    }

    for _, row in games_df.iterrows():
        times = row.get("time_ms", []) or []
        nodes = row.get("nodes", []) or []
        sims = row.get("sims", []) or []

        opening_plies = int(row.get("opening_plies", 0))
        first_side = opening_plies % 2

        for index, elapsed in enumerate(times):
            side = (first_side + index) % 2
            agent = row["agent_x"] if side == 0 else row["agent_o"]

            collected[agent]["times"].append(elapsed)

            node_value = nodes[index] if index < len(nodes) else 0
            sim_value = sims[index] if index < len(sims) else 0

            collected[agent]["nodes"].append(node_value)
            collected[agent]["sims"].append(sim_value)

    for agent, values in collected.items():
        times = values["times"]
        nodes = values["nodes"]
        sims = values["sims"]

        stats[agent] = {
            "time_mean_ms": float(np.mean(times)) if times else 0.0,
            "time_p95_ms": float(np.percentile(times, 95)) if times else 0.0,
            "nodes_mean": float(np.mean(nodes)) if nodes else 0.0,
            "sims_mean": float(np.mean(sims)) if sims else 0.0,
        }

    return stats


def generate_matrix(games_df, out_dir):
    """Xuất ma trận điểm đối đầu và khoảng tin cậy Wilson cho tỷ lệ thắng."""

    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agents = sorted(
        pd.unique(games_df[["agent_x", "agent_o"]].values.ravel())
    )
    matrix = pd.DataFrame(index=agents, columns=agents)

    for a1 in agents:
        for a2 in agents:
            if a1 == a2:
                matrix.loc[a1, a2] = "—"
                continue

            m_x = games_df[
                (games_df["agent_x"] == a1)
                & (games_df["agent_o"] == a2)
            ]
            m_o = games_df[
                (games_df["agent_o"] == a1)
                & (games_df["agent_x"] == a2)
            ]

            wins = int((m_x["result"] == 1).sum())
            wins += int((m_o["result"] == -1).sum())

            draws = int((m_x["result"] == 0).sum())
            draws += int((m_o["result"] == 0).sum())

            losses = int((m_x["result"] == -1).sum())
            losses += int((m_o["result"] == 1).sum())

            total = wins + draws + losses

            if total == 0:
                matrix.loc[a1, a2] = "—"
                continue

            score = (wins + 0.5 * draws) / total

            # Wilson chỉ áp dụng cho tỷ lệ thắng nhị phân.
            # Trận hòa được báo cáo riêng, không tính nửa thành công.
            low, high = wilson(wins, total)

            matrix.loc[a1, a2] = (
                f"{score:.3f} "
                f"(W {low:.3f}-{high:.3f}; D {draws / total:.3f})"
            )

    matrix.to_csv(out_dir / "matrix.csv")

    with (out_dir / "matrix.md").open("w", encoding="utf-8") as file:
        file.write(matrix.to_markdown())

    return matrix
