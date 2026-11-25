"""Sinh bảng, hình và hồ sơ LaTeX cho bao-cao-caro.tex từ log đã lưu."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/ban-mo-rong"
RUNS = ROOT / "runs"
FIG = OUT / "hinh"
sys.path.insert(0, str(ROOT))
from caro.engine import CaroGame, iter_bits, winning_squares


def read(path):
    return json.loads((RUNS / path).read_text(encoding="utf-8"))


def rows(path):
    with (RUNS / path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_csv(path):
    with (RUNS / path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def emit(name, text):
    (OUT / f"{name}.tex").write_text(text, encoding="utf-8")


SPECIAL = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "&": r"\&", "%": r"\%",
    "$": r"\$", "#": r"\#", "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def tex(text):
    return "".join(SPECIAL.get(char, char) for char in str(text))


def dec(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value, digits=1):
    return dec(value * 100, digits) + "%"


def table(headers, data, layout=None):
    spec = layout or ("l" + "r" * (len(headers) - 1))
    lines = [r"\begin{center}\small", rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule"]
    lines.append(" & ".join(map(tex, headers)) + r" \\")
    lines.append(r"\midrule")
    lines.extend(" & ".join(map(tex, row)) + r" \\" for row in data)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}"])
    return "\n".join(lines)


def new_page(title):
    return rf"\HoSoTrang{{{tex(title)}}}"


def heading(title):
    return rf"\paragraph{{{tex(title)}}}"


def coord(move):
    return f"{chr(65 + move % 11)}{move // 11 + 1}"


def wdl(results):
    c = Counter(results)
    return c[1], c[0], c[-1]


def format_wdl(results):
    return "/".join(map(str, wdl(results)))


def result_for(game, agent):
    assert agent in (game["agent_x"], game["agent_o"])
    return game["result"] if game["agent_x"] == agent else -game["result"]


def score(results):
    return (1 + np.mean(results)) / 2


def opening_of(game):
    return game["moves"][:len(game["moves"]) - len(game["time_ms"])]


def clocks(game, agent):
    first_x = len(opening_of(game)) % 2 == 0
    return game["time_ms"][int(first_x != (game["agent_x"] == agent))::2]


def bootstrap(games, agent):
    paired = defaultdict(list)
    for game in games:
        paired[game["opening_id"]].append(result_for(game, agent))
    assert all(len(pair) == 2 for pair in paired.values())
    values = np.array([score(pair) for pair in paired.values()])
    rng = np.random.default_rng(42)
    lo, hi = np.quantile(rng.choice(values, (20_000, len(values))).mean(axis=1), [0.025, 0.975])
    return f"{pct(lo)}–{pct(hi)}"


def overview_data():
    sources = {
        "A1": "bench/a1_ab_pruning/20261029-200321/summary.json",
        "A2": "bench/a2_ordering_tt/20261029-212610/summary.json",
        "A7": "bench/a7_eval_weights/20261029-223250/summary.json",
    }
    data = []
    for tag, filename in sources.items():
        run = read(filename)
        data.extend((tag, agent, run["n_positions"], run[agent]["nodes_median"],
                     dec(run[agent]["time_ms_median"]), dec(run[agent]["time_ms_p95"]))
                    for agent in run["agents"] if isinstance(run[agent], dict))
    emit("so-lieu-benchmark", table(
        ("Nhóm", "Cấu hình", "Thế", "Nút (P50)", "ms (P50)", "ms (P95)"), data, "llrrrr"
    ))

    archive = read_csv("arena_matrix/ranking.csv")
    emit("so-lieu-arena", table(
        ("Tác tử", "Ván", "Thắng", "Hòa", "Thua", "Điểm"),
        [(r["tác tử"], r["ván"], r["thắng"], r["hoà"], r["thua"], pct(float(r["điểm %"]) / 100)) for r in archive],
    ))
    sl = []
    for label, suffix in [("SL v1", ""), ("SL v2", "_v2"), ("SL v3", "_v3")]:
        logs = read(f"train/sl_policy{suffix}/metrics.json")
        best = min(logs, key=lambda row: row["val_loss"])
        sl.append((label, len(logs), best["epoch"], dec(best["val_loss"], 4), dec(logs[-1]["val_loss"], 4)))
    evals = read_csv("train_eval/results.csv")
    emit("so-lieu-hoc", table(("Mạng", "Số epoch", "Epoch tốt nhất", "Val loss tốt nhất", "Val loss cuối"), sl)
         + "\n" + table(("Checkpoint", "Đối thủ", "Ván", "W/D/L", "Điểm"), [
             (r["checkpoint"], "L2" if "l2" in r["opponent"] else "L3", r["n"],
              f"{r['wins']}/{r['draws']}/{r['losses']}", pct(float(r["score"]))) for r in evals
         ], "llrrr"))

    a5 = read_csv("eval/a5_puct/results.csv")
    emit("so-lieu-a5", table(("Tác tử", "Đối thủ", "Ván", "W/D/L", "Điểm"), [
        (r["agent"], r["opponent"], r["n"], f"{r['wins']}/{r['draws']}/{r['losses']}", pct(float(r["score"]), 2))
        for r in a5
    ], "llrrr"))
    trend = [r for r in a5 if r["agent"].startswith("L7-PUCT[l5,") and r["opponent"] == "l3_alphabeta"]
    fig, ax = plt.subplots(figsize=(7.1, 3.3), layout="constrained")
    sims = [int(r["agent"].split(",")[1][:-1]) for r in trend]
    y = np.array([float(r["score"]) for r in trend])
    lo, hi = [np.array([float(r[key]) for r in trend]) for key in ("ci_low", "ci_high")]
    ax.errorbar(sims, y * 100, yerr=np.stack([y - lo, hi - y]) * 100, fmt="o-", color="#166d63", capsize=4)
    ax.axhline(50, color="0.5", linestyle="--")
    ax.set(xscale="log", xticks=sims, xticklabels=sims, xlabel="Mô phỏng mỗi nước", ylabel="Điểm gặp L3 (%)")
    save(fig, "a5")


def save(fig, filename):
    fig.savefig(FIG / f"{filename}.png", dpi=190, facecolor="white")
    plt.close(fig)


def opening_profiles():
    logs = {
        "A3": rows("tournament/a3_mcts_sims/20261114-232353/games.jsonl"),
        "A4": rows("tournament/a4_rollout/20261114-234556/games.jsonl"),
    }
    ids = sorted({r["opening_id"] for r in logs["A3"]})
    assert len(ids) == 100
    indexed = defaultdict(list)
    for exp, records in logs.items():
        for r in records:
            challenger = next(a for a in (r["agent_x"], r["agent_o"]) if a.startswith("MCTS"))
            rival = r["agent_o"] if r["agent_x"] == challenger else r["agent_x"]
            indexed[(r["opening_id"], exp, challenger, rival)].append(r)
    assert len(indexed) == 1_200
    result = [r"\chapter{Hồ sơ đối chiếu theo khai cuộc A3 và A4}",
              ("Mỗi hồ sơ ghép hai màu của từng cấu hình tại cùng một thế mở đầu. "
              "Ở cột X/O, W là thắng, D là hòa, L là thua từ góc nhìn MCTS. "
              "Độ dài tính cả khai cuộc; thời gian chỉ tính các nước của MCTS. "
              "Đối thủ L3 trong hồ sơ này dùng độ sâu 4.")]
    letters = {1: "W", 0: "D", -1: "L"}
    for opening in ids:
        group = {key[1:]: value for key, value in indexed.items() if key[0] == opening}
        common = {tuple(opening_of(r)) for pair in group.values() for r in pair}
        assert len(common) == 1
        moves = next(iter(common))
        data = []
        for (exp, agent, rival), pair in group.items():
            oriented = sorted(pair, key=lambda r: r["agent_x"] != agent)
            assert len(oriented) == 2 and oriented[0]["agent_x"] == oriented[1]["agent_o"] == agent
            outcomes = [result_for(r, agent) for r in oriented]
            times = [ms for r in pair for ms in clocks(r, agent)]
            data.append((exp, agent, rival.split("-")[0], *map(letters.get, outcomes), pct(score(outcomes), 0),
                         "/".join(str(len(r["moves"])) for r in oriented), dec(np.median(times))))
        order = {"MCTS-100": 0, "MCTS-400": 1, "MCTS-1600": 2, "MCTS-6400": 3, "MCTS-random": 4, "MCTS-light": 5}
        data.sort(key=lambda entry: (order[entry[1]], entry[2]))
        result += [new_page(f"Khai cuộc {opening:02d} · đối chiếu MCTS"),
                   tex(f"Mở đầu ({len(moves)} nước): " + " → ".join(map(coord, moves)) + ". Có 24 ván đối chiếu."),
                   table(("Mã", "MCTS", "Gặp", "X", "O", "Điểm", "Ply X/O", "ms P50"), data, "lllccrrr")]
        top = max(data, key=lambda row: int(row[5].rstrip("%")))
        max_score = top[5]
        won = sum(row[3:5].count("W") for row in data)
        avg_plies = np.mean([len(r["moves"]) for pair in group.values() for r in pair])
        result += [heading("Tóm lược khai cuộc"),
                   tex(f"Số ván MCTS thắng: {won}/24. Điểm cao nhất của một cặp: {max_score}. "
                       f"Độ dài trung bình trên toàn bộ ván: {dec(avg_plies)} nước."),
                   heading("Bàn sau chuỗi khai cuộc"), board(moves)]
    emit("ho-so-khai-cuoc", "\n".join(result))


def training_profiles(metrics, log):
    selfplay = sorted((r for r in log if r["tournament_id"] == "selfplay"), key=lambda r: r["game_id"])
    gating = defaultdict(list)
    for r in log:
        if r["tournament_id"].startswith("L8-incumbent-"):
            gating[int(r["tournament_id"].rsplit("-", 1)[1])].append(r)
    assert len(metrics) == 50 and len(selfplay) == 10_000
    section = [r"\chapter{Hồ sơ các vòng huấn luyện L8}",
               ("Mỗi mục ghi số liệu sinh mẫu, thống kê đấu kiểm tra và điểm của ứng viên theo màu. "
               "Khoảng bootstrap dùng 50 khai cuộc, mỗi khai cuộc giữ hai màu. "
               "Điểm qua các vòng có thể dùng đối thủ khác nhau.")]
    accepted = 0
    previous_time = 0
    for row in metrics:
        step = row["step"]
        games = selfplay[200 * (step - 1):200 * step]
        gate = gating[step]
        assert len(gate) == 100
        outcomes = [result_for(r, "L8") for r in gate]
        opponent_version = accepted
        accepted = step if row["accepted"] else accepted
        assert abs(score(outcomes) - row["gating"]["score"]) < 1e-6
        clocks_self = [ms for r in games for ms in r["time_ms"]]
        gate_times = [ms for r in gate for ms in r["time_ms"]]
        lengths = [len(r["moves"]) for r in games]
        elapsed = row["time"] - previous_time
        previous_time = row["time"]
        color_data = [
            (color, len(sub), format_wdl(sub), pct(score(sub)))
            for color, sub in ((c, [result_for(r, "L8") for r in gate if r[f"agent_{c.lower()}"] == "L8"])
                               for c in ("X", "O"))
        ]
        section += [
            new_page(f"Vòng {step:02d} · huấn luyện và quyết định gating"),
            tex(f"Bản kiểm tra: vòng {opponent_version}. Bản giữ cuối vòng: {accepted}. "
                f"Quyết định: {'duyệt' if row['accepted'] else 'giữ bản cũ'}."),
            table(("Thông số", "Giá trị"), [
                ("Tự đấu", f"{len(games)} ván; X thắng / hòa / O thắng: {format_wdl([r['result'] for r in games])}"),
                ("Độ dài P50 / P95", f"{dec(np.median(lengths), 0)} / {dec(np.quantile(lengths, .95))} nước"),
                ("Mẫu phát lại cuối vòng", row["replay_positions"]),
                ("Loss tổng", dec(row["loss"], 4)),
                ("Loss policy / value", f"{dec(row['policy_loss'], 4)} / {dec(row['value_loss'], 4)}"),
                ("Thời gian vòng", f"{dec(elapsed / 60, 2)} phút"),
                ("Đồng hồ nước tự đấu", f"{dec(sum(clocks_self) / 1000)} giây"),
                ("Đồng hồ nước kiểm tra", f"{dec(sum(gate_times) / 1000)} giây"),
                ("Tự đấu: ms/nước P50 / P95", f"{dec(np.median(clocks_self))} / {dec(np.quantile(clocks_self, .95))}"),
            ], "lp{8cm}"),
            heading("Ứng viên gặp bản đang giữ"),
            table(("Màu", "Ván", "W/D/L", "Điểm"), [*color_data, ("Tổng", len(gate), format_wdl(outcomes), pct(score(outcomes)))]),
            tex(f"Khoảng bootstrap ghép 95%: {bootstrap(gate, 'L8')}. Ngưỡng chấp nhận 55%. "
                "Khoảng mô tả lượt này và không hiệu chỉnh cho việc lặp nhiều vòng kiểm tra."),
        ]
        margins = defaultdict(list)
        for r in gate:
            margins[r["opening_id"]].append(result_for(r, "L8"))
        hist = Counter(sum(values) for values in margins.values())
        section += [
            heading("Phân bố điểm theo 50 khai cuộc đổi màu"),
            table(("Điểm của ứng viên trên 2 ván", "0", "0,5", "1", "1,5", "2"),
                  [("Số khai cuộc", *(hist[k] for k in range(-2, 3)))]),
        ]
    emit("ho-so-huan-luyen", "\n".join(section))


def board(moves):
    game = CaroGame.from_moves(moves)
    lines = ["   A B C D E F G H I J K"]
    for r in range(11):
        chars = []
        for c in range(11):
            pos = 11 * r + c
            char = next((symbol for side, symbol in enumerate("XO") if game.bb[side] >> pos & 1), ".")
            chars.append(char.lower() if pos == game.last_move else char)
        lines.append(f"{r + 1:2} " + " ".join(chars))
    return "\\begin{verbatim}\n" + "\n".join(lines) + "\n\\end{verbatim}"


def tactical_events(row):
    opening = opening_of(row)
    game = CaroGame.from_moves(opening)
    events = []
    for i, move in enumerate(row["moves"][len(opening):], start=len(opening)):
        me, opponent = game.me_opp()
        wins = set(iter_bits(winning_squares(me, opponent)))
        threats = set(iter_bits(winning_squares(opponent, me)))
        if move in wins:
            label, rank, choices = "Thắng ngay", 2, wins
        elif wins:
            label, rank, choices = "Bỏ lỡ thắng ngay", 5, wins
        elif len(threats) == 1 and move not in threats:
            label, rank, choices = "Bỏ chặn duy nhất", 4, threats
        elif len(threats) == 1:
            label, rank, choices = "Chặn bắt buộc", 3, threats
        elif len(threats) > 1:
            label, rank, choices = "Đã ở thế buộc thua ở lượt kế", 1, ()
        else:
            placed = me | (1 << move)
            if winning_squares(placed, opponent).bit_count() < 2 or winning_squares(opponent, placed):
                game.step(move)
                continue
            label, rank, choices = "Tạo hai ô thắng", 3, ()
        events.append((i + 1, "XO"[game.side], coord(move), label, ", ".join(map(coord, sorted(choices))) or "—", rank))
        game.step(move)
    assert game.result() == row["result"]
    return events


def replay_profiles(matchups):
    section = [r"\chapter{Hồ sơ ván đấu và so sánh hai màu}",
               ("Hồ sơ ván trình bày các nước chiến thuật và thế cờ tại sự kiện đáng chú ý. "
               "Chữ thường đánh dấu nước vừa đánh. Hồ sơ cặp đặt hai ván chung khai cuộc cạnh nhau. "
               "Nhận xét chiến thuật chỉ dùng luật thắng ngay và đe dọa nước kế tiếp.")]
    for label, matches in matchups.items():
        section += [rf"\section{{{tex(label)}: từng ván}}"]
        for row in sorted(matches, key=lambda r: r["game_id"]):
            events = tactical_events(row)
            selected = sorted(sorted(events, key=lambda ev: ev[-1], reverse=True)[:5])
            key = max(events, key=lambda ev: ev[-1]) if events else None
            outcome = {1: "X thắng", 0: "hòa", -1: "O thắng"}[row["result"]]
            before = len(opening_of(row))
            sequence = " ".join(f"{i + 1}.{coord(move)}" for i, move in enumerate(row["moves"]))
            clocks_x, clocks_o = clocks(row, row["agent_x"]), clocks(row, row["agent_o"])
            section += [
                new_page(f"{label} · ván {row['game_id']}"),
                tex(f"Khai cuộc {row['opening_id']} ({before} nước). X: {row['agent_x']}. "
                    f"O: {row['agent_o']}. Kết quả: {outcome}, {len(row['moves'])} nước."),
                heading("Thống kê từng bên"),
                table(("Màu", "Nước đo", "P50 (ms)", "P95 (ms)"), [
                    (color, len(times), dec(np.median(times)), dec(np.quantile(times, .95))) if times
                    else (color, 0, "—", "—") for color, times in (("X", clocks_x), ("O", clocks_o))
                ]),
            ]
            if key:
                section += [heading("Nước chiến thuật"),
                            table(("Ply", "Màu", "Đi", "Nhận xét", "Ô liên quan"), [r[:-1] for r in selected], "rccll"),
                            heading(f"Thế cờ sau nước {key[0]} ({key[2]})"), board(row["moves"][:key[0]])]
            else:
                section += [heading("Thế kết thúc"), board(row["moves"])]
            section += [heading("Diễn biến"), "{\\footnotesize " + tex(sequence) + "\\par}"]
        section += [rf"\section{{{tex(label)}: đối chiếu theo khai cuộc}}"]
        by_opening = defaultdict(list)
        for row in matches:
            by_opening[row["opening_id"]].append(row)
        comparison = []
        for opening, pair in sorted(by_opening.items()):
            pair.sort(key=lambda row: row["agent_x"] != "L8")
            moves = opening_of(pair[0])
            assert len(pair) == 2 and opening_of(pair[1]) == moves
            outcomes = [result_for(row, "L8") for row in pair]
            mistakes = [sum(ev[-1] >= 4 for ev in tactical_events(row)) for row in pair]
            labels = {1: "W", 0: "D", -1: "L"}
            comparison.append((opening, f"{pair[0]['game_id']}/{pair[1]['game_id']}",
                               " ".join(map(coord, moves)), "/".join(map(labels.get, outcomes)),
                               pct(score(outcomes)), "/".join(map(str, mistakes))))
        section += [
            ("Các cột dùng thứ tự L8 cầm X rồi cầm O. Số lỗi đếm cả hai bên trong từng ván; "
            "xem hồ sơ từng ván để biết tác tử và nước thay thế."),
            table(("Khai cuộc", "Ván X/O", "Chuỗi mở đầu", "W/D/L", "Điểm L8", "Lỗi X/O"),
                  comparison, "rrlcrr"),
        ]
    emit("ho-so-van", "\n".join(section))


def l8_data():
    train = "train/alphazero/20261124-160615_gpu-native"
    values = [r for r in rows(f"{train}/metrics.jsonl") if "gating" in r]
    log = rows(f"{train}/games.jsonl")
    summary = read(f"{train}/summary.json")
    accepted = [r for r in values if r["accepted"]]
    assert summary["best_iteration"] == accepted[-1]["step"] == 37
    emit("so-lieu-l8", table(("Vòng được duyệt", "Ván kiểm tra", "W/D/L", "Điểm"), [
        (r["step"], r["gating"]["n"], f"{r['gating']['wins']}/{r['gating']['draws']}/{r['gating']['losses']}",
         pct(r["gating"]["score"])) for r in accepted
    ]))
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, layout="constrained")
    x = [r["step"] for r in values]
    axes[0].plot(x, [r["gating"]["score"] * 100 for r in values], color="#526e88")
    axes[0].scatter([r["step"] for r in accepted], [r["gating"]["score"] * 100 for r in accepted],
                    color="#187366", label="Duyệt", zorder=3)
    axes[0].axhline(55, ls="--", color="0.55")
    axes[0].set(ylabel="Điểm gating (%)")
    axes[0].legend(loc="lower right")
    axes[1].plot(x, [r["loss"] for r in values], color="#ac562f")
    axes[1].set(xlabel="Vòng huấn luyện", ylabel="Loss tổng")
    save(fig, "l8")

    matches = {
        "L8 gặp L7": [r for r in log if r["tournament_id"] == "L7-PUCT[l5]-37"],
        "L8 gặp L3": rows("eval/tournament/l8_vs_l3/20261125-134954_round37/games.jsonl"),
    }
    table_rows, series = [], []
    for label, sample in matches.items():
        assert len(sample) == (20 if label.endswith("L7") else 40)
        all_results = [result_for(r, next(a for a in [r["agent_x"], r["agent_o"]] if a.startswith("L8")))
                       for r in sample]
        # Chuẩn hóa tên L8 từ log arena độc lập để dùng cùng các bộ phân tích.
        for r in sample:
            for key in ("agent_x", "agent_o"):
                if r[key].startswith("L8"):
                    r[key] = "L8"
        values_by_color = []
        for color in ("X", "O"):
            results = [result_for(r, "L8") for r in sample if r[f"agent_{color.lower()}"] == "L8"]
            table_rows.append((label, color, len(results), format_wdl(results), pct(score(results), 2), "—"))
            values_by_color.append(score(results))
        table_rows.append((label, "Cả hai", len(sample), format_wdl(all_results),
                           pct(score(all_results), 2), bootstrap(sample, "L8")))
        series.append(values_by_color)
    emit("so-lieu-doi-dau", table(("Đối sánh", "Màu", "Ván", "W/D/L", "Điểm", "Bootstrap 95%"), table_rows, "llrrrr"))
    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    for i, color in enumerate(("X", "O")):
        bars = ax.bar(np.arange(2) + (i - .5) * .32, np.array(series)[:, i] * 100, .32,
                      color=("#187366", "#d99047")[i], label=f"L8 cầm {color}")
        ax.bar_label(bars, fmt="%.1f%%", padding=2)
    ax.set(xticks=[0, 1], xticklabels=list(matches), ylim=(0, 112), ylabel="Điểm (%)")
    ax.axhline(50, ls="--", color="0.55", lw=.8)
    ax.legend()
    save(fig, "mau")
    training_profiles(values, log)
    replay_profiles(matches)


def main():
    FIG.mkdir(exist_ok=True, parents=True)
    overview_data()
    opening_profiles()
    l8_data()


if __name__ == "__main__":
    main()
