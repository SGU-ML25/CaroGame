"""Sinh các bảng của báo cáo chính (main.tex) trực tiếp từ log, ghi vào report/bang/.

    uv run python report/scripts/bang_chuong4.py

Chỉ đọc log trong runs/ và số liệu phân tích checkpoint; không chạy ván nào. Bảng top-1/top-3 đọc
report/ban-mo-rong/so-lieu-hoc-moi.json (sinh bằng report/scripts/phan_tich_hoc.py).
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from caro.arena.stats import bradley_terry, wilson
from caro.engine import CaroGame, winning_squares

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = ROOT / "report/bang"

LEVELS = ["L1-Random", "L2-Rule", "L3-AlphaBeta", "L4-MCTS", "L5-PolicySL", "L6-PolicyRL", "L7-PUCT",
          "L8-AlphaZero"]

# Mỗi bảng trỏ tới lượt chạy chỉ định (không dùng liên kết `latest`, vốn không được commit). Giải chính L1–L8
# gộp ba lượt mỗi chế độ: L1–L6, rồi L7 gặp L1–L6, rồi L8 gặp L1–L7; đủ 28 cặp, mỗi cặp đúng một lượt.
MAIN_RUNS = {
    "a": ("tournament/main_mode_a_l1_l6/20261124-212727",
          "tournament/main_mode_a_l7_additional/20261126-154639",
          "tournament/main_mode_a_l8_additional/20261126-174418"),
    "b": ("tournament/main_mode_b_l1_l6/20261124-222606",
          "tournament/main_mode_b_l7_additional/20261126-200736",
          "tournament/main_mode_b_l8_additional/20261126-221554"),
}
ERROR_RUN = "analysis/error_analysis/*"
GAMES = {
    "main_a": MAIN_RUNS["a"],
    "main_b": MAIN_RUNS["b"],
    "rr8": "eval/tournament/rr8_mode_b/20261125-233308",
    "rr8_ho": "eval/tournament/rr8_heldout/*",
    "e3_main": "tournament/e3_first_move_l1_l6/20261124-230633",
    "e3_old": "eval/tournament/e3_selfplay/20261125-233229",
    "e3_ho": "eval/tournament/e3_heldout/*",
    "a8": "eval/tournament/a8_heldout/*",
}


def run_dir(pattern: str) -> Path:
    found = sorted(p for p in RUNS.glob(pattern) if p.name != "latest" and (p / "summary.json").is_file())
    if not found:
        raise SystemExit(f"không tìm thấy lượt chạy hoàn chỉnh: runs/{pattern}")
    return found[-1]


def games(key: str) -> list[dict]:
    patterns = GAMES[key] if isinstance(GAMES[key], tuple) else (GAMES[key],)
    return [json.loads(line) for pattern in patterns
            for line in (run_dir(pattern) / "games.jsonl").open(encoding="utf-8") if line.strip()]


def num(x: float, digits: int = 1) -> str:
    text = f"{abs(x):.{digits}f}".replace(".", "{,}")
    return "$-$" + text if x < 0 and float(f"{x:.{digits}f}") != 0 else text


def count(x: float) -> str:
    return f"{x:,.0f}".replace(",", r"\,")


def short(agent: str) -> str:
    return agent.split("-")[0]


def table(name: str, caption: str, spec: str, header: list[str], rows: list[list[str]], size: str = r"\small") -> None:
    body = [r"\begin{table}[H]", r"  \centering" + size, rf"  \caption{{{caption}}}", rf"  \label{{tab:{name}}}",
            r"  \setlength{\tabcolsep}{4pt}", rf"  \begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"    \toprule",
            "    " + " & ".join(header) + r" \\", r"    \midrule"]
    body += ["    " + " & ".join(map(str, row)) + r" \\" for row in rows]
    body += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    note = "% Sinh bởi report/scripts/bang_chuong4.py — không sửa tay.\n"
    (OUT / f"{name}.tex").write_text(note + "\n".join(body) + "\n", encoding="utf-8")


def wdl(records: list[dict], agent: str) -> tuple[int, int, int]:
    c = Counter()
    for r in records:
        if r["result"] == 0:
            c["d"] += 1
        else:
            c["w" if r["result"] == (1 if r["agent_x"] == agent else -1) else "l"] += 1
    return c["w"], c["d"], c["l"]


def points(records: list[dict], agent: str) -> float:
    w, d, _ = wdl(records, agent)
    return (w + 0.5 * d) / len(records)


def move_values(records: list[dict], agent: str, key: str) -> list[float]:
    """Giá trị theo từng nước (time_ms, nodes, sims) của đúng tác tử, theo số nước khai cuộc."""
    out: list[float] = []
    for r in records:
        values = r.get(key) or []
        first = (len(r["moves"]) - len(r["time_ms"])) % 2          # bên đi nước đo đầu tiên
        for side, name in enumerate((r["agent_x"], r["agent_o"])):
            if name == agent:
                out += values[(side - first) % 2::2]
    return out


def agents_of(records: list[dict]) -> list[str]:
    present = {r["agent_x"] for r in records} | {r["agent_o"] for r in records}
    return [a for a in LEVELS if a in present]


def elo_ci(records: list[dict], reps: int = 1000, seed: int = 2026) -> dict[str, tuple[float, float, float]]:
    """Điểm Bradley-Terry và khoảng 95% bootstrap theo khai cuộc (lấy mẫu lại cả cụm ván cùng khai cuộc)."""
    clusters = defaultdict(list)
    for r in records:
        clusters[r["opening_id"]].append((r["agent_x"], r["agent_o"], r["result"]))
    units = list(clusters.values())
    point = bradley_terry(m for u in units for m in u)
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(reps):
        pick = rng.integers(0, len(units), len(units))
        for agent, value in bradley_terry(m for i in pick for m in units[i]).items():
            samples[agent].append(value)
    return {a: (point[a], *np.quantile(samples[a], [0.025, 0.975])) for a in point}


def ranking_tables(key: str, name: str, title: str) -> None:
    records = games(key)
    agents = agents_of(records)
    rows = []
    for a in agents:
        sub = [r for r in records if a in (r["agent_x"], r["agent_o"])]
        w, d, lost = wdl(sub, a)
        lo, hi = wilson(w + 0.5 * d, len(sub))
        times = move_values(sub, a, "time_ms")
        rows.append((points(sub, a), [
            a, f"{w}/{d}/{lost}", num(100 * points(sub, a)), f"{num(100 * max(0, lo))}--{num(100 * hi)}",
            num(100 * points([r for r in sub if r["agent_x"] == a], a)),
            num(100 * points([r for r in sub if r["agent_o"] == a], a)),
            num(float(np.median(times))), num(float(np.quantile(times, 0.95)))]))
    n = sum(1 for r in records if agents[0] in (r["agent_x"], r["agent_o"]))
    table(name, f"{title}: điểm của từng tác tử trên {n} ván (\\%), khoảng Wilson 95\\%, điểm theo màu "
          "và thời gian mỗi nước.", "lrrrrrrr",
          ["Tác tử", "T/H/B", "Điểm", "Wilson 95\\%", "Cầm X", "Cầm O", "P50 (ms)", "P95 (ms)"],
          [row for _, row in sorted(rows, key=lambda x: -x[0])], r"\footnotesize")

    matrix = []
    per_pair = set()
    for a in agents:
        cells = [a]
        for b in agents:
            sub = [r for r in records if {r["agent_x"], r["agent_o"]} == {a, b}] if a != b else []
            if sub:
                per_pair.add(len(sub))
            cells.append(num(100 * points(sub, a)) if sub else "---")
        matrix.append(cells)
    table(name + "-matran", f"{title}: ma trận điểm (\\%) của tác tử ở hàng trước tác tử ở cột, "
          f"{'/'.join(map(str, sorted(per_pair)))} ván mỗi cặp.", "l" + "r" * len(agents),
          ["Tác tử"] + [short(a) for a in agents], matrix, r"\footnotesize")


def main_runs_table() -> None:
    """Điều kiện của sáu lượt tạo nên giải chính: commit, trạng thái mã và khối lượng tìm kiếm đạt được."""
    rows = []
    for mode in ("a", "b"):
        for pattern, label in zip(MAIN_RUNS[mode], ("L1--L6", "L7", "L8"), strict=True):
            folder = run_dir(pattern)
            records = [json.loads(line) for line in (folder / "games.jsonl").open(encoding="utf-8")]
            git = json.loads((folder / "meta.json").read_text(encoding="utf-8"))["git"]
            cells = [mode.upper(), label, count(len(records)), rf"\texttt{{{git['commit'][:7]}}}",
                     "có" if git["dirty"] else "không",
                     count(float(np.median(move_values(records, "L3-AlphaBeta", "time_ms"))))]
            for agent in ("L4-MCTS", "L7-PUCT", "L8-AlphaZero"):
                values = move_values(records, agent, "sims")
                cells.append(count(np.mean([v for v in values if v])) if values else "---")
            rows.append(cells)
    table("giai-luot", "Sáu lượt chạy tạo nên giải chính L1--L8: commit và cờ \\texttt{dirty} trong "
          "\\texttt{meta.json}, thời gian trung vị mỗi nước của L3 và số mô phỏng trung bình mỗi nước có tìm kiếm.",
          "llrlrrrrr", ["Chế độ", "Lượt", "Ván", "Commit", "Dirty", "L3 P50 (ms)", "L4 mp", "L7 mp", "L8 mp"],
          rows, r"\footnotesize")


def error_table() -> None:
    summary = json.loads((run_dir(ERROR_RUN) / "summary.json").read_text(encoding="utf-8"))
    categories = ("thang_ngay", "chan_nam", "chan_don_kep")
    rows = []
    for mode in ("a", "b"):
        total = defaultdict(Counter)
        for item in summary["runs"]:
            if f"main_mode_{mode}_" not in item["source"]:
                continue
            for agent, row in item["by_agent"].items():
                total[agent].update(moves=row["moves"], lost=row["da_thua"])
                for c in categories:
                    total[agent].update({c: row[c]["chances"], c + "_sai": row[c]["errors"]})
        rule = r"\midrule " if mode == "b" else ""
        rows.append([rule + rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{Chế độ {mode.upper()}}}}}"])
        for agent in LEVELS:
            t = total[agent]
            cells = [agent, count(t["moves"])]
            for c in categories:
                cells.append(f"{count(t[c + '_sai'])}/{count(t[c])} ({num(100 * t[c + '_sai'] / t[c])})"
                             if t[c] else "---")
            rows.append(cells + [count(t["lost"])])
    table("loi-theo-cap", "Lỗi chiến thuật trên toàn bộ giải chính L1--L8: số nước sai / số cơ hội (tỷ lệ \\%). "
          "Nước khai cuộc không tính. Cột cuối đếm số lượt đi ở thế đã thua không cứu được trong 4 ply.",
          "lrrrrr", ["Tác tử", "Nước xét", "Thắng ngay", "Chặn năm", "Chặn đòn kép", "Đã thua"], rows,
          r"\footnotesize")


def elo_table() -> None:
    columns = {k: elo_ci(games(k)) for k in ("main_a", "main_b", "rr8", "rr8_ho")}
    rows = []
    for a in sorted(LEVELS, key=lambda a: -columns["rr8_ho"][a][0]):
        cells = [a]
        for k in ("main_a", "main_b", "rr8"):
            cells.append(num(columns[k][a][0], 0) if a in columns[k] else "---")
        point, lo, hi = columns["rr8_ho"][a]
        cells += [num(point, 0), f"{num(lo, 0)} đến {num(hi, 0)}"]
        rows.append(cells)
    table("elo", "Điểm Elo ước lượng theo Bradley-Terry (một ván hoà ảo mỗi cặp, trung bình mỗi cột là 1000). "
          "Chỉ so sánh trong cùng một cột. Khoảng 95\\% lấy bootstrap theo khai cuộc, 1000 lần.", "lrrrrr",
          ["Tác tử", "Chế độ A", "Chế độ B", "RR8 cũ", "RR8 giữ riêng", "Khoảng 95\\%"], rows)


def heldout_compare_table() -> None:
    old, new = games("rr8"), games("rr8_ho")
    old_elo, new_elo = bradley_terry((r["agent_x"], r["agent_o"], r["result"]) for r in old), bradley_terry(
        (r["agent_x"], r["agent_o"], r["result"]) for r in new)
    rows = []
    for a in sorted(LEVELS, key=lambda a: -new_elo[a]):
        o = points([r for r in old if a in (r["agent_x"], r["agent_o"])], a)
        n = points([r for r in new if a in (r["agent_x"], r["agent_o"])], a)
        rows.append([a, num(100 * o), num(100 * n), ("$+$" if n >= o else "$-$") + num(100 * abs(n - o)),
                     num(old_elo[a], 0), num(new_elo[a], 0)])
    table("rr8-so-sanh", "Điểm (\\%) và Elo của RR8 trên bộ khai cuộc cũ (đã có trong dữ liệu huấn luyện) "
          "và bộ giữ riêng. Mỗi tác tử 700 ván ở mỗi bộ.", "lrrrrr",
          ["Tác tử", "Điểm cũ", "Điểm giữ riêng", "Chênh", "Elo cũ", "Elo giữ riêng"], rows)


def cost_table() -> None:
    runs = {key: games(key) for key in ("rr8", "rr8_ho")}
    rows = []
    for a in LEVELS:
        cells = [a]
        for key in ("rr8", "rr8_ho"):
            times = move_values(runs[key], a, "time_ms")
            cells += [num(float(np.mean(times))), num(float(np.median(times))), num(float(np.quantile(times, 0.95)))]
        nodes, sims = move_values(runs["rr8_ho"], a, "nodes"), move_values(runs["rr8_ho"], a, "sims")
        cells.append(f"{count(np.mean(nodes))} nút" if any(nodes) else
                     f"{count(np.mean(sims))} mô phỏng" if any(sims) else "---")
        rows.append(cells)
    table("chi-phi-nuoc", "Thời gian mỗi nước (ms) ở hai lượt RR8 trên cùng máy (một tiến trình, CPU Intel Core "
          "i7-11800H; L3, L7, L8 dùng lõi Cython) và khối lượng tìm kiếm trung bình mỗi nước ở lượt giữ riêng.",
          "lrrrrrrr", ["", r"\multicolumn{3}{c}{RR8 cũ}", r"\multicolumn{3}{c}{RR8 giữ riêng}",
                       (r" \\ \cmidrule(lr){2-4} \cmidrule(lr){5-7}" "\n    "
                        "Tác tử & TB & P50 & P95 & TB & P50 & P95 & Tìm kiếm")],
          rows, r"\footnotesize")


def ram_table() -> None:
    data = json.loads((RUNS / "bench/ram_rss.json").read_text(encoding="utf-8"))
    table("ram", "Đỉnh bộ nhớ thường trú (RSS) của một tiến trình chỉ nạp một tác tử và tìm nước trên 10 thế "
          "giữa ván, ngân sách như RR8, CPU. Gồm cả trình thông dịch và thư viện (PyTorch với L5--L8).",
          "lr", ["Tác tử", "RSS đỉnh (MB)"], [[a, count(data[a])] for a in LEVELS])


def e3_tables() -> None:
    records = games("e3_ho")
    rows = []
    for a in agents_of(records):
        sub = [r for r in records if r["agent_x"] == a]
        xw, ow = sum(r["result"] == 1 for r in sub), sum(r["result"] == -1 for r in sub)
        d, n = len(sub) - xw - ow, len(sub)
        score = (xw + 0.5 * d) / n
        lo, hi = wilson(xw + 0.5 * d, n)
        parity = []
        for p in (0, 1):                       # 0: X đi ngay sau khai cuộc, 1: O đi ngay sau khai cuộc
            part = [r for r in sub if r["opening_plies"] % 2 == p]
            parity.append(num(100 * sum((r["result"] + 1) / 2 for r in part) / len(part)))
        rows.append([a, n, len({tuple(r["moves"]) for r in sub}), f"{xw}/{d}/{ow}", num(100 * score),
                     f"{num(100 * lo)}--{num(100 * hi)}", *parity])
    table("e3-giu-rieng", "E3 trên bộ khai cuộc giữ riêng, mỗi khai cuộc một ván. T/H/B và điểm tính cho X. "
          "Hai cột cuối tách theo bên đi ngay sau khai cuộc (khai cuộc 2 và 4 nước: X; 3 nước: O).",
          "lrrrrrrr", ["Tác tử", "Ván", "Ván khác nhau", "X: T/H/B", "Điểm X", "Wilson 95\\%", "X đi tiếp",
                       "O đi tiếp"], rows, r"\footnotesize")

    old = {"main": games("e3_main"), "old": games("e3_old")}
    # L1–L4 của lượt tự đấu 8 cấp nằm ở lượt chạy trước đó (cùng cấu hình, chưa ghi summary).
    extra = RUNS / "eval/tournament/e3_selfplay/20261125-232320/games.jsonl"
    old["old"] += [json.loads(line) for line in extra.open(encoding="utf-8")
                   if json.loads(line)["agent_x"] in LEVELS[:4]]
    rows = []
    for a in LEVELS:
        cells = [a]
        for key in ("main", "old"):
            sub = [r for r in old[key] if r["agent_x"] == a and r["agent_o"] == a]
            if not sub:
                cells += ["---", "---"]
                continue
            cells += [f"{len({tuple(r['moves']) for r in sub})}/{len(sub)}",
                      num(100 * sum((r["result"] + 1) / 2 for r in sub) / len(sub))]
        rows.append(cells)
    table("e3-cu", "Hai lượt tự đấu trên bộ khai cuộc cũ: số ván khác nhau trên tổng số ván và điểm của X (\\%). "
          "Không ghi khoảng tin cậy vì nhiều ván trùng nhau.", "lrrrr",
          ["Tác tử", "L1--L6: khác/tổng", "Điểm X", "L1--L8: khác/tổng", "Điểm X"], rows)


def a8_table() -> None:
    records = games("a8")
    rows = []
    for agent, label in (("MCTS-light", "Tắt"), ("MCTS-light-tactic", "Bật")):
        for opponent in ("L2-Rule", "L3-AlphaBeta"):
            sub = [r for r in records if {r["agent_x"], r["agent_o"]} == {agent, opponent}]
            w, d, lost = wdl(sub, agent)
            lo, hi = wilson(w + 0.5 * d, len(sub))
            rows.append([label, opponent, f"{w}/{d}/{lost}", num(100 * points(sub, agent)),
                         f"{num(100 * max(0, lo))}--{num(100 * hi)}", num(np.mean([len(r["moves"]) for r in sub])),
                         count(np.mean(move_values(sub, agent, "sims"))),
                         count(float(np.median(move_values(sub, agent, "time_ms"))))])
    table("a8", "A8: MCTS (800 mô phỏng, rollout \\texttt{light}) tắt và bật chiến thuật ở gốc, 100 ván mỗi dòng "
          "trên khai cuộc giữ riêng. Số mô phỏng trung bình gồm cả nước trả ngay (0 mô phỏng). Thời gian đo khi chạy "
          "4 tiến trình song song với một giải khác.", "llrrrrrr",
          ["Gốc", "Đối thủ", "T/H/B", "Điểm", "Wilson 95\\%", "Số nước TB", "Mô phỏng TB", "P50 (ms)"], rows,
          r"\footnotesize")


def sl_table() -> None:
    data = json.loads((ROOT / "report/ban-mo-rong/so-lieu-hoc-moi.json").read_text(encoding="utf-8"))
    rows = []
    for name, m in data["models"].items():
        v = m["value_relative"]
        label = {"A6 no-aug": "SL v3 không augment"}.get(name, name)
        rows.append([label, num(100 * m["top1"]), num(100 * m["top3"]), num(m["policy_ce"], 3), num(v["mse"], 3),
                     num(v["corr"], 3), num(100 * v["sign_acc"])])
    test = f"{data['n_test']:,}".replace(",", r"\,")
    table("sl-test", f"Đo trên tập test của expert\\_v2 ({test} mẫu, {data['n_games_test']} ván, tách theo ván "
          f"đúng như lúc huấn luyện, seed {data['split_seed']}). Value theo góc nhìn người đi; cột đúng dấu tính trên mọi mẫu, "
          "mẫu thuộc ván hoà luôn bị tính là sai. SL v1 học trên expert\\_v1 nên dòng đó là phép đo chéo dữ liệu.", "lrrrrrr",
          ["Mạng", "Top-1 (\\%)", "Top-3 (\\%)", "CE policy", "MSE value", "Tương quan", "Đúng dấu (\\%)"], rows, r"\footnotesize")


def checkpoint_scores(name: str) -> dict[str, dict]:
    with (RUNS / "eval" / name).open(encoding="utf-8", newline="") as f:
        return {row["opponent"]: row for row in csv.DictReader(f)}


def eval_cell(row: dict) -> str:
    return f"{num(100 * float(row['score']))} [{num(100 * float(row['ci_low']))}; " \
           f"{num(100 * float(row['ci_high']))}]"


def a5b_table() -> None:
    sl, rnd = checkpoint_scores("a5b_from_sl_best.csv"), checkpoint_scores("a5b_from_random_best.csv")
    rows = []
    for key, label in (("l1_random", "L1-Random"), ("l2_rule", "L2-Rule"), ("l3_alphabeta", "L3-AlphaBeta"),
                       ("l5_policy_sl", "L5-PolicySL")):
        rows.append([label, eval_cell(sl[key]), eval_cell(rnd[key])])
    table("a5b", "A5b: điểm (\\%) và khoảng Wilson 95\\% của checkpoint tốt nhất sau 49\\,920 episode REINFORCE, "
          "100 ván mỗi ô, một seed.", "lrr", ["Đối thủ", "Khởi tạo từ SL v3", "Khởi tạo ngẫu nhiên"], rows)


def a6_table() -> None:
    aug, noaug = checkpoint_scores("a6_sl_aug.csv"), checkpoint_scores("a6_sl_noaug.csv")
    rows = [[label, eval_cell(aug[key]), eval_cell(noaug[key])]
            for key, label in (("l1_random", "L1-Random"), ("l2_rule", "L2-Rule"), ("l3_alphabeta", "L3-AlphaBeta"))
            if key in aug and key in noaug]
    table("a6", "A6: điểm (\\%) và khoảng Wilson 95\\% khi đấu, SL v3 có và không tăng cường đối xứng, "
          "100 ván mỗi ô.", "lrr", ["Đối thủ", "Có augment", "Không augment"], rows)


def bench_rows(folder: Path) -> dict[int, dict[str, tuple[int, float, int, int]]]:
    """Từ run.log của một lượt benchmark: thế → tác tử → (nút, ms, nước chọn, giá trị gốc)."""
    pattern = re.compile(r"thế\s+(\d+) \| (\S+)\s*\| nodes_med=\s*(\d+) \| time_med=\s*([\d.]+) ms \| "
                         r"move=(\d+) \| score=(-?\d+)")
    per_pos: dict = defaultdict(dict)
    for line in (folder / "run.log").open(encoding="utf-8"):
        if m := pattern.search(line):
            per_pos[int(m[1])][m[2]] = (int(m[3]), float(m[4]), int(m[5]), int(m[6]))
    return per_pos


def bench_row(per_pos: dict, agent: str, base: str) -> list[str]:
    vals = [p[agent] for p in per_pos.values()]
    times = sorted(v[1] for v in vals)
    return [count(statistics.median(v[0] for v in vals)), num(statistics.median(times)),
            num(times[int(len(times) * 0.95)]),           # cùng cách lấy p95 với caro/arena/bench.py
            f"{sum(p[agent][3] == p[base][3] for p in per_pos.values())}/{len(per_pos)}",
            f"{sum(p[agent][2] == p[base][2] for p in per_pos.values())}/{len(per_pos)}"]


def a1_a7_tables() -> None:
    header = ["Cấu hình", "Nút P50", "P50 (ms)", "P95 (ms)", "Cùng giá trị", "Cùng nước"]
    a1 = bench_rows(RUNS / "bench/a1_ab_pruning/20261029-200321")
    table("a1", "A1: bật/tắt cắt tỉa ở độ sâu 3 trên 50 thế (lõi Python, máy Intel Model 140).", "lrrrrr", header,
          [["Có cắt tỉa", *bench_row(a1, "AB-prune", "AB-full")],
           ["Không cắt tỉa", *bench_row(a1, "AB-full", "AB-full")]])
    a7 = bench_rows(RUNS / "bench/a7_eval_weights/20261029-223250")
    table("a7", "A7: trọng số lượng giá mặc định và bản bỏ thành phần phòng thủ, 10 thế, giới hạn thời gian "
          "(lõi Python). Hai cột cuối so với bản mặc định.", "lrrrrr", header,
          [["Mặc định", *bench_row(a7, "AB-default", "AB-default")],
           ["Bỏ phòng thủ", *bench_row(a7, "AB-no-defense", "AB-default")]])


def a2_table() -> None:
    rows = []
    for depth in (3, 4, 5):
        found = sorted(RUNS.glob(f"bench/a2_fixed_depth/*_d{depth}"))
        if not found:
            raise SystemExit(f"thiếu lượt A2 độ sâu {depth}: python -m caro.arena.bench experiments/a2_fixed_depth "
                             f"budget.max_depth={depth} run.tag=d{depth}")
        per_pos = bench_rows(found[-1])
        base = statistics.median(p["AB-plain"][0] for p in per_pos.values())
        for agent, label in (("AB-plain", "Không sắp xếp"), ("AB-order", "Sắp xếp"),
                             ("AB-order-tt", "Sắp xếp + TT")):
            nodes, p50, _, value, move = bench_row(per_pos, agent, "AB-plain")
            median_nodes = statistics.median(p[agent][0] for p in per_pos.values())
            rows.append([depth, label, nodes, num(base / median_nodes), p50, value, move])
    table("a2", "A2 ở độ sâu cố định, không giới hạn thời gian, 10 thế, lõi Cython: số nút và thời gian trung vị, "
          "hệ số giảm nút, số thế có cùng giá trị gốc và cùng nước chọn so với bản không sắp xếp.", "rlrrrrr",
          ["Độ sâu", "Cấu hình", "Nút", "Giảm (lần)", "Thời gian (ms)", "Cùng giá trị", "Cùng nước"], rows)


def board_figure(name: str, record: dict, ply: int, caption: str) -> None:
    """Vẽ thế cờ trước nước thứ `ply` (đếm từ 1, gồm cả khai cuộc), kiểm lại bằng engine.

    Ô đỏ nhạt: nước thực sự được chọn. Ô xanh: ô duy nhất chặn được năm quân của đối thủ.
    """
    game = CaroGame.from_moves(record["moves"][:ply - 1])
    me, opp = game.me_opp()
    threats = winning_squares(opp, me)
    chosen = record["moves"][ply - 1]
    if winning_squares(me, opp) or threats.bit_count() != 1 or (threats >> chosen) & 1:
        raise SystemExit(f"{name}: nước {ply} không phải trường hợp bỏ chặn duy nhất")
    block = threats.bit_length() - 1
    lines = [r"\begin{figure}[H]", r"  \centering", r"  \begin{tikzpicture}[x=.5cm, y=.5cm]"]
    for cell, colour in ((chosen, "red!25"), (block, "green!35")):
        r, c = divmod(cell, 11)
        lines.append(rf"    \fill[{colour}] ({c},{-r}) rectangle ++(1,-1);")
    lines.append(r"    \draw[gray!55, line width=.3pt, step=1] (0,0) grid (11,-11);")
    for i in range(11):
        lines.append(rf"    \node[font=\scriptsize, text=gray] at ({i}.5,.45) {{{'ABCDEFGHIJK'[i]}}};")
        lines.append(rf"    \node[font=\scriptsize, text=gray] at (-.55,-{i}.5) {{{i + 1}}};")
    for number, cell in enumerate(record["moves"][:ply - 1], 1):
        r, c = divmod(cell, 11)
        fill = "black!85" if number % 2 else "red!75!black"
        lines.append(rf"    \fill[{fill}] ({c}.5,-{r}.5) circle[radius=.40];")
        lines.append(rf"    \node[font=\tiny, text=white] at ({c}.5,-{r}.5) {{{number}}};")
    lines += [r"  \end{tikzpicture}", rf"  \caption{{{caption}}}", rf"  \label{{fig:{name}}}", r"\end{figure}"]
    note = "% Sinh bởi report/scripts/bang_chuong4.py — không sửa tay.\n"
    (OUT / f"{name}.tex").write_text(note + "\n".join(lines) + "\n", encoding="utf-8")


def example_boards() -> None:
    def pick(path: str, game_id: int, test) -> dict:
        for line in (RUNS / path).open(encoding="utf-8"):
            record = json.loads(line)
            if record["game_id"] == game_id and test(record):
                return record
        raise SystemExit(f"không thấy ván {game_id} trong {path}")

    duel = pick("train/alphazero/20261124-160615_gpu-native/games.jsonl", 13,
                lambda r: "PUCT" in str(r["tournament_id"]))
    board_figure("van13", duel, 35, "Ván 13, thế trước nước 35. L7 (X, quân đen) đi B5 (ô đỏ nhạt) trong khi "
                 "G4 (ô xanh) là ô thắng ngay duy nhất của L8 (O, quân đỏ). Số trên quân là thứ tự nước.")
    versus = pick("eval/tournament/l8_vs_l3/20261125-134954_round37/games.jsonl", 26, lambda r: True)
    board_figure("van26", versus, 20, "Ván 26, thế trước nước 20. L8 (O, quân đỏ) đi C8 (ô đỏ nhạt) trong khi "
                 "E7 (ô xanh) là ô thắng ngay duy nhất của AlphaBeta (X, quân đen).")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ranking_tables("main_a", "giai-a", "Giải chính, chế độ A (1\\,000 ms/nước)")
    ranking_tables("main_b", "giai-b", "Giải chính, chế độ B (ngân sách cố định)")
    ranking_tables("rr8_ho", "rr8-giu-rieng", "RR8 trên khai cuộc giữ riêng")
    main_runs_table()
    error_table()
    heldout_compare_table()
    elo_table()
    cost_table()
    ram_table()
    e3_tables()
    a8_table()
    sl_table()
    a5b_table()
    a6_table()
    a1_a7_tables()
    a2_table()
    example_boards()
    print("xong; ghi", OUT)


if __name__ == "__main__":
    main()
