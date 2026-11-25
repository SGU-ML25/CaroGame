"""Sinh bảng, hình và số liệu tổng hợp cho báo cáo mở rộng.

Chạy từ gốc repo (một tiến trình, chỉ đọc log):

    uv run python report/scripts/phan-tich-tong-hop.py

Ghi bảng LaTeX vào report/ban-mo-rong/, hình vào report/ban-mo-rong/hinh/,
và bản tóm tắt JSON vào report/ban-mo-rong/so-lieu-tong-hop.json.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/ban-mo-rong"
FIG = OUT / "hinh"
RUNS = ROOT / "runs"
import sys

sys.path.insert(0, str(ROOT))
from caro.engine import CaroGame, iter_bits, winning_squares

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": 190, "savefig.facecolor": "white",
})
C1, C2, C3, C4 = "#166d63", "#c26a3d", "#526e88", "#8a6f3e"


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


SPECIAL = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "&": r"\&", "%": r"\%",
           "$": r"\$", "#": r"\#", "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(text) -> str:
    return "".join(SPECIAL.get(ch, ch) for ch in str(text))


def dec(x, d=1) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d=1) -> str:
    return dec(100 * x, d) + "%"


def table(headers, rows, spec=None) -> str:
    spec = spec or ("l" + "r" * (len(headers) - 1))
    lines = [r"\begin{center}\small", rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule",
             " & ".join(tex(h) for h in headers) + r" \\", r"\midrule"]
    lines += [" & ".join(tex(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(lines)


def save_fig(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_pairs(records: list[dict], agent: str, seed: int = 42) -> tuple[float, float]:
    by_op = defaultdict(list)
    for r in records:
        if agent in (r["agent_x"], r["agent_o"]):
            mine = 1 if r["agent_x"] == agent else -1
            by_op[r["opening_id"]].append(0.5 if r["result"] == 0 else (1.0 if r["result"] == mine else 0.0))
    units = np.array([np.mean(v) for v in by_op.values() if v])
    if len(units) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(units, (20000, len(units)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def outcome_for(record: dict, agent: str) -> int:
    mine = 1 if record["agent_x"] == agent else -1
    if record["result"] == 0:
        return 0
    return 1 if record["result"] == mine else -1


def wdl(records: list[dict], agent: str) -> tuple[int, int, int]:
    c = Counter(outcome_for(r, agent) for r in records)
    return c[1], c[0], c[-1]


def score(records: list[dict], agent: str) -> float:
    w, d, n = *wdl(records, agent)[:2], len(records)
    return (w + 0.5 * d) / n if n else float("nan")


# ---------------------------------------------------------------------------
# Bradley-Terry (Elo xấp xỉ) bằng gradient ascent
# ---------------------------------------------------------------------------

def bradley_terry(records: list[dict], agents: list[str], iters: int = 20000,
                  lr: float = 0.5, reg: float = 1e-3) -> dict[str, float]:
    """Bradley-Terry có điều chuẩn nhẹ; chặn ±8 log-odds vì kỷ lục tuyệt đối cho MLE vô hạn."""
    idx = {a: i for i, a in enumerate(agents)}
    rating = np.zeros(len(agents))
    games = []
    for r in records:
        if r["agent_x"] not in idx or r["agent_o"] not in idx:
            continue
        s = 0.5 if r["result"] == 0 else (1.0 if r["result"] == 1 else 0.0)
        games.append((idx[r["agent_x"]], idx[r["agent_o"]], s))
    if not games:
        return {a: 1000.0 for a in agents}
    for _ in range(iters):
        grad = -2.0 * reg * rating
        for i, j, s in games:
            p = 1.0 / (1.0 + 10 ** ((rating[j] - rating[i]) / 400.0))
            g = s - p
            grad[i] += g
            grad[j] -= g
        rating += lr * grad / max(1, len(games))
        rating -= rating.mean()
        rating = np.clip(rating, -8.0, 8.0)
    scale = 400.0 / math.log(10)
    return {a: float(1000 + scale * rating[idx[a]]) for a in agents}


# ---------------------------------------------------------------------------
# Thống kê một giải
# ---------------------------------------------------------------------------

def summarise_tournament(records: list[dict], agents: list[str], tag: str) -> dict:
    ranking = []
    for a in agents:
        sub = [r for r in records if a in (r["agent_x"], r["agent_o"])]
        if not sub:
            continue
        w, d, l = wdl(sub, a)
        n = len(sub)
        lo, hi = wilson(w + 0.5 * d, n)
        ranking.append({"agent": a, "n": n, "w": w, "d": d, "l": l, "score": (w + 0.5 * d) / n,
                        "ci_low": lo, "ci_high": hi})
    bt = bradley_terry(records, agents)
    for entry in ranking:
        entry["bt"] = bt[entry["agent"]]
    ranking.sort(key=lambda x: (-x["score"], x["agent"]))
    color = {}
    lengths = {}
    for a in agents:
        x = [r for r in records if r["agent_x"] == a]
        o = [r for r in records if r["agent_o"] == a]
        color[a] = {
            "x": (wdl(x, a), len(x)), "o": (wdl(o, a), len(o)),
        }
        lengths[a] = [len(r["moves"]) for r in records if a in (r["agent_x"], r["agent_o"])]
    return {"tag": tag, "n_games": len(records), "ranking": ranking, "bt": bt,
            "color": color, "lengths": lengths}


def matrix_table(records: list[dict], agents: list[str]) -> str:
    header = " & ".join([""] + [tex(a) for a in agents])
    lines = [r"\begin{center}\scriptsize\setlength{\tabcolsep}{3pt}",
             rf"\begin{{tabular}}{{@{{}}l{'c' * len(agents)}@{{}}}}", r"\toprule",
             header + r" \\", r"\midrule"]
    for a in agents:
        cells = []
        for b in agents:
            if a == b:
                cells.append("—")
                continue
            sub = [r for r in records if {r["agent_x"], r["agent_o"]} == {a, b}]
            if not sub:
                cells.append("·")
                continue
            w, d, l = wdl(sub, a)
            n = len(sub)
            cells.append(f"{100 * (w + 0.5 * d) / n:.0f}")
        lines.append(tex(a) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phân tích chiến thuật
# ---------------------------------------------------------------------------

def opening_length(record: dict) -> int:
    if record.get("time_ms"):
        return max(0, len(record["moves"]) - len(record["time_ms"]))
    return 0


def agent_series(record: dict, agent: str, key: str) -> list:
    """Các giá trị per-move (time_ms, sims, nodes) thuộc về `agent`, bỏ nước khai cuộc."""
    values = record.get(key) or []
    if not values:
        return []
    first_x = opening_length(record) % 2 == 0
    start = 0 if first_x == (record["agent_x"] == agent) else 1
    return values[start::2]


def tactical_stats(records: list[dict], label: str) -> dict:
    """Đếm cơ hội thắng ngay / chặn bắt buộc và số lần bỏ lỡ, theo từng tác tử."""
    stats = defaultdict(lambda: Counter())
    decisive = Counter()
    for rec in records:
        g = CaroGame.from_moves(rec["moves"][:opening_length(rec)])
        moves = rec["moves"][opening_length(rec):]
        agents = {0: rec["agent_x"], 1: rec["agent_o"]}
        for move in moves:
            side = g.side
            me, opp = g.me_opp()
            wins = winning_squares(me, opp)
            threats = winning_squares(opp, me)
            k_wins = wins.bit_count()
            k_threats = threats.bit_count()
            a = agents[side]
            stats[a]["moves"] += 1
            if k_wins:
                stats[a]["win_opp"] += 1
                if (wins >> move) & 1:
                    stats[a]["win_taken"] += 1
                else:
                    stats[a]["win_missed"] += 1
            elif k_threats == 1:
                stats[a]["block_opp"] += 1
                if (threats >> move) & 1:
                    stats[a]["block_taken"] += 1
                else:
                    stats[a]["block_missed"] += 1
            elif k_threats >= 2:
                stats[a]["already_lost"] += 1
            g.step(move)
        # nước cuối của bên thua: có buộc chặn không?
        if rec["result"] != 0 and moves:
            loser = rec["agent_o"] if rec["result"] == 1 else rec["agent_x"]
            decisive[loser] += 1
    return {"label": label, "stats": {a: dict(c) for a, c in stats.items()},
            "decisive_losses": dict(decisive)}


# ---------------------------------------------------------------------------
# Các khối sinh bảng
# ---------------------------------------------------------------------------

def make_rr(tag: str, records: list[dict], agents: list[str]) -> dict:
    out = summarise_tournament(records, agents, tag)
    rows = [(r["agent"], r["n"], f"{r['w']}/{r['d']}/{r['l']}", pct(r["score"]),
             f"[{pct(r['ci_low'])}; {pct(r['ci_high'])}]")
            for r in out["ranking"]]
    text = table(("Tác tử", "Ván", "W/D/L", "Điểm", "Wilson 95%"), rows, "lrrrr")
    text += "\n" + matrix_table(records, agents)
    (OUT / f"so-lieu-{tag}.tex").write_text(text, encoding="utf-8")

    # hình xếp hạng
    fig, ax = plt.subplots(figsize=(7.4, 3.6), layout="constrained")
    names = [r["agent"] for r in out["ranking"]][::-1]
    vals = [100 * r["score"] for r in out["ranking"]][::-1]
    lo = [100 * (r["score"] - r["ci_low"]) for r in out["ranking"]][::-1]
    hi = [100 * (r["ci_high"] - r["score"]) for r in out["ranking"]][::-1]
    ax.barh(names, vals, color=C1, alpha=.85)
    ax.errorbar(vals, names, xerr=[lo, hi], fmt="none", ecolor="0.25", capsize=3, lw=1)
    ax.axvline(50, ls="--", color="0.5", lw=.8)
    ax.set(xlabel="Điểm (%)", xlim=(0, 100))
    save_fig(fig, f"{tag}_ranking")

    # hình theo màu
    fig, ax = plt.subplots(figsize=(7.4, 3.4), layout="constrained")
    xs = np.arange(len(agents))
    for i, (color, key) in enumerate((("X", "x"), ("O", "o"))):
        vals = []
        for a in agents:
            (w, d, l), n = out["color"][a][key]
            vals.append(100 * (w + .5 * d) / n if n else 0)
        ax.bar(xs + (i - .5) * .36, vals, .36, label=f"cầm {color}", color=(C1, C2)[i])
    ax.set(xticks=xs, xticklabels=agents, ylabel="Điểm (%)", ylim=(0, 100))
    ax.tick_params(axis="x", rotation=20)
    ax.axhline(50, ls="--", color="0.5", lw=.8)
    ax.legend()
    save_fig(fig, f"{tag}_color")

    # phân bố độ dài
    fig, ax = plt.subplots(figsize=(7.4, 3.2), layout="constrained")
    data = [out["lengths"][a] for a in agents if out["lengths"][a]]
    labels = [a for a in agents if out["lengths"][a]]
    ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#d8e6e3", edgecolor=C1),
               medianprops=dict(color=C2, lw=1.4))
    ax.set(ylabel="Số nước", xlabel="Tác tử")
    ax.tick_params(axis="x", rotation=20)
    save_fig(fig, f"{tag}_lengths")
    return out


def make_a3a4() -> dict:
    sources = {
        "A3": RUNS / "tournament/a3_mcts_sims/20261114-232353/games.jsonl",
        "A4": RUNS / "tournament/a4_rollout/20261114-234556/games.jsonl",
    }
    rows, summary = [], {}
    for tag, path in sources.items():
        records = read_jsonl(path)
        if not records:
            continue
        challengers = sorted({a for r in records for a in (r["agent_x"], r["agent_o"]) if a.startswith("MCTS")})
        rivals = sorted({a for r in records for a in (r["agent_x"], r["agent_o"]) if not a.startswith("MCTS")})
        for c in challengers:
            for rival in rivals:
                sub = [r for r in records if {r["agent_x"], r["agent_o"]} == {c, rival}]
                if not sub:
                    continue
                w, d, l = wdl(sub, c)
                n = len(sub)
                times = [t for r in sub for t in agent_series(r, c, "time_ms")]
                sims = [s for r in sub for s in agent_series(r, c, "sims")]
                rate = (sum(sims) / (sum(times) / 1000)) if times and sum(times) > 0 else float("nan")
                rows.append((tag, c, rival.split("-")[0], n, f"{w}/{d}/{l}", pct((w + .5 * d) / n),
                             dec(np.median(times), 1) if times else "—",
                             dec(np.quantile(times, .95), 1) if times else "—",
                             f"{rate:.0f}" if rate == rate else "—"))
                summary[f"{tag}:{c}:{rival}"] = {"score": (w + .5 * d) / n, "w": w, "d": d, "l": l,
                                                  "time_median": float(np.median(times)) if times else None,
                                                  "sims_per_s": rate}
    (OUT / "so-lieu-a3a4.tex").write_text(
        table(("Mã", "Cấu hình", "Gặp", "Ván", "W/D/L", "Điểm", "ms P50", "ms P95", "mp/s"), rows,
              "llrrrrrrr"), encoding="utf-8")
    return summary


def cost_table(records: list[dict], agents: list[str]) -> str:
    rows = []
    for a in agents:
        times = [t for r in records if a in (r["agent_x"], r["agent_o"]) for t in agent_series(r, a, "time_ms")]
        sims = [s for r in records if a in (r["agent_x"], r["agent_o"]) for s in agent_series(r, a, "sims")]
        nodes = [n for r in records if a in (r["agent_x"], r["agent_o"]) for n in agent_series(r, a, "nodes")]
        if not times:
            rows.append((a, "—", "—", "—", "—", "—"))
            continue
        rows.append((a, len(times), dec(np.mean(times), 1), dec(np.median(times), 1),
                     dec(np.quantile(times, .95), 1),
                     f"{np.mean(sims):.0f}" if sims else (f"{np.mean(nodes):.0f}" if nodes else "—")))
    return table(("Tác tử", "Nước đo", "TB (ms)", "P50 (ms)", "P95 (ms)", "TB mô phỏng/nút"), rows, "lrrrrr")


def opening_summary(records: list[dict], agents: list[str]) -> str:
    by_op = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_op[r["opening_id"]][r["agent_x"]].append(outcome_for(r, r["agent_x"]))
        by_op[r["opening_id"]][r["agent_o"]].append(outcome_for(r, r["agent_o"]))
    rows = []
    for a in agents:
        vals = []
        for op, d in by_op.items():
            xs = d.get(a, [])
            if xs:
                vals.append((op, np.mean([0.5 if x == 0 else (1 if x == 1 else 0) for x in xs])))
        if not vals:
            continue
        scores = np.array([v for _, v in vals])
        best = sorted(vals, key=lambda x: -x[1])[:3]
        worst = sorted(vals, key=lambda x: x[1])[:3]
        rows.append((a, len(vals), dec(np.mean(scores), 3), dec(np.std(scores), 3),
                     ", ".join(f"{op}:{pct(v,0)}" for op, v in best),
                     ", ".join(f"{op}:{pct(v,0)}" for op, v in worst)))
    lines = [
        r"\begin{center}\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\textwidth}{@{}lrrr>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X@{}}",
        r"\toprule",
        (r"\textbf{Tác tử} & \textbf{Khai cuộc} & \textbf{Điểm TB} & \textbf{Độ lệch} & "
         r"\textbf{3 tốt nhất} & \textbf{3 xấu nhất} \\"),
        r"\midrule",
    ]
    for a, n, mean, std, best, worst in rows:
        best_s = best.replace("%", r"\%")
        worst_s = worst.replace("%", r"\%")
        lines.append(f"{tex(a)} & {n} & {mean} & {std} & {best_s} & {worst_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{center}"]
    return "\n".join(lines)


def tactical_table(datasets: dict[str, list[dict]]) -> str:
    rows = []
    for label, records in datasets.items():
        if not records:
            continue
        stats = tactical_stats(records, label)["stats"]
        for a in sorted(stats):
            s = stats[a]
            rows.append((label, a, s.get("moves", 0),
                         s.get("win_opp", 0), s.get("win_taken", 0), s.get("win_missed", 0),
                         s.get("block_opp", 0), s.get("block_taken", 0), s.get("block_missed", 0)))
    return table(("Bộ dữ liệu", "Tác tử", "Nước", "Có thắng", "Đã thắng", "Bỏ thắng",
                  "Phải chặn", "Đã chặn", "Bỏ chặn"), rows, "llrrrrrrr")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    summary = {}

    # ---- RR8 mới (nếu đã xong) ----
    rr8 = read_jsonl(RUNS / "eval/tournament/rr8_mode_b/latest/games.jsonl")
    if rr8:
        agents = ["L1-Random", "L2-Rule", "L3-AlphaBeta", "L4-MCTS", "L5-PolicySL",
                  "L6-PolicyRL", "L7-PUCT", "L8-AlphaZero"]
        rr8 = [r for r in rr8 if r["agent_x"] in agents and r["agent_o"] in agents]
        summary["rr8"] = make_rr("rr8", rr8, agents)
        summary["rr8"]["tactics"] = tactical_stats(rr8, "rr8")
        (OUT / "so-lieu-chi-phi.tex").write_text(cost_table(rr8, agents), encoding="utf-8")
        (OUT / "so-lieu-khai-cuoc-moi.tex").write_text(opening_summary(rr8, agents), encoding="utf-8")
        datasets = {"RR8": rr8}
        if (RUNS / "tournament/a3_mcts_sims/20261114-232353/games.jsonl").is_file():
            datasets["A3"] = read_jsonl(RUNS / "tournament/a3_mcts_sims/20261114-232353/games.jsonl")
        if (RUNS / "tournament/a4_rollout/20261114-234556/games.jsonl").is_file():
            datasets["A4"] = read_jsonl(RUNS / "tournament/a4_rollout/20261114-234556/games.jsonl")
        l8v3 = read_jsonl(RUNS / "eval/tournament/l8_vs_l3/20261125-134954_round37/games.jsonl")
        if l8v3:
            datasets["L8-L3"] = l8v3
        (OUT / "so-lieu-chien-thuat.tex").write_text(tactical_table(datasets), encoding="utf-8")

    summary["a3a4"] = make_a3a4()

    # ---- E3 ----
    e3_records = []
    for d in sorted((RUNS / "eval/tournament/e3_selfplay").glob("2026*")):
        e3_records += read_jsonl(d / "games.jsonl")
    if e3_records:
        levels = ["L1-Random", "L2-Rule", "L3-AlphaBeta", "L4-MCTS", "L5-PolicySL",
                  "L6-PolicyRL", "L7-PUCT", "L8-AlphaZero"]
        by = defaultdict(list)
        for r in e3_records:
            by[r["agent_x"]].append(r)
        rows, adv = [], []
        for lvl in levels:
            rs = by.get(lvl, [])
            if not rs:
                continue
            n = len(rs)
            xw = sum(r["result"] == 1 for r in rs)
            ow = sum(r["result"] == -1 for r in rs)
            dr = n - xw - ow
            lo, hi = wilson(xw + .5 * dr, n)
            rows.append((lvl, n, xw, ow, dr, pct((xw + .5 * dr) / n),
                         f"[{pct(lo)}; {pct(hi)}]", f"{(xw - ow) / n:+.3f}".replace(".", ",")))
            adv.append((lvl, 100 * (xw + .5 * dr) / n, 100 * (ow + .5 * dr) / n))
        (OUT / "so-lieu-e3.tex").write_text(
            table(("Cấp", "Ván", "X thắng", "O thắng", "Hoà", "Điểm của X", "Wilson 95%", "Chênh X−O"),
                  rows, "lrrrrrrr"),
            encoding="utf-8")
        fig, ax = plt.subplots(figsize=(7.4, 3.2), layout="constrained")
        xs = np.arange(len(adv))
        ax.bar(xs - .18, [a[1] for a in adv], .36, label="X thắng/hoà", color=C1)
        ax.bar(xs + .18, [a[2] for a in adv], .36, label="O thắng/hoà", color=C2)
        ax.set(xticks=xs, xticklabels=[a[0] for a in adv], ylabel="Điểm (%)", ylim=(0, 100))
        ax.axhline(50, ls="--", color="0.5", lw=.8)
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
        save_fig(fig, "e3")
        summary["e3"] = rows

    # ---- SL mới ----
    hoc = read_json(OUT / "so-lieu-hoc-moi.json")
    if hoc:
        rows = []
        for name, m in hoc["models"].items():
            rows.append((name, pct(m["top1"]), pct(m["top3"]), dec(m["policy_ce"], 3),
                         dec(m["value_relative"]["mse"], 3), f"{m['value_relative']['corr']:+.3f}".replace(".", ","),
                         pct(m["value_relative"]["sign_acc"])))
        text = table(("Mạng", "Top-1", "Top-3", "CE policy", "MSE value", "Tương quan", "Đúng dấu"), rows, "lrrrrrr")
        rows2 = []
        for name, m in hoc["models"].items():
            rows2.append((name, *(f"{m['value_by_phase'][i]['mse']:.3f}".replace(".", ",") for i in range(3)),
                          *(f"{m['value_by_phase'][i]['corr']:+.3f}".replace(".", ",") for i in range(3))))
        text += "\n" + table(("Mạng", "MSE đầu", "MSE giữa", "MSE cuối", "r đầu", "r giữa", "r cuối"), rows2, "lrrrrrr")
        (OUT / "so-lieu-hoc-moi.tex").write_text(text, encoding="utf-8")
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), layout="constrained")
        names = list(hoc["models"])
        axes[0].bar(np.arange(len(names)) - .18, [100 * hoc["models"][n]["top1"] for n in names], .36, label="Top-1", color=C1)
        axes[0].bar(np.arange(len(names)) + .18, [100 * hoc["models"][n]["top3"] for n in names], .36, label="Top-3", color=C3)
        axes[0].set(xticks=np.arange(len(names)), xticklabels=names, ylabel="Độ chính xác (%)", ylim=(0, 100))
        axes[0].legend()
        axes[1].bar(np.arange(len(names)) - .18, [hoc["models"][n]["value_relative"]["mse"] for n in names], .36,
                    label="MSE (góc người đi)", color=C2)
        axes[1].bar(np.arange(len(names)) + .18, [hoc["models"][n]["value_absolute"]["mse"] for n in names], .36,
                    label="MSE (tuyệt đối)", color=C4)
        axes[1].set(xticks=np.arange(len(names)), xticklabels=names, ylabel="MSE value")
        axes[1].legend()
        save_fig(fig, "hoc-moi")
        summary["hoc"] = hoc

    # ---- L6 learning curve ----
    l6_rows = []
    for tag, path in (("RL v2", RUNS / "train/pg_policy/20261117-224555_v2_init/metrics.jsonl"),
                      ("RL v3", RUNS / "train/pg_policy/20261118-125103_v3_init/metrics.jsonl")):
        met = read_jsonl(path)
        ev = [m for m in met if m.get("eval")]
        for m in ev:
            l6_rows.append((tag, m["step"], pct(m.get("L2-Rule_score", float("nan")), 1),
                           pct(m.get("L3-AlphaBeta_score", float("nan")), 1)))
        if ev:
            fig, ax = plt.subplots(figsize=(7, 3.1), layout="constrained")
            ax.plot([m["step"] for m in ev], [100 * m["L2-Rule_score"] for m in ev], "o-", color=C1, label="gặp L2")
            ax.plot([m["step"] for m in ev], [100 * m["L3-AlphaBeta_score"] for m in ev], "s-", color=C2, label="gặp L3")
            ax.set(xlabel="Episode", ylabel="Điểm (%)", ylim=(0, 100))
            ax.axhline(50, ls="--", color="0.5", lw=.8)
            ax.legend()
            save_fig(fig, f"l6-{tag.replace(' ', '')}")
    if l6_rows:
        (OUT / "so-lieu-l6.tex").write_text(
            table(("Mạng", "Episode", "Điểm gặp L2", "Điểm gặp L3"), l6_rows, "lrrr"), encoding="utf-8")

    # ---- L8 training ----
    l8_metrics = [r for r in read_jsonl(RUNS / "train/alphazero/20261124-160615_gpu-native/metrics.jsonl") if "gating" in r]
    if l8_metrics:
        accepted = [r for r in l8_metrics if r["accepted"]]
        rows = [(r["step"], r["gating"]["n"], f"{r['gating']['wins']}/{r['gating']['draws']}/{r['gating']['losses']}",
                 pct(r["gating"]["score"])) for r in accepted]
        (OUT / "so-lieu-l8-moi.tex").write_text(
            table(("Vòng duyệt", "Ván", "W/D/L", "Điểm"), rows, "lrrr"), encoding="utf-8")
        fig, ax = plt.subplots(figsize=(7.4, 3.2), layout="constrained")
        x = [r["step"] for r in l8_metrics]
        ax.plot(x, [100 * r["gating"]["score"] for r in l8_metrics], color=C3, lw=1.2)
        ax.scatter([r["step"] for r in accepted], [100 * r["gating"]["score"] for r in accepted],
                   color=C1, zorder=3, label="được duyệt")
        ax.axhline(55, ls="--", color="0.55")
        ax.set(xlabel="Vòng", ylabel="Điểm gating (%)", ylim=(0, 100))
        ax.legend()
        save_fig(fig, "l8-moi")
        summary["l8"] = {"rounds": len(l8_metrics), "accepted": [r["step"] for r in accepted],
                         "best": max(accepted, key=lambda r: r["gating"]["score"])["step"] if accepted else None}

    # ---- A5 ----
    a5 = read_csv(RUNS / "eval/a5_puct/results.csv")
    if a5:
        (OUT / "so-lieu-a5.tex").write_text(
            table(("Tác tử", "Đối thủ", "Ván", "W/D/L", "Điểm", "Wilson 95%"),
                  [(r["agent"], r["opponent"], r["n"], f"{r['wins']}/{r['draws']}/{r['losses']}",
                    pct(float(r["score"]), 2),
                    f"[{pct(float(r['ci_low']),1)}; {pct(float(r['ci_high']),1)}]") for r in a5],
                  "llrrrr"), encoding="utf-8")

    # ---- Arena lịch sử ----
    arena = read_csv(RUNS / "arena_matrix/games.csv")
    if arena:
        agents = sorted({r["agent_x"] for r in arena} | {r["agent_o"] for r in arena})
        recs = []
        for r in arena:
            recs.append({**r, "result": int(r["result"]), "moves": [0] * int(r["plies"]),
                         "opening_id": int(r["opening_id"]), "agent_x": r["agent_x"], "agent_o": r["agent_o"]})
        summary["arena"] = summarise_tournament(recs, agents, "arena")
        rows = [(r["agent"], r["n"], f"{r['w']}/{r['d']}/{r['l']}", pct(r["score"]),
                 f"[{pct(r['ci_low'])}; {pct(r['ci_high'])}]") for r in summary["arena"]["ranking"]]
        (OUT / "so-lieu-arena-moi.tex").write_text(
            table(("Tác tử", "Ván", "W/D/L", "Điểm", "Wilson 95%"), rows, "lrrrr"), encoding="utf-8")

    (OUT / "so-lieu-tong-hop.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                               encoding="utf-8")
    print("xong; ghi", OUT / "so-lieu-tong-hop.json")


if __name__ == "__main__":
    main()
