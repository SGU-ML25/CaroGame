"""Sinh bảng/hình cho giải chính thức L1–L6 trên nhánh main (mode A, mode B, E3)."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/ban-mo-rong"
FIG = OUT / "hinh"
RUNS = {
    "A": ROOT / "runs/tournament/main_mode_a_l1_l6/20261124-212727/games.jsonl",
    "B": ROOT / "runs/tournament/main_mode_b_l1_l6/20261124-222606/games.jsonl",
    "E3": ROOT / "runs/tournament/e3_first_move_l1_l6/20261124-230633/games.jsonl",
}
AGENTS = ["L1-Random", "L2-Rule", "L3-AlphaBeta", "L4-MCTS", "L5-PolicySL", "L6-PolicyRL"]
C1, C2, C3 = "#166d63", "#c26a3d", "#526e88"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


def pct(x: float) -> str:
    return f"{100 * x:.1f}\\%".replace(".", ",")


def table(headers, data, spec=None) -> str:
    spec = spec or ("l" + "r" * (len(headers) - 1))
    lines = [r"\begin{center}\small", rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(map(str, row)) + r" \\" for row in data]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(lines)


def outcome(r: dict, a: str) -> int:
    mine = 1 if r["agent_x"] == a else -1
    if r["result"] == 0:
        return 0
    return 1 if r["result"] == mine else -1


def wdl(records: list[dict], a: str) -> tuple[int, int, int]:
    c = Counter(outcome(r, a) for r in records)
    return c[1], c[0], c[-1]


def ranking(records: list[dict], agents: list[str]):
    out = []
    for a in agents:
        sub = [r for r in records if a in (r["agent_x"], r["agent_o"])]
        w, d, l = wdl(sub, a)
        n = len(sub)
        lo, hi = wilson(w + .5 * d, n)
        out.append((a, n, w, d, l, (w + .5 * d) / n, lo, hi))
    return sorted(out, key=lambda x: -x[5])


def matrix(records: list[dict], agents: list[str]) -> str:
    lines = [r"\begin{center}\scriptsize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}l" + "c" * len(agents) + r"@{}}", r"\toprule",
             " & ".join([""] + agents) + r" \\", r"\midrule"]
    for a in agents:
        cells = []
        for b in agents:
            if a == b:
                cells.append("—")
                continue
            sub = [r for r in records if {r["agent_x"], r["agent_o"]} == {a, b}]
            w, d, _ = wdl(sub, a)
            cells.append(f"{100 * (w + .5 * d) / len(sub):.0f}" if sub else "·")
        lines.append(a + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(lines)


def summary_table(records: list[dict], agents: list[str]) -> str:
    data = [(a, n, f"{w}/{d}/{l}", pct(s), f"[{pct(lo)}; {pct(hi)}]")
            for a, n, w, d, l, s, lo, hi in ranking(records, agents)]
    return table(["Tác tử", "Ván", "W/D/L", "Điểm", "Wilson 95\\%"], data, "lrrrr")


def color_table(records: list[dict], agents: list[str]) -> str:
    data = []
    for a in agents:
        xs = [r for r in records if r["agent_x"] == a]
        os_ = [r for r in records if r["agent_o"] == a]
        def line(sub, a=a):
            w, d, l = wdl(sub, a)
            return len(sub), f"{w}/{d}/{l}", pct((w + .5 * d) / len(sub))
        nx, wx, sx = line(xs)
        no, wo, so = line(os_)
        data.append((a, nx, wx, sx, no, wo, so))
    return table(["Tác tử", "Ván X", "W/D/L X", "Điểm X", "Ván O", "W/D/L O", "Điểm O"],
                 data, "lrrrrrr")


def main() -> None:
    data = {k: rows(p) for k, p in RUNS.items()}

    for key, tag in (("A", "main-a"), ("B", "main-b")):
        recs = data[key]
        (OUT / f"so-lieu-{tag}.tex").write_text(
            summary_table(recs, AGENTS) + "\n" + matrix(recs, AGENTS), encoding="utf-8")
        (OUT / f"so-lieu-{tag}-color.tex").write_text(color_table(recs, AGENTS), encoding="utf-8")

    # So sánh A/B
    ra = {r[0]: r for r in ranking(data["A"], AGENTS)}
    rb = {r[0]: r for r in ranking(data["B"], AGENTS)}
    ab = []
    for a in AGENTS:
        sa, sb = ra[a][5], rb[a][5]
        ab.append((a, pct(sa), pct(sb), f"{(sb - sa) * 100:+.1f}".replace(".", ",")))
    (OUT / "so-lieu-main-ab.tex").write_text(
        table(["Tác tử", "Điểm mode A", "Điểm mode B", "B − A (đpt)"], ab, "lrrr"),
        encoding="utf-8")

    # E3 chính thức
    e3 = data["E3"]
    e3rows, adv = [], []
    for a in AGENTS:
        sub = [r for r in e3 if r["agent_x"] == a]
        xw = sum(r["result"] == 1 for r in sub)
        ow = sum(r["result"] == -1 for r in sub)
        dr = len(sub) - xw - ow
        lo, hi = wilson(xw + .5 * dr, len(sub))
        e3rows.append((a, len(sub), xw, ow, dr, pct((xw + .5 * dr) / len(sub)),
                       f"[{pct(lo)}; {pct(hi)}]"))
        adv.append((a, 100 * (xw + .5 * dr) / len(sub), 100 * (ow + .5 * dr) / len(sub)))
    (OUT / "so-lieu-e3-official.tex").write_text(
        table(["Cấp", "Ván", "X thắng", "O thắng", "Hoà", "Điểm X", "Wilson 95\\%"],
              e3rows, "lrrrrrr"), encoding="utf-8")

    # Hình A/B
    fig, ax = plt.subplots(figsize=(7.4, 3.2), layout="constrained")
    xs = np.arange(len(AGENTS))
    ax.bar(xs - .2, [100 * ra[a][5] for a in AGENTS], .4, label="Mode A", color=C1)
    ax.bar(xs + .2, [100 * rb[a][5] for a in AGENTS], .4, label="Mode B", color=C2)
    ax.set(xticks=xs, xticklabels=AGENTS, ylabel="Điểm (%)", ylim=(0, 100))
    ax.tick_params(axis="x", rotation=20)
    ax.axhline(50, ls="--", color="0.5", lw=.8)
    ax.legend()
    fig.savefig(FIG / "main-ab.png", dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # Hình E3
    fig, ax = plt.subplots(figsize=(7.4, 3.2), layout="constrained")
    xs = np.arange(len(adv))
    ax.bar(xs - .18, [a[1] for a in adv], .36, label="X thắng/hoà", color=C1)
    ax.bar(xs + .18, [a[2] for a in adv], .36, label="O thắng/hoà", color=C2)
    ax.set(xticks=xs, xticklabels=[a[0] for a in adv], ylabel="Điểm (%)", ylim=(0, 100))
    ax.tick_params(axis="x", rotation=20)
    ax.axhline(50, ls="--", color="0.5", lw=.8)
    ax.legend()
    fig.savefig(FIG / "e3-official.png", dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    (OUT / "so-lieu-main.json").write_text(json.dumps({
        "A": {r[0]: {"score": r[5], "w": r[2], "d": r[3], "l": r[4]} for r in ranking(data["A"], AGENTS)},
        "B": {r[0]: {"score": r[5], "w": r[2], "d": r[3], "l": r[4]} for r in ranking(data["B"], AGENTS)},
        "E3": {a[0]: {"x": a[1], "o": a[2]} for a in adv},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("xong")


if __name__ == "__main__":
    main()
