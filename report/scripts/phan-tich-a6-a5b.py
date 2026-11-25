"""Sinh bảng/hình cho A6 (augmentation) và A5b (khởi tạo RL)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/ban-mo-rong"
FIG = OUT / "hinh"
RUNS = ROOT / "runs"
C1, C2, C3, C4 = "#166d63", "#c26a3d", "#526e88", "#8a6f3e"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pct(x: float) -> str:
    return f"{100 * x:.1f}\\%".replace(".", ",")


def dec(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def table(headers, rows, spec=None) -> str:
    spec = spec or ("l" + "r" * (len(headers) - 1))
    lines = [r"\begin{center}\small", rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(lines)


def score(csv_rows: list[dict], opponent: str) -> float | None:
    for row in csv_rows:
        if row["opponent"] == opponent:
            return float(row["score"])
    return None


def main() -> None:
    hoc = json.loads((OUT / "so-lieu-hoc-moi.json").read_text(encoding="utf-8"))["models"]
    a6 = json.loads((RUNS / "train/a6_noaug/metrics.json").read_text(encoding="utf-8"))
    aug = read_csv(RUNS / "eval/a6_sl_aug.csv")
    noaug = read_csv(RUNS / "eval/a6_sl_noaug.csv")

    rows = []
    for label, key, csv_rows, note in (
        ("SL v3 (aug)", "SL v3", aug, "60 epoch"),
        ("A6 no-aug", "A6 no-aug", noaug, f"dừng sớm epoch {len(a6)}"),
    ):
        m = hoc[key]
        rows.append((label, pct(m["top1"]), pct(m["top3"]), dec(m["value_relative"]["mse"]),
                     f"{m['value_relative']['corr']:+.3f}".replace(".", ","),
                     pct(score(csv_rows, "l2_rule")), pct(score(csv_rows, "l3_alphabeta")), note))
    lines = [r"\begin{center}\scriptsize\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{@{}lrrrrrrl@{}}", r"\toprule",
             r"Mô hình & Top-1 & Top-3 & MSE & $r$ & Vs L2 & Vs L3 & Ghi chú \\", r"\midrule"]
    lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    (OUT / "so-lieu-a6.tex").write_text("\n".join(lines), encoding="utf-8")

    from_sl = read_csv(RUNS / "eval/a5b_from_sl_best.csv")
    from_rand = read_csv(RUNS / "eval/a5b_from_random_best.csv")
    summary = json.loads((RUNS / "train/a5b_from_random/20261126-013405_from_random/summary.json")
                         .read_text(encoding="utf-8"))
    a5b_rows = [
        ("Từ SL (RL v3)", "49.920", pct(score(from_sl, "l1_random")), pct(score(from_sl, "l2_rule")),
         pct(score(from_sl, "l3_alphabeta")), pct(score(from_sl, "l5_policy_sl")), "45,7 phút"),
        ("Từ random", "49.920", pct(score(from_rand, "l1_random")), pct(score(from_rand, "l2_rule")),
         pct(score(from_rand, "l3_alphabeta")), pct(score(from_rand, "l5_policy_sl")),
         f"{summary['train_seconds'] / 60:.1f} phút".replace(".", ",")),
    ]
    (OUT / "so-lieu-a5b.tex").write_text(
        table(["Khởi tạo", "Episode", "L1", "L2", "L3", "L5", "Thời gian"], a5b_rows, "lrrrrrr"),
        encoding="utf-8")

    # Hình A5b: đường học từ random + mốc from-SL
    metrics = [json.loads(line) for line in
               (RUNS / "train/a5b_from_random/20261126-013405_from_random/metrics.jsonl").open(encoding="utf-8")]
    evals = [r for r in metrics if r.get("eval")]
    xs = [r["step"] for r in evals]
    fig, ax = plt.subplots(figsize=(7.4, 3.3), layout="constrained")
    for key, label, color in (("L1-Random_score", "L1", C1), ("L2-Rule_score", "L2", C2),
                              ("L3-AlphaBeta_score", "L3", C3), ("L5-PolicySL_score", "L5", C4)):
        ax.plot(xs, [100 * r[key] for r in evals], "o-", color=color, label=f"from random · {label}")
    opp = {"L2-Rule_score": "l2_rule", "L3-AlphaBeta_score": "l3_alphabeta", "L5-PolicySL_score": "l5_policy_sl"}
    for key, label, color in (("L2-Rule_score", "L2", C2), ("L3-AlphaBeta_score", "L3", C3),
                              ("L5-PolicySL_score", "L5", C4)):
        ax.axhline(100 * float(next(r["score"] for r in from_sl if r["opponent"] == opp[key])),
                   ls="--", color=color, lw=1, alpha=.7)
    ax.set(xlabel="Episode", ylabel="Điểm (%)", ylim=(0, 100))
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIG / "a5b.png", dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("xong")


if __name__ == "__main__":
    main()
