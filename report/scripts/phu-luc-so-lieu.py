"""Sinh phụ lục bảng số liệu đầy đủ cho RR8 và E3."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/ban-mo-rong"
RUNS = ROOT / "runs"

AGENTS = ["L1-Random", "L2-Rule", "L3-AlphaBeta", "L4-MCTS", "L5-PolicySL",
          "L6-PolicyRL", "L7-PUCT", "L8-AlphaZero"]


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def coord(m: int) -> str:
    return f"{chr(65 + m % 11)}{m // 11 + 1}"


def outcome(r: dict, a: str) -> int:
    mine = 1 if r["agent_x"] == a else -1
    if r["result"] == 0:
        return 0
    return 1 if r["result"] == mine else -1


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return c - h, c + h


def pct(x: float) -> str:
    return f"{100 * x:.1f}\\%".replace(".", ",")


def main() -> None:
    rr = rows(RUNS / "eval/tournament/rr8_mode_b/latest/games.jsonl")
    rr = [r for r in rr if r["agent_x"] in AGENTS and r["agent_o"] in AGENTS]
    e3 = []
    for d in sorted((RUNS / "eval/tournament/e3_selfplay").glob("2026*")):
        e3 += rows(d / "games.jsonl")

    sec = [r"\chapter{Phụ lục bảng số liệu đầy đủ}", r"\label{app:so-lieu}",
           ("Các bảng ở đây là dữ liệu thô đã tổng hợp, dùng để kiểm tra các bảng tóm tắt trong "
            "Chương~4. Mỗi dòng là một cặp đấu hoặc một khai cuộc; không có bảng nào dùng số "
            "liệu từ nguồn khác ngoài log trong \\repoFile{runs/eval/tournament/}.")]

    # 1. Từng cặp đấu (X vs O)
    sec += [r"\section{Điểm từng cặp đấu}", r"\begin{small}",
            r"\begin{longtable}{@{}llrrrr@{}}", r"\toprule",
            r"X & O & Ván & W/D/L của X & Điểm X & Điểm O \\", r"\midrule"]
    for a in AGENTS:
        for b in AGENTS:
            if a == b:
                continue
            sub = [r for r in rr if r["agent_x"] == a and r["agent_o"] == b]
            if not sub:
                continue
            w = sum(r["result"] == 1 for r in sub)
            d = sum(r["result"] == 0 for r in sub)
            l = len(sub) - w - d
            sec.append(f"{a} & {b} & {len(sub)} & {w}/{d}/{l} & {pct((w + .5 * d) / len(sub))} & {pct((l + .5 * d) / len(sub))} \\\\")
    sec += [r"\bottomrule", r"\end{longtable}", r"\end{small}"]

    # 2. Theo màu
    sec += [r"\section{Điểm theo màu}", r"\begin{small}",
            r"\begin{longtable}{@{}lrrrrrr@{}}", r"\toprule",
            r"Tác tử & Ván X & W/D/L X & Điểm X & Ván O & W/D/L O & Điểm O \\", r"\midrule"]
    for a in AGENTS:
        xs = [r for r in rr if r["agent_x"] == a]
        os_ = [r for r in rr if r["agent_o"] == a]
        def line(sub):
            w = sum(outcome(r, a) == 1 for r in sub)
            d = sum(outcome(r, a) == 0 for r in sub)
            l = len(sub) - w - d
            return len(sub), f"{w}/{d}/{l}", pct((w + .5 * d) / len(sub)) if sub else "—"
        nx, wx, sx = line(xs)
        no, wo, so = line(os_)
        sec.append(f"{a} & {nx} & {wx} & {sx} & {no} & {wo} & {so} \\\\")
    sec += [r"\bottomrule", r"\end{longtable}", r"\end{small}"]

    # 3. Theo khai cuộc cho cả 8 tác tử
    focus = AGENTS
    short = {a: f"L{i + 1}" for i, a in enumerate(AGENTS)}
    by_op = defaultdict(lambda: defaultdict(list))
    moves_at = {}
    for r in rr:
        by_op[r["opening_id"]][r["agent_x"]].append(outcome(r, r["agent_x"]))
        by_op[r["opening_id"]][r["agent_o"]].append(outcome(r, r["agent_o"]))
        moves_at[r["opening_id"]] = r["moves"][: max(0, len(r["moves"]) - len(r.get("time_ms") or []))]
    sec += [r"\section{Điểm theo khai cuộc}", r"\begin{small}\scriptsize",
            r"\begin{longtable}{@{}rl" + "r" * len(focus) + r"@{}}", r"\toprule",
            r"KC & Chuỗi mở đầu & " + " & ".join(short[a] for a in focus) + r" \\", r"\midrule"]
    for op in sorted(by_op):
        cells = []
        for a in focus:
            vals = by_op[op].get(a, [])
            if not vals:
                cells.append("—")
            else:
                cells.append(pct(np.mean([0.5 if v == 0 else (1 if v == 1 else 0) for v in vals])))
        sec.append(f"{op:02d} & " + " ".join(coord(m) for m in moves_at[op]) + " & " + " & ".join(cells) + r" \\")
    sec += [r"\bottomrule", r"\end{longtable}", r"\end{small}"]

    # 4. Thời gian
    sec += [r"\section{Thời gian mỗi nước}", r"\begin{small}",
            r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
            r"Tác tử & Nước & P50 (ms) & P95 (ms) \\", r"\midrule"]
    for a in AGENTS:
        times = []
        for r in rr:
            if a not in (r["agent_x"], r["agent_o"]):
                continue
            vals = r.get("time_ms") or []
            if not vals:
                continue
            first_x = (len(r["moves"]) - len(vals)) % 2 == 0
            start = 0 if first_x == (r["agent_x"] == a) else 1
            times += vals[start::2]
        if times:
            sec.append(f"{a} & {len(times)} & {np.median(times):.1f} & {np.quantile(times, .95):.1f} \\\\".replace(".", ","))
    sec += [r"\bottomrule", r"\end{tabular}", r"\end{small}"]

    # 5. E3 chi tiết
    sec += [r"\section{Tự đấu E3}", r"\begin{small}",
            r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
            r"Cấp & X thắng & O thắng & Hoà & Điểm X & Wilson 95\% \\", r"\midrule"]
    for a in AGENTS:
        sub = [r for r in e3 if r["agent_x"] == a]
        if not sub:
            continue
        xw = sum(r["result"] == 1 for r in sub)
        ow = sum(r["result"] == -1 for r in sub)
        dr = len(sub) - xw - ow
        lo, hi = wilson(xw + .5 * dr, len(sub))
        sec.append(f"{a} & {xw} & {ow} & {dr} & {pct((xw + .5 * dr) / len(sub))} & [{pct(lo)}; {pct(hi)}] \\\\")
    sec += [r"\bottomrule", r"\end{tabular}", r"\end{small}",
            r"\section{Ghi chú}",
            ("Các bảng dùng điểm $(W+0.5D)/n$. Với cặp đấu, cột ``Điểm O'' tính từ góc nhìn O; "
             "tổng điểm X và O trong một cặp luôn bằng 1. Với bảng khai cuộc, mỗi ô dựa trên hai "
             "ván của cùng khai cuộc (một X, một O) cho tác tử đó. Wilson 95\\% chỉ dùng để mô tả "
             "độ bất định, không dùng để so sánh nhiều cặp.")]

    (OUT / "phu-luc-so-lieu.tex").write_text("\n".join(sec), encoding="utf-8")
    print("ghi", OUT / "phu-luc-so-lieu.tex")


if __name__ == "__main__":
    main()
