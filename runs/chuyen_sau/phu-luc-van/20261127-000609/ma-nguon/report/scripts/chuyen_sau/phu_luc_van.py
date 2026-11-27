"""Sinh phụ lục ván đại diện: mỗi cặp của RR8 giữ riêng một ván, chọn theo quy tắc cố định.

    .venv/bin/python report/scripts/chuyen_sau/phu_luc_van.py

Quy tắc chọn: trong các ván có thắng thua của cặp (nếu không có thì mọi ván), sắp theo (số nước, mã ván) và lấy
ván ở vị trí giữa. Không nhìn vào nội dung ván khi chọn. Chú giải sinh từ nhãn của phan_tich_van và bộ giải VCF.
"""

from __future__ import annotations

import itertools
import statistics

import chung
import vcf
from chung import AGENT_ORDER, SHORT, cell_name, num

from caro.engine.bitboard import DIRECTIONS, N, iter_bits, winning_squares

QUIET = ("safe_kept",)
MAX_EVENTS = 8


def representative(records: list[dict], a: str, b: str) -> dict:
    games = [r for r in records if {r["agent_x"], r["agent_o"]} == {a, b}]
    decisive = [r for r in games if r["result"] != 0] or games
    decisive.sort(key=lambda r: (len(r["moves"]), r["game_id"]))
    return decisive[(len(decisive) - 1) // 2]


def winning_line(board: int, last: int) -> list[int]:
    r0, c0 = divmod(last, N)
    for dr, dc in DIRECTIONS:
        cells = [last]
        for sign in (1, -1):
            r, c = r0 + sign * dr, c0 + sign * dc
            while 0 <= r < N and 0 <= c < N and board >> (r * N + c) & 1:
                cells.append(r * N + c)
                r, c = r + sign * dr, c + sign * dc
        if len(cells) >= 5:
            return cells
    return []


def cells(mask: int) -> str:
    return ", ".join(cell_name(c) for c in iter_bits(mask))


def line_text(line: tuple[int, ...]) -> str:
    return "--".join(cell_name(c) for c in line)


def describe(label: str, me: int, opp: int, move: int) -> str:
    threats = winning_squares(opp, me)
    if label == "win1_taken":
        return "thắng ngay"
    if label == "win1_missed":
        return f"bỏ lỡ nước thắng ngay tại {cells(winning_squares(me, opp))}"
    if label == "block_done":
        return f"chặn ô thắng {cells(threats)}"
    if label == "block_missed":
        return f"không chặn ô thắng {cells(threats)}"
    if label == "lost":
        return f"đối thủ đang có các ô thắng {cells(threats)}, không đỡ hết được"
    if label == "vcf_kept":
        line = vcf.solve(me, opp)
        return f"đang có VCF {vcf.attacks_in(line)} nước và đi đúng một đường ({line_text(line)} là đường ngắn nhất)"
    if label == "vcf_missed":
        line = vcf.solve(me, opp)
        return f"bỏ lỡ VCF {vcf.attacks_in(line)} nước: {line_text(line)}"
    if label == "open_blunder":
        line = vcf.solve(opp, me | 1 << move)
        return f"để đối thủ có VCF: {line_text(line)}" if line else "để đối thủ có VCF"
    if label == "open_forced":
        return "không còn nước nào ngăn được VCF của đối thủ"
    return label


def move_table(record: dict) -> str:
    start, moves = record["opening_plies"], record["moves"]
    entries = [f"{i + 1}.~{'X' if i % 2 == 0 else 'O'} {cell_name(m)}" + (r"$^\ast$" if i < start else "")
               for i, m in enumerate(moves)]
    columns = 6
    rows = (len(entries) + columns - 1) // columns
    lines = [r"{\footnotesize\linespread{1}\selectfont", r"\noindent\begin{tabularx}{\linewidth}{*{6}{X}}",
             r"\toprule"]
    for r in range(rows):
        lines.append(" & ".join(entries[r + k * rows] if r + k * rows < len(entries) else ""
                                for k in range(columns)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}\par}"]
    return "\n".join(lines)


def section(record: dict, labels: dict, index: int, pair: tuple[str, str], summary: str) -> str:
    start, moves, result = record["opening_plies"], record["moves"], record["result"]
    bb, events = [0, 0], []
    for k, m in enumerate(moves[:start]):
        bb[k % 2] |= 1 << m
    key_ply = None
    for k, label in enumerate(labels["labels"]):
        ply = start + k
        side = ply % 2
        me, opp = bb[side], bb[side ^ 1]
        if label not in QUIET:
            events.append((ply, side, label, describe(label, me, opp, moves[ply])))
            if key_ply is None and label.startswith(("vcf_", "win1_", "open_blunder")):
                key_ply = ply
        bb[side] |= 1 << moves[ply]
    if key_ply is None:
        key_ply = (start + len(moves)) // 2

    x_name, o_name = record["agent_x"], record["agent_o"]
    outcome = {1: f"X ({SHORT[x_name]}) thắng", -1: f"O ({SHORT[o_name]}) thắng", 0: "hoà"}[result]
    title = f"{SHORT[pair[0]]} -- {SHORT[pair[1]]}: ván {record['game_id']}"
    out = [rf"\section{{{title}}}", rf"\label{{sec:cs-van-{index:02d}}}",
           rf"X: \texttt{{{x_name}}}; O: \texttt{{{o_name}}}; khai cuộc {record['opening_id']} "
           rf"({start} nước, đánh dấu $\ast$); {len(moves)} nước; kết quả: {outcome}. {summary}"]

    x_key, o_key = chung.replay(moves, key_ply)
    x_end, o_end = chung.replay(moves)
    final_marks = {}
    if result:
        winner_board = x_end if result == 1 else o_end
        final_marks = {c: "green!35" for c in winning_line(winner_board, moves[-1])}
    numbers = {m: i + 1 for i, m in enumerate(moves)}
    mover = "X" if key_ply % 2 == 0 else "O"
    out += [r"\begin{figure}[H]", r"  \centering", r"  \begin{subfigure}[t]{.48\linewidth}", r"    \centering",
            chung.board_tikz(x_key, o_key, numbers, {moves[key_ply]: "red!25"}, unit=0.40),
            rf"    \caption{{Trước nước {key_ply + 1} ({mover} đi; ô đỏ là nước đã chọn)}}",
            r"  \end{subfigure}\hfill", r"  \begin{subfigure}[t]{.48\linewidth}", r"    \centering",
            chung.board_tikz(x_end, o_end, numbers, final_marks, unit=0.40),
            r"    \caption{Thế cuối" + (" (ô xanh: dãy thắng)" if result else "") + "}",
            r"  \end{subfigure}",
            rf"  \caption{{Ván {record['game_id']} ({SHORT[x_name]} cầm X, {SHORT[o_name]} cầm O).}}",
            rf"  \label{{fig:cs-van-{index:02d}}}", r"\end{figure}", move_table(record)]

    out.append(r"\paragraph{Diễn biến chiến thuật.}")
    if not events:
        out.append("Không nước nào rơi vào các tình huống chiến thuật được gắn nhãn.")
    else:
        shown = events if len(events) <= MAX_EVENTS else events[:3] + events[-(MAX_EVENTS - 3):]
        out += [r"\begin{itemize}\footnotesize\linespread{1.1}\selectfont", r"\setlength{\itemsep}{0pt}"]
        for i, (ply, side, _, text) in enumerate(shown):
            if len(events) > MAX_EVENTS and i == 3:
                out.append(rf"  \item[] \emph{{\dots\ lược {len(events) - MAX_EVENTS} sự kiện giữa ván\dots}}")
            name = SHORT[x_name if side == 0 else o_name]
            out.append(rf"  \item Nước {ply + 1} ({'XO'[side]}={name}, {cell_name(moves[ply])}): {text}.")
        out.append(r"\end{itemize}")
    times = [record["time_ms"][0::2], record["time_ms"][1::2]]
    if start % 2:
        times.reverse()
    out.append(rf"Thời gian trung vị mỗi nước: X {num(statistics.median(times[0] or [0]), 1)} ms, O "
               rf"{num(statistics.median(times[1] or [0]), 1)} ms.")
    out.append(r"\clearpage")
    return "\n".join(out)


def main() -> None:
    records = chung.games(chung.RUNS["rr8"])
    labels = {g["game_id"]: g for g in chung.load_json("phan-tich-van")["games"]}
    pairs = {(p["a"], p["b"]): p for p in chung.load_json("phan-tich-thong-ke")["pairs"]}
    parts = []
    for index, (a, b) in enumerate(itertools.combinations(AGENT_ORDER, 2), 1):
        record = representative(records, a, b)
        p = pairs[a, b]
        summary = (f"Cả cặp: {SHORT[a]} thắng {p['w']}, hoà {p['d']}, thua {p['l']} trên {p['n']} ván; trung vị "
                   f"{num(p['length_median'], 0)} nước.")
        parts.append(section(record, labels[record["game_id"]], index, (a, b), summary))
    (chung.TABLES / "pl-van.tex").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(len(parts), "ván")


if __name__ == "__main__":
    main()
