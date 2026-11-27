"""Phần dùng chung của các script phân tích chuyên sâu: đọc log, dựng bảng LaTeX, bàn cờ TikZ.

Chạy mọi script từ thư mục gốc dự án, ví dụ:
    .venv/bin/python report/scripts/chuyen_sau/phan_tich_van.py
    uv run --no-sync --with matplotlib python report/scripts/chuyen_sau/ve_hinh.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from caro.engine.bitboard import CELLS, N  # noqa: E402

OUT = ROOT / "report" / "chuyen-sau"
DATA = OUT / "data"
TABLES = OUT / "bang"
FIGURES = OUT / "hinh"
for folder in (DATA, TABLES, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 2026
AGENT_ORDER = ("L1-Random", "L2-Rule", "L3-AlphaBeta", "L4-MCTS", "L5-PolicySL", "L6-PolicyRL",
               "L7-PUCT", "L8-AlphaZero")
SHORT = {name: name.split("-")[0] for name in AGENT_ORDER}

RUNS = {
    "rr8": "runs/eval/tournament/rr8_heldout/20261126-194315",
    "l8_sims": "runs/eval/tournament/l8_sims_heldout/20261126-230159",    # dừng sau game_id 399
    "cpuct": "runs/eval/tournament/cpuct_heldout/latest",
}


def games(path: str | Path) -> list[dict]:
    path = ROOT / path
    if path.is_dir():
        path = path / "games.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(name: str, value) -> None:
    (DATA / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


def load_json(name: str):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def cell_name(i: int) -> str:
    """Ô 0..120 → 'F6': cột A–K, hàng 1–11 từ trên xuống (cùng quy ước chương 4)."""
    r, c = divmod(i, N)
    return f"{'ABCDEFGHIJK'[c]}{r + 1}"


def num(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def count(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", ".")


def percent(part: float, whole: float, digits: int = 1) -> str:
    return "---" if not whole else num(100 * part / whole, digits)


def body(row: list[str]) -> str:
    """Một hàng; ô đầu bắt đầu bằng "[" được bọc lại để không bị đọc thành tham số tuỳ chọn của \\\\ hay \\midrule."""
    if not row:
        return r"    \midrule"
    return "    " + ("{}" if row[0].startswith("[") else "") + " & ".join(row) + r" \\"


def table(name: str, caption: str, spec: str, header: list[str], rows: list[list[str]],
          size: str = r"\small", note: str = "") -> None:
    """Bảng nổi cố định vị trí, ghi vào report/chuyen-sau/bang/<name>.tex."""
    lines = [r"\begin{table}[H]", r"  \centering", f"  {size}", rf"  \caption{{{caption}}}",
             rf"  \label{{tab:cs-{name}}}", rf"  \begin{{tabular}}{{{spec}}}", r"    \toprule",
             "    " + " & ".join(header) + r" \\", r"    \midrule"]
    lines += [body(row) for row in rows]
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    if note:
        lines.append(rf"  \par\smallskip\parbox{{.95\linewidth}}{{\footnotesize {note}}}")
    lines += [r"\end{table}", ""]
    (TABLES / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def longtable(name: str, caption: str, spec: str, header: list[str], rows: list[list[str]],
              size: str = r"\footnotesize") -> None:
    head = "    " + " & ".join(header) + r" \\"
    lines = ["{" + size, r"\renewcommand{\arraystretch}{0.95}",
             rf"\begin{{longtable}}{{{spec}}}",
             rf"  \caption{{{caption}}}\label{{tab:cs-{name}}}\\", r"    \toprule", head, r"    \midrule",
             r"  \endfirsthead", r"    \toprule", head, r"    \midrule", r"  \endhead",
             r"    \bottomrule", r"  \endfoot"]
    lines += [body(row) for row in rows]
    lines += [r"\end{longtable}", "}", ""]
    (TABLES / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def board_tikz(x: int, o: int, numbers: dict[int, int] | None = None, marks: dict[int, str] | None = None,
               heat: dict[int, float] | None = None, unit: float = 0.5) -> str:
    """Bàn cờ TikZ. numbers: ô → số thứ tự nước; marks: ô → màu tô nền; heat: ô → cường độ 0..1."""
    numbers, marks, heat = numbers or {}, marks or {}, heat or {}
    out = [rf"\begin{{tikzpicture}}[x={unit}cm, y={unit}cm]"]
    for cell, level in heat.items():
        r, c = divmod(cell, N)
        out.append(rf"\fill[blue!{max(1, round(100 * level * 0.75))}] ({c},{-r}) rectangle ({c + 1},{-r - 1});")
    for cell, color in marks.items():
        r, c = divmod(cell, N)
        out.append(rf"\fill[{color}] ({c},{-r}) rectangle ({c + 1},{-r - 1});")
    out.append(rf"\draw[gray!55, line width=.3pt, step=1] (0,0) grid ({N},{-N});")
    out.append(rf"\draw[gray!90] (0,0) rectangle ({N},{-N});")
    for c in range(N):
        out.append(rf"\node[font=\tiny, anchor=south] at ({c + 0.5},0) {{{'ABCDEFGHIJK'[c]}}};")
    for r in range(N):
        out.append(rf"\node[font=\tiny, anchor=east] at (0,{-r - 0.5}) {{{r + 1}}};")
    for cell in range(CELLS):
        is_x, is_o = x >> cell & 1, o >> cell & 1
        if not (is_x or is_o):
            continue
        r, c = divmod(cell, N)
        center = f"({c + 0.5},{-r - 0.5})"
        fill, text = ("black!85", "white") if is_x else ("white", "black")
        out.append(rf"\draw[fill={fill}, line width=.4pt] {center} circle (.40);")
        if cell in numbers:
            out.append(rf"\node[font=\tiny, text={text}, inner sep=0pt] at {center} {{{numbers[cell]}}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out)


def replay(moves: list[int], upto: int | None = None) -> tuple[int, int]:
    """(bitboard X, bitboard O) sau `upto` nước đầu."""
    bb = [0, 0]
    for k, move in enumerate(moves[:upto]):
        bb[k % 2] |= 1 << move
    return bb[0], bb[1]
