"""Trích đoạn mã thật từ kho vào report/chuyen-sau/ma/ để báo cáo \\lstinputlisting.

    .venv/bin/python report/scripts/chuyen_sau/trich_ma.py

Mỗi đoạn lấy theo tên hàm / lớp (không theo số dòng) nên vẫn đúng khi mã dịch chỗ.
"""

from __future__ import annotations

import ast
import textwrap

import chung

CODE = chung.OUT / "ma"
# Ký tự ngoài bảng của listings: thay bằng ASCII để không đảo thứ tự khi in.
ASCII = str.maketrans({"→": "->", "←": "<-", "↔": "<->", "≥": ">=", "≤": "<=", "…": "...", "—": "-", "–": "-",
                       "−": "-", "↘": "\\", "↙": "/", "∈": " in ", "∞": "inf", "Δ": "d", "θ": "theta",
                       "π": "pi", "τ": "tau", "λ": "lambda", "β": "beta", "∝": "~", "√": "sqrt", "·": "*",
                       "×": "x", "“": '"', "”": '"', "’": "'", "≠": "!=", "∑": "sum", "α": "alpha"})

SNIPPETS: dict[str, tuple[str, list[str]]] = {
    "puct-chon": ("caro/agents/puct.py", ["select_child"]),
    "sl-chia": ("caro/training/train_sl.py", ["mover_relative_values", "split_groups"]),
}


def extract(path: str, names: list[str]) -> str:
    source = (chung.ROOT / path).read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    parts = []
    for name in names:
        scope, node = tree.body, None
        for part in name.split("."):
            node = next(n for n in scope if isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name == part)
            scope = node.body
        start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
        parts.append(textwrap.dedent("\n".join(lines[start - 1:node.end_lineno])))
    return "\n\n\n".join(parts) + "\n"


def main() -> None:
    CODE.mkdir(parents=True, exist_ok=True)
    files = []
    for name, (path, symbols) in SNIPPETS.items():
        files.append(CODE / f"{name}.py")
        files[-1].write_text(extract(path, symbols).translate(ASCII), encoding="utf-8")
    print(len(files), "đoạn,", sum(len(f.read_text(encoding="utf-8").splitlines()) for f in files), "dòng")
    odd = sorted({c for f in files for c in f.read_text(encoding="utf-8")
                  if ord(c) > 255 and not (0x1EA0 <= ord(c) <= 0x1EF9) and c not in "ĂăĐđĨĩŨũƠơƯư"})
    print("ký tự ngoài bảng:", odd)


if __name__ == "__main__":
    main()
