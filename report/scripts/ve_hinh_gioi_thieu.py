"""Sinh hình TikZ cho Chương 1 của báo cáo và kiểm tra các con số mà phần thân bài khẳng định.

Mọi thế cờ được engine kiểm tra lại nên hình luôn khớp với code. Chạy từ gốc repo, kết quả ghi vào
report/figures/gioi-thieu/:

    uv run python report/scripts/ve_hinh_gioi_thieu.py
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from caro.agents.base import Limits
from caro.agents.rule_agent import RuleAgent
from caro.engine.bitboard import (
    LINES,
    WIN,
    N,
    candidates,
    has_five,
    index,
    iter_bits,
    popcount,
    rc,
    winning_squares,
)
from caro.engine.game import CaroGame

GOC = Path(__file__).resolve().parents[2]
OUT = GOC / "report" / "figures" / "gioi-thieu"

MAU = r"""\definecolor{gtX}{HTML}{1E293B}
\definecolor{gtO}{HTML}{C81E1E}
\definecolor{gtThang}{HTML}{1B7F4B}
\definecolor{gtVang}{HTML}{FDE68A}
\definecolor{gtNhiet}{HTML}{4A90C8}
"""


def ghi(ten: str, noi_dung: str) -> None:
    (OUT / f"{ten}.tex").write_text(
        "% Sinh bởi report/scripts/ve_hinh_gioi_thieu.py, không sửa tay.\n" + noi_dung, encoding="utf-8"
    )


def bb(o: list[tuple[int, int]]) -> int:
    x = 0
    for r, c in o:
        x |= 1 << index(r, c)
    return x


def luoi(n: int) -> str:
    return rf"\draw[gray!55, line width=.3pt, step=1] (0,0) grid ({n},-{n}); \draw[gray!90] (0,0) rectangle ({n},-{n});" + "\n"


def nhan_truc() -> str:
    s = ""
    for c in range(N):
        s += rf"\node[font=\footnotesize, text=gray] at ({c + .5},.45) {{{chr(65 + c)}}};" + "\n"
    for r in range(N):
        s += rf"\node[font=\footnotesize, text=gray] at (-.55,{-r - .5}) {{{r + 1}}};" + "\n"
    return s


def hinh_van_thang() -> None:
    x = [(5, 5), (4, 4), (6, 6), (3, 3), (7, 7)]
    o = [(2, 2), (8, 8), (3, 7), (7, 3)]
    assert has_five(bb(x)) and not has_five(bb(o))
    assert len(x) == len(o) + 1                    # X đi trước và vừa đi nước thắng
    s = MAU + "\\begin{tikzpicture}[x=.47cm, y=.47cm]\n"
    for r, c in x:
        s += rf"\fill[gtVang] ({c},{-r}) rectangle ++(1,-1);" + "\n"
    s += luoi(N) + nhan_truc()
    for r, c in x:
        s += rf"\node[text=gtX, font=\bfseries\footnotesize] at ({c + .5},{-r - .5}) {{X}};" + "\n"
    for r, c in o:
        s += rf"\node[text=gtO, font=\bfseries\footnotesize] at ({c + .5},{-r - .5}) {{O}};" + "\n"
    ghi("van-thang", s + "\\end{tikzpicture}\n")


def so_cua_so() -> list[list[int]]:
    dem = [[0] * N for _ in range(N)]
    tong = 0
    for d, start in LINES:
        for i in iter_bits(start):
            tong += 1
            for k in range(WIN):
                r, c = rc(i + k * d)
                dem[r][c] += 1
    assert tong == 252
    assert sum(map(sum, dem)) == 252 * WIN
    assert dem[5][5] == 20 and dem[0][5] == 8 and dem[0][0] == 3
    return dem


def hinh_so_cua_so(dem: list[list[int]]) -> None:
    lon, nho = max(map(max, dem)), min(map(min, dem))
    assert (nho, lon) == (3, 20)
    s = MAU + "\\begin{tikzpicture}[x=.47cm, y=.47cm]\n"
    for r in range(N):
        for c in range(N):
            muc = round(70 * (dem[r][c] - nho) / (lon - nho))   # giữ ô đậm nhất vẫn đọc được chữ đen
            s += rf"\fill[gtNhiet!{muc}] ({c},{-r}) rectangle ++(1,-1);" + "\n"
            s += rf"\node[font=\scriptsize] at ({c + .5},{-r - .5}) {{{dem[r][c]}}};" + "\n"
    s += luoi(N) + nhan_truc()
    ghi("so-cua-so", s + "\\end{tikzpicture}\n")


def hinh_mau_the() -> None:
    """Các thế đe doạ; số ô thắng được winning_squares xác nhận."""
    hang = 3
    tu = [(hang, c) for c in range(1, 5)]
    mau = [
        ("Tứ bị chặn", "1 ô thắng:\\\\đối thủ phải chặn", tu, [(hang, 0)], 1),
        ("Tứ mở", "2 ô thắng:\\\\không chặn nổi", tu, [], 2),
        ("Tứ kép", "2 ô thắng:\\\\không chặn nổi",
         tu + [(1, 4), (2, 4), (4, 4)], [(hang, 0), (0, 4)], 2),
    ]
    huong = [
        ("Ngang", 77, [(3, c) for c in range(1, 6)]),
        ("Dọc", 77, [(r, 3) for r in range(1, 6)]),
        ("Chéo xuống phải", 49, [(i, i) for i in range(1, 6)]),
        ("Chéo xuống trái", 49, [(i, 6 - i) for i in range(1, 6)]),
    ]
    s = MAU + "\\begin{tikzpicture}[x=.3cm, y=.3cm]\n"

    def khung(dx: int, dy: int, tren: str, duoi: str, x: list, o: list, thang: list) -> str:
        k = rf"\begin{{scope}}[shift={{({dx},{dy})}}]" + "\n"
        k += rf"\node[anchor=south, font=\footnotesize\bfseries, align=center] at (3.5,.1) {{{tren}}};" + "\n"
        k += rf"\node[anchor=north, font=\footnotesize, align=center] at (3.5,-7.1) {{{duoi}}};" + "\n"
        k += luoi(7)
        for r, c in x:
            k += rf"\fill[gtX] ({c + .5},{-r - .5}) circle (.36);" + "\n"
        for r, c in o:
            k += rf"\fill[gtO] ({c + .5},{-r - .5}) circle (.36);" + "\n"
        for r, c in thang:
            k += rf"\draw[gtThang, line width=1pt] ({c + .5},{-r - .5}) circle (.34);"
            k += rf"\fill[gtThang] ({c + .5},{-r - .5}) circle (.1);" + "\n"
        return k + "\\end{scope}\n"

    for j, (ten, so, o_x) in enumerate(huong):
        assert has_five(bb(o_x))
        s += khung(j * 10, 0, ten, f"{so} cửa sổ", o_x, [], [])
    for j, (ten, ghi_chu, x, o, so_thang) in enumerate(mau):
        w = winning_squares(bb(x), bb(o))
        assert popcount(w) == so_thang, ten
        assert not has_five(bb(x))
        s += khung(4 + j * 11, -13, ten, ghi_chu, x, o, [rc(i) for i in iter_bits(w)])
    ghi("mau-the", s + "\\end{tikzpicture}\n")


def in_so_lieu() -> None:
    """Các con số trong mục Đặc điểm trò chơi."""
    hop_le = sum(
        math.comb(121, 2 * o) * math.comb(2 * o, o) + math.comb(121, 2 * o + 1) * math.comb(2 * o + 1, o)
        for o in range(61)
    )
    print(f"3^121 = {3**121:.3e}; số quân hợp lệ = {hop_le:.3e}; 121! = {math.factorial(121):.3e}")
    the = json.loads((GOC / "runs/openings/midgame_positions_v1.json").read_text())["positions"]
    so_ung_vien = []
    for nuoc in the:
        x = sum(1 << a for a in nuoc[::2])
        o = sum(1 << a for a in nuoc[1::2])
        # bên đến lượt không ảnh hưởng tập ứng viên
        so_ung_vien.append(popcount(candidates(x, o, 2)))
    print(
        f"{len(the)} thế, {min(map(len, the))}-{max(map(len, the))} nước; ứng viên bán kính 2: "
        f"min {min(so_ung_vien)}, trung vị {statistics.median(so_ung_vien)}, max {max(so_ung_vien)}"
    )
    # Trong ván thật quân tụ lại nên ít ứng viên hơn: 40 ván Cấp 2 tự đấu, seed cố định.
    do_dai, ung_vien, dau_van = [], [], []
    for g in range(40):
        van, a, b = CaroGame(), RuleAgent(), RuleAgent()
        a.reset(g)
        b.reset(g + 1000)
        while not van.is_terminal():
            van.step((a if len(van.history) % 2 == 0 else b).select_action(van, Limits()))
            so = popcount(candidates(van.bb[0], van.bb[1], 2))
            ung_vien.append(so)
            if len(van.history) <= 10:
                dau_van.append(so)
        do_dai.append(len(van.history))
    print(
        f"40 ván Cấp 2 tự đấu: {len(ung_vien)} nước, dài {min(do_dai)}-{max(do_dai)} nước; ứng viên: "
        f"trung vị {statistics.median(ung_vien)}, max {max(ung_vien)}; "
        f"10 nước đầu: {min(dau_van)}-{max(dau_van)}, trung vị {statistics.median(dau_van)}"
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    hinh_van_thang()
    hinh_so_cua_so(so_cua_so())
    hinh_mau_the()
    in_so_lieu()
