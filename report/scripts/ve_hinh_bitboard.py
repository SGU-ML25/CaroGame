"""Sinh hình TikZ và bảng cho mục "Engine bitboard" của báo cáo (chương 3).

Mọi số liệu lấy từ chính caro/engine nên hình luôn khớp với code; các assert dưới đây kiểm tra
những điều phần thân bài khẳng định. Chạy từ gốc repo, kết quả ghi vào report/figures/bitboard/:

    uv run python report/scripts/ve_hinh_bitboard.py
"""

from __future__ import annotations

from pathlib import Path

from caro.engine.bitboard import (
    CELLS,
    DIRECTIONS,
    FULL,
    LINES,
    NOT_COL0,
    NOT_COL10,
    WEIGHT,
    WIN,
    N,
    bit,
    candidates,
    dilate,
    evaluate,
    exactly,
    has_five,
    index,
    inside,
    iter_bits,
    popcount,
    rc,
    side_score,
    window_counts,
    winning_squares,
)
from caro.engine.game import CaroGame
from caro.engine.symmetry import COUNT, PERMS, canonical_key, transform_action, transform_bb
from caro.engine.zobrist import SIDE_KEY, ZOBRIST, full_key

OUT = Path(__file__).resolve().parents[1] / "figures" / "bitboard"

MAU = r"""\definecolor{bbX}{HTML}{1F4E8C}
\definecolor{bbO}{HTML}{C8553D}
\definecolor{bbKq}{HTML}{2E8B57}
\definecolor{bbSai}{HTML}{D7263D}
\definecolor{bbMat}{HTML}{DCE8F5}
\definecolor{bbVang}{HTML}{FBE7A1}
\definecolor{bbXam}{HTML}{BDBDBD}
\definecolor{bbNhat}{HTML}{9CC3E6}
"""

# kiểu ô: (màu nền, màu chữ)
KIEU = {
    "X": ("bbX", "white"),
    "O": ("bbO", "white"),
    "kq": ("bbKq", "white"),
    "sai": ("bbSai", "white"),
    "mat": ("bbMat", "black"),
    "vang": ("bbVang", "black"),
    "xam": ("bbXam", "black"),
    "nhat": ("bbNhat", "black"),
}

TEN_HUONG = ("ngang", "dọc", "chéo $\\searrow$", "chéo $\\swarrow$")
LUOI5 = r"\fontsize{5}{5}\selectfont"


def cells(*rcs: tuple[int, int]) -> int:
    m = 0
    for r, c in rcs:
        m |= bit(r, c)
    return m


class Hinh:
    """Một tikzpicture gồm nhiều bàn cờ."""

    def __init__(self) -> None:
        self.dong: list[str] = []

    def ban(self, ox, oy, tieu_de, lop, o=0.31, truc=False, nhan=None, chan=None, n=N, quan=(), co=LUOI5):
        """lop = [(mask, kiểu)], lớp sau đè lớp trước; nhan: 'so' (chỉ số) | 'chu' (X/O) cho ô có màu đậm.

        quan = [(mask, chữ)]: in chữ đen đè lên ô mà không che màu nền của ô đó.
        """
        d = self.dong
        d.append(f"\\begin{{scope}}[shift={{({ox:.3f},{oy:.3f})}}]")
        da_ve: dict[int, str] = {}
        for mask, kieu in lop:
            for i in iter_bits(mask & FULL):
                da_ve[i] = kieu
        for i, kieu in da_ve.items():
            r, c = rc(i)
            nen, chu = KIEU[kieu]
            d.append(f"\\fill[{nen}] ({c * o:.3f},{-r * o:.3f}) rectangle ++({o:.3f},{-o:.3f});")
            if nhan and kieu not in ("mat", "vang", "xam", "nhat"):
                txt = str(i) if nhan == "so" else {"X": "X", "O": "O"}.get(kieu, "")
                if txt:
                    d.append(
                        f"\\node[text={chu},inner sep=0,font={co}] at "
                        f"({(c + .5) * o:.3f},{-(r + .5) * o:.3f}) {{{txt}}};"
                    )
        for mask, chu in quan:
            for i in iter_bits(mask):
                r, c = rc(i)
                d.append(
                    f"\\node[inner sep=0,font=\\fontsize{{7}}{{7}}\\selectfont\\bfseries] at "
                    f"({(c + .5) * o:.3f},{-(r + .5) * o:.3f}) {{{chu}}};"
                )
        w = n * o
        d.append(f"\\draw[gray!55,line width=.2pt] (0,0) grid[step={o:.3f}] ({w:.3f},{-w:.3f});")
        d.append(f"\\draw[gray!90!black,line width=.5pt] (0,0) rectangle ({w:.3f},{-w:.3f});")
        if truc:
            for k in range(n):
                d.append(f"\\node[font={LUOI5},gray] at ({(k + .5) * o:.3f},.09) {{{k}}};")
                d.append(f"\\node[font={LUOI5},gray] at (-.12,{-(k + .5) * o:.3f}) {{{k}}};")
        if tieu_de:
            d.append(f"\\node[anchor=south,font=\\footnotesize] at ({w / 2:.3f},{.16 if truc else .03}) {{{tieu_de}}};")
        if chan:
            d.append(f"\\node[anchor=north,font=\\footnotesize] at ({w / 2:.3f},{-w - .03:.3f}) {{{chan}}};")
        d.append("\\end{scope}")

    def ghi(self, ten: str) -> None:
        (OUT / f"{ten}.tex").write_text(
            "% Sinh tự động bởi report/scripts/ve_hinh_bitboard.py, đừng sửa tay.\n"
            + MAU
            + "\\begin{tikzpicture}\n"
            + "\n".join(self.dong)
            + "\n\\end{tikzpicture}\n",
            encoding="utf-8",
        )


def ghi_tex(ten: str, noi_dung: str) -> None:
    (OUT / f"{ten}.tex").write_text(
        "% Sinh tự động bởi report/scripts/ve_hinh_bitboard.py, đừng sửa tay.\n" + noi_dung, encoding="utf-8"
    )


def luoi(hinh: Hinh, panels, cot, o, gx, gy, **kw):
    """Xếp các panel (tiêu đề, lớp) theo lưới `cot` cột."""
    w = N * o
    for k, (tieu_de, lop) in enumerate(panels):
        hinh.ban(k % cot * (w + gx), -(k // cot) * (w + gy), tieu_de, lop, o=o, **kw)


# ---- 1. Chỉ số ô và biểu diễn ---------------------------------------------------------------

X_VD, O_VD = cells((4, 4), (5, 5), (5, 6)), cells((4, 6), (6, 4))


def hinh_chi_so() -> None:
    h = Hinh()
    o = 0.62
    d = h.dong
    nb = [(60, "bbVang")] + [(60 + s, "bbNhat") for s, _ in LINES]
    for i, mau in nb:
        r, c = rc(i)
        d.append(f"\\fill[{mau}] ({c * o:.3f},{-r * o:.3f}) rectangle ++({o:.3f},{-o:.3f});")
    d.append(f"\\draw[gray!55,line width=.2pt] (0,0) grid[step={o}] ({N * o:.3f},{-N * o:.3f});")
    d.append(f"\\draw[gray!90!black,line width=.5pt] (0,0) rectangle ({N * o:.3f},{-N * o:.3f});")
    for i in range(CELLS):
        r, c = rc(i)
        d.append(f"\\node[font=\\fontsize{{7}}{{7}}\\selectfont] at ({(c + .5) * o:.3f},{-(r + .5) * o:.3f}) {{{i}}};")
    for k in range(N):
        d.append(f"\\node[font=\\fontsize{{7}}{{7}}\\selectfont\\itshape,gray] at ({(k + .5) * o:.3f},.17) {{{k}}};")
        d.append(f"\\node[font=\\fontsize{{7}}{{7}}\\selectfont\\itshape,gray] at (-.2,{-(k + .5) * o:.3f}) {{{k}}};")
    d.append(f"\\node[anchor=south,font=\\footnotesize] at ({N * o / 2:.3f},.32) {{(a) Chỉ số bit $i = 11r + c$; từ ô 60 đi 4 hướng}};")
    h.ban(N * o + 0.9, 0, "(b) Thế cờ ví dụ", [(X_VD, "X"), (O_VD, "O")], o=o, nhan="chu", truc=True, co="\\footnotesize\\bfseries")
    h.ghi("chi-so")


def bang_bieu_dien() -> None:
    def tong(b):
        return " + ".join(f"2^{{{i}}}" for i in iter_bits(b))

    ghi_tex(
        "bang-bieu-dien",
        "\\begin{tabular}{@{}lll@{}}\n\\toprule\n"
        "\\textbf{Bên} & \\textbf{Giá trị số nguyên} & \\textbf{Dạng hexa} \\\\\n\\midrule\n"
        f"X & ${tong(X_VD)}$ & \\texttt{{{X_VD:#x}}} \\\\\n"
        f"O & ${tong(O_VD)}$ & \\texttt{{{O_VD:#x}}} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n",
    )


def hinh_dich_bit() -> None:
    x = cells((5, 5), (3, 0), (7, 10))
    panels = [("$x$", [(x, "X")])]
    for (d, _), (dr, dc), ten in zip(LINES, DIRECTIONS, ("1", "11", "12", "10"), strict=True):
        dung = sai = 0
        for i in iter_bits(x):
            r, c = rc(i)
            if i - d < 0:
                continue
            if inside(r - dr) and inside(c - dc):
                dung |= 1 << (i - d)
            else:
                sai |= 1 << (i - d)
        assert dung | sai == x >> d
        panels.append((f"$x \\gg {ten}$", [(dung, "X"), (sai, "sai")]))
    h = Hinh()
    luoi(h, panels, 5, 0.27, 0.27, 0)
    h.ghi("dich-bit")


def hinh_start() -> None:
    h = Hinh()
    mau_cua_so = ((2, 6), (6, 3), (3, 5), (1, 7))
    panels = []
    for (d, start), ten, s in zip(LINES, TEN_HUONG, mau_cua_so, strict=True):
        i0 = index(*s)
        assert (start >> i0) & 1
        cs = sum(1 << (i0 + k * d) for k in range(WIN))
        panels.append((f"{ten}, $d={d}$: {popcount(start)} ô", [(start, "mat"), (cs, "X")]))
    luoi(h, panels, 4, 0.31, 0.65, 0, nhan="so")
    h.ghi("start")


# ---- 2. Kiểm tra thắng ------------------------------------------------------------------------


def hinh_has_five() -> None:
    d, start = LINES[2]
    x = cells(*[(k, k) for k in range(2, 7)], (2, 8), (7, 1), (9, 6))
    t1 = x & (x >> d)
    s1 = t1 >> (2 * d)
    t2 = t1 & s1
    s2 = x >> (4 * d)
    t3 = t2 & s2
    kq = t3 & start
    assert list(iter_bits(t1)) == [24, 36, 48, 60]
    assert list(iter_bits(t2)) == [24, 36]
    assert list(iter_bits(kq)) == [24] and has_five(x)
    assert all(_ba_buoc(x, dd) & st == 0 for dd, st in LINES[:2])  # hai hướng đầu chưa thấy dãy 5
    panels = [
        ("$x$", [(x, "X")]),
        (f"$x \\gg {d}$", [(x >> d, "nhat")]),
        (f"$t_1 = x \\mathbin{{\\&}} (x \\gg {d})$", [(t1, "X")]),
        (f"$t_1 \\gg {2 * d}$", [(s1, "nhat")]),
        (f"$t_2 = t_1 \\mathbin{{\\&}} (t_1 \\gg {2 * d})$", [(t2, "X")]),
        (f"$x \\gg {4 * d}$", [(s2, "nhat")]),
        (f"$t_3 = t_2 \\mathbin{{\\&}} (x \\gg {4 * d})$", [(t3, "X")]),
        ("$t_3 \\mathbin{\\&} \\mathrm{START}_{\\searrow}$", [(start, "mat"), (kq, "kq")]),
    ]
    h = Hinh()
    luoi(h, panels, 4, 0.31, 0.65, 0.8, nhan="so")
    h.ghi("has-five")


def _ba_buoc(x: int, d: int) -> int:
    t = x & (x >> d)
    t &= t >> (2 * d)
    t &= x >> (4 * d)
    return t


def hinh_lot_hang() -> None:
    a = cells((2, 8), (2, 9), (2, 10), (3, 0), (3, 1))
    b = cells((0, 8), (1, 9), (2, 10), (4, 0), (5, 1))
    assert list(iter_bits(a)) == [30, 31, 32, 33, 34]
    assert list(iter_bits(b)) == [8, 20, 32, 44, 56]
    panels = []
    for x, kh, (d, start) in ((a, "ngang", LINES[0]), (b, "\\searrow", LINES[2])):
        t3 = _ba_buoc(x, d)
        assert popcount(t3) == 1 and t3 & start == 0 and not has_five(x)
        ten = "\\text{ngang}" if kh == "ngang" else kh
        panels += [
            (f"$x$ (bước $d={d}$)", [(x, "X")]),
            ("$t_3$ (dãy 5 giả)", [(t3, "sai")]),
            (f"$\\mathrm{{START}}_{{{ten}}}$", [(start, "mat")]),
            ("$t_3 \\mathbin{\\&} \\mathrm{START} = 0$", [(start, "mat")]),
        ]
    h = Hinh()
    luoi(h, panels, 4, 0.31, 0.65, 0.8, nhan="so")
    h.ghi("lot-hang")


# ---- 3. Giãn nở và ứng viên -------------------------------------------------------------------


def hinh_gian_no() -> None:
    x = cells((3, 10), (7, 0))
    v = x | (x << N) | (x >> N)
    trai = v << 1
    phai = v >> 1
    h_ = v | (trai & NOT_COL0) | (phai & NOT_COL10)
    assert h_ & FULL == dilate(x)
    assert popcount(dilate(x)) == 12
    panels = [
        ("$x$", [(x, "X")]),
        ("$v = x \\mid (x \\ll 11) \\mid (x \\gg 11)$", [(v, "nhat"), (x, "X")]),
        ("$v \\ll 1$ (chưa lọc)", [(trai & NOT_COL0, "nhat"), (trai & ~NOT_COL0, "sai")]),
        ("$v \\gg 1$ (chưa lọc)", [(phai & NOT_COL10, "nhat"), (phai & ~NOT_COL10, "sai")]),
        ("$(v \\ll 1) \\mathbin{\\&} \\mathrm{NOT\\_COL0}$", [(trai & NOT_COL0, "nhat")]),
        ("$\\mathrm{dilate}(x)$, 12 ô", [(dilate(x), "nhat"), (x, "X")]),
    ]
    h = Hinh()
    luoi(h, panels, 3, 0.4, 0.75, 0.8)
    h.ghi("gian-no")


def hinh_ung_vien() -> None:
    occ = X_VD | O_VD
    d1 = dilate(occ)
    d2 = dilate(d1)
    cand = candidates(X_VD, O_VD, 2)
    assert cand == d2 & ~occ & FULL
    panels = [
        ("Thế cờ", [(X_VD, "X"), (O_VD, "O")]),
        (f"$\\mathrm{{dilate}}^1$: {popcount(d1)} ô", [(d1, "nhat"), (X_VD, "X"), (O_VD, "O")]),
        (f"$\\mathrm{{dilate}}^2$: {popcount(d2)} ô", [(d2, "nhat"), (X_VD, "X"), (O_VD, "O")]),
        (f"Ứng viên: {popcount(cand)} ô", [(cand, "kq")]),
    ]
    h = Hinh()
    luoi(h, panels, 4, 0.31, 0.65, 0)
    h.ghi("ung-vien")
    print("ung vien:", popcount(d1), popcount(d2), popcount(cand), "so voi hop le:", CELLS - popcount(occ))


# ---- 4. Cửa sổ, ô thắng ngay ------------------------------------------------------------------

HANG = 5
X_CS = cells((5, 1), (5, 2), (5, 4), (5, 5), (1, 7), (2, 6), (3, 5), (4, 4))
O_CS = cells((5, 7), (2, 2), (3, 8), (6, 6), (7, 3), (8, 8), (9, 1))
assert not (X_CS & O_CS) and not has_five(X_CS) and not has_five(O_CS)
DONG_CS = [(5, c) for c in range(7)]


def hinh_cua_so() -> None:
    d0, start = LINES[0]
    c0, c1, c2, free = window_counts(X_CS, O_CS, d0, start)
    h = Hinh()
    o = 0.62
    d = h.dong
    for c in range(N):
        i = index(HANG, c)
        kieu = "X" if (X_CS >> i) & 1 else "O" if (O_CS >> i) & 1 else None
        if kieu:
            nen, _ = KIEU[kieu]
            d.append(f"\\fill[{nen}] ({c * o:.3f},0) rectangle ++({o:.3f},{-o:.3f});")
            d.append(f"\\node[text=white,font=\\footnotesize\\bfseries] at ({(c + .5) * o:.3f},{-o / 2:.3f}) {{{kieu}}};")
        d.append(f"\\node[font=\\scriptsize,gray] at ({(c + .5) * o:.3f},.18) {{{c}}};")
    d.append(f"\\draw[gray!70,line width=.3pt] (0,0) grid[step={o}] ({N * o:.3f},{-o:.3f});")
    d.append(f"\\node[anchor=east,font=\\footnotesize] at (-.12,{-o / 2:.3f}) {{Hàng 5}};")
    y = -o - 0.16
    cao = 0.36
    for c in range(N - WIN + 1):
        i = index(HANG, c)
        song = (free >> i) & 1
        dem = ((c0 >> i) & 1) + 2 * ((c1 >> i) & 1) + 4 * ((c2 >> i) & 1)
        if song:
            mau = "bbKq" if dem == 4 else "bbNhat"
            chu = "white" if dem == 4 else "black"
            nhan = f"{dem} quân X"
        else:
            mau, chu, nhan = "bbXam", "black", "chứa O: loại"
        d.append(f"\\fill[{mau}] ({c * o + .03:.3f},{y:.3f}) rectangle ++({WIN * o - .06:.3f},{-cao:.3f});")
        d.append(f"\\node[text={chu},font=\\scriptsize] at ({(c + WIN / 2) * o:.3f},{y - cao / 2:.3f}) {{{nhan}}};")
        d.append(f"\\node[anchor=east,font=\\scriptsize] at (-.12,{y - cao / 2:.3f}) {{bắt đầu ở cột {c}}};")
        y -= cao + 0.07
    h.ghi("cua-so")


def nhi_phan(m: int) -> str:
    """Bit của hàng 5, cột 0..6, viết từ trái sang phải."""
    return "".join(str((m >> index(HANG, c)) & 1) for c in range(N - WIN + 1))


def bang_cong() -> None:
    d, start = LINES[0]
    free = start
    for k in range(WIN):
        free &= ~(O_CS >> (k * d))
    c0 = c1 = c2 = 0
    dong = []
    for k in range(WIN):
        p = (X_CS >> (k * d)) & free
        carry = c0 & p
        c0 ^= p
        carry2 = c1 & carry
        c1 ^= carry
        c2 |= carry2
        cot = [p, carry, c0, carry2, c1, c2]
        dong.append(f"{k} & " + " & ".join(f"\\texttt{{{nhi_phan(m)}}}" for m in cot) + " \\\\")
    assert (c0, c1, c2, free) == window_counts(X_CS, O_CS, d, start)
    dem = [
        ((c0 >> index(HANG, c)) & 1) + 2 * ((c1 >> index(HANG, c)) & 1) + 4 * ((c2 >> index(HANG, c)) & 1)
        for c in range(7)
    ]
    assert dem == [3, 4, 3, 0, 0, 0, 0], dem
    assert nhi_phan(free) == "1110000"
    assert nhi_phan(exactly(4, c0, c1, c2, free)) == "0100000"
    assert nhi_phan(exactly(3, c0, c1, c2, free)) == "1010000"
    ghi_tex(
        "bang-cong",
        "\\begin{tabular}{@{}c cccccc@{}}\n\\toprule\n"
        "$k$ & $p$ & \\textit{carry} & $c_0$ & \\textit{carry2} & $c_1$ & $c_2$ \\\\\n\\midrule\n"
        + "\n".join(dong)
        + "\n\\bottomrule\n\\end{tabular}\n",
    )


def bang_o_thang() -> None:
    empty = ~(X_CS | O_CS) & FULL
    d, start = LINES[0]
    w4 = exactly(4, *window_counts(X_CS, O_CS, d, start))
    assert list(iter_bits(w4)) == [index(5, 1)]
    dong = []
    for j in range(WIN):
        s = w4 << (j * d)
        o_ = next(iter(iter_bits(s)))
        r, c = rc(o_)
        trong = (s & empty) != 0
        dong.append(f"{j} & {o_} & $({r}, {c})$ & {'trống, giữ lại' if trong else 'đã có quân X, bị loại'} \\\\")
    ghi_tex(
        "bang-o-thang",
        "\\begin{tabular}{@{}cccl@{}}\n\\toprule\n"
        "$j$ & \\textbf{Bit của} $w_4 \\ll j d$ & \\textbf{Ô} & \\textbf{Sau khi} AND \\textit{empty} \\\\\n\\midrule\n"
        + "\n".join(dong)
        + "\n\\bottomrule\n\\end{tabular}\n",
    )


def hinh_o_thang() -> None:
    panels = [("Thế cờ", [(X_CS, "X"), (O_CS, "O")])]
    wtong = 0
    for (d, start), ten in zip(LINES, TEN_HUONG, strict=True):
        w4 = exactly(4, *window_counts(X_CS, O_CS, d, start))
        wtong |= w4
        if d in (1, 10):
            assert popcount(w4) == (1 if d == 1 else 2)
            panels.append((f"$w_4$ {ten}", [(X_CS, "mat"), (w4, "vang")]))
        else:
            assert w4 == 0
    res = winning_squares(X_CS, O_CS)
    assert sorted(rc(i) for i in iter_bits(res)) == [(0, 8), (5, 3)]
    assert winning_squares(O_CS, X_CS) == 0          # O đến lượt mà không thắng ngay được
    panels.append(("Ô thắng ngay", [(X_CS, "X"), (O_CS, "O"), (res, "kq")]))
    h = Hinh()
    luoi(h, panels, 4, 0.31, 0.65, 0, nhan="so")
    h.ghi("o-thang")
    # sau khi đặt vào ô thắng, has_five phải đúng
    assert has_five(X_CS | bit(5, 3)) and has_five(X_CS | bit(0, 8))
    print("w4 cheo:", [rc(i) for i in iter_bits(exactly(4, *window_counts(X_CS, O_CS, 10, LINES[3][1])))])


# ---- 5. Hàm đánh giá --------------------------------------------------------------------------

X_DG, O_DG = cells((5, 5), (5, 6)), cells((4, 6))


def bang_danh_gia() -> None:
    dong = []
    tong = {}
    for ten_ben, me, opp in (("X", X_DG, O_DG), ("O", O_DG, X_DG)):
        s = 0
        for (d, start), ten in zip(LINES, TEN_HUONG, strict=True):
            cnt = window_counts(me, opp, d, start)
            n = [exactly(k, *cnt).bit_count() for k in range(1, WIN)]
            diem = sum(WEIGHT[k] * n[k - 1] for k in range(1, WIN))
            s += diem
            dong.append(f"{ten_ben} & {ten} & {n[0]} & {n[1]} & {n[2]} & {n[3]} & {diem} \\\\")
        assert s == side_score(me, opp)
        tong[ten_ben] = s
        dong.append(f"\\multicolumn{{6}}{{@{{}}r}}{{\\textbf{{Tổng bên {ten_ben}}}}} & \\textbf{{{s}}} \\\\")
        if ten_ben == "X":
            dong.append("\\midrule")
    assert evaluate(X_DG, O_DG) == tong["X"] - tong["O"]
    print("danh gia:", tong, "evaluate =", evaluate(X_DG, O_DG))
    ghi_tex(
        "bang-danh-gia",
        "\\begin{tabular}{@{}ll rrrr r@{}}\n\\toprule\n"
        "\\textbf{Bên} & \\textbf{Hướng} & $n_1$ & $n_2$ & $n_3$ & $n_4$ & \\textbf{Điểm} \\\\\n\\midrule\n"
        + "\n".join(dong)
        + "\n\\bottomrule\n\\end{tabular}\n",
    )


def hinh_danh_gia() -> None:
    panels = []
    for ten_ben, me, opp in (("X", X_DG, O_DG), ("O", O_DG, X_DG)):
        for (d, start), ten in zip(LINES, TEN_HUONG, strict=True):
            cnt = window_counts(me, opp, d, start)
            panels.append((f"{ten_ben}, {ten}", [(exactly(1, *cnt), "nhat"), (exactly(2, *cnt), "vang")]))
    h = Hinh()
    luoi(h, panels, 4, 0.31, 0.65, 0.8, quan=((X_DG, "X"), (O_DG, "O")))
    h.ghi("danh-gia")


# ---- 6. iter_bits, Zobrist, đối xứng ---------------------------------------------------------


def bang_iter_bits() -> None:
    b = 0b1101000
    dong = []
    thu = []
    while b:
        am = (-b) & 0xFF
        low = b & -b
        i = low.bit_length() - 1
        sau = b ^ low
        dong.append(
            f"\\texttt{{{b:08b}}} & \\texttt{{{am:08b}}} & \\texttt{{{low:08b}}} & {i} & \\texttt{{{sau:08b}}} \\\\"
        )
        thu.append(i)
        b = sau
    assert thu == list(iter_bits(0b1101000)) == [3, 5, 6]
    ghi_tex(
        "bang-iter-bits",
        "\\begin{tabular}{@{}ccccc@{}}\n\\toprule\n"
        "$b$ & $-b$ (bù 2, 8 bit thấp) & $low = b \\mathbin{\\&} -b$ & \\textbf{Chỉ số} & $b \\oplus low$ \\\\\n\\midrule\n"
        + "\n".join(dong)
        + "\n\\bottomrule\n\\end{tabular}\n",
    )


def bang_zobrist() -> None:
    moves = [60, 61, 48, 50]
    g = CaroGame()
    dong = [f"--- & --- & --- & \\texttt{{{g.key:#018x}}} \\\\"]
    for a in moves:
        ben = g.side
        z = ZOBRIST[ben][a]
        g.step(a)
        dong.append(f"{a} & {'XO'[ben]} & \\texttt{{{z:#018x}}} & \\texttt{{{g.key:#018x}}} \\\\")
    g2 = CaroGame.from_moves([48, 50, 60, 61])
    assert g2.key == g.key == full_key(g.bb[0], g.bb[1], g.side)
    assert g.side == 0
    k_cuoi = g.key
    a = g.undo()
    assert a == 50 and g.key == full_key(g.bb[0], g.bb[1], g.side)
    ghi_tex(
        "bang-zobrist",
        "\\begin{tabular}{@{}cccc@{}}\n\\toprule\n"
        "\\textbf{Nước} & \\textbf{Bên} & $Z[\\text{bên}][a]$ & \\textbf{Khoá sau nước đi} \\\\\n\\midrule\n"
        + "\n".join(dong)
        + "\n\\bottomrule\n\\end{tabular}\n",
    )
    print(f"SIDE_KEY = {SIDE_KEY:#018x}, khoa cuoi = {k_cuoi:#018x}")
    ghi_tex("side-key", f"\\texttt{{{SIDE_KEY:#018x}}}%\n")


TEN_DX = (
    "0: giữ nguyên",
    "1: xoay 90°",
    "2: xoay 180°",
    "3: xoay 270°",
    "4: lật trái--phải",
    "5: lật trên--dưới",
    "6: lật chéo chính",
    "7: lật chéo phụ",
)


def hinh_doi_xung() -> None:
    x = cells((1, 2), (1, 3), (2, 2))
    o_ = cells((3, 4), (0, 5))
    khoa = [full_key(transform_bb(x, k), transform_bb(o_, k), 0) for k in range(COUNT)]
    assert len(set(khoa)) == COUNT and len({transform_bb(x, k) for k in range(COUNT)}) == COUNT
    kc = canonical_key(x, o_, 0)
    nho = khoa.index(kc)
    h = Hinh()
    o = 0.31
    w = N * o
    for k in range(COUNT):
        tag = "\\textbf{" + f"{khoa[k]:016x}"[:8] + "\\dots}" if k == nho else f"{khoa[k]:016x}"[:8] + "\\dots"
        h.ban(
            k % 4 * (w + 0.65),
            -(k // 4) * (w + 1.3),
            TEN_DX[k],
            [(transform_bb(x, k), "X"), (transform_bb(o_, k), "O")],
            o=o,
            nhan="chu",
            chan=tag,
        )
    h.ghi("doi-xung")
    print("o 13 qua 8 phep:", [transform_action(13, k) for k in range(COUNT)], "PERMS ok:", PERMS[1][13])
    print("khoa nho nhat o phep", nho, f"{kc:#018x}")
    ghi_tex(
        "bang-doi-xung",
        "\\begin{tabular}{@{}l" + "c" * COUNT + "@{}}\n\\toprule\n"
        "\\textbf{Phép $k$} & " + " & ".join(str(k) for k in range(COUNT)) + " \\\\\n\\midrule\n"
        "\\textbf{Ảnh của ô 13} & " + " & ".join(str(transform_action(13, k)) for k in range(COUNT)) + " \\\\\n"
        "\\bottomrule\n\\end{tabular}\n",
    )


def kiem_tra_van_mau() -> None:
    """Đoạn chạy thử in trong mục API: phải đúng từng dòng."""
    g = CaroGame()
    for a in (60, 0, 61, 1, 62, 2, 63, 3):
        assert g.step(a) == (0.0, False, {"winner": None})
    assert g.step(64) == (1.0, True, {"winner": 0})
    assert g.result() == 1 and g.termination() == "five" and g.legal_actions() == []
    assert g.undo() == 64 and g.result() is None and g.side == 0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("cua so moi huong:", [popcount(s) for _, s in LINES], "tong", sum(popcount(s) for _, s in LINES))
    for f in (
        hinh_chi_so, bang_bieu_dien, hinh_dich_bit, hinh_start, hinh_has_five, hinh_lot_hang, hinh_gian_no,
        hinh_ung_vien, hinh_cua_so, bang_cong, bang_o_thang, hinh_o_thang, bang_danh_gia, hinh_danh_gia,
        bang_iter_bits, bang_zobrist, hinh_doi_xung, kiem_tra_van_mau,
    ):
        f()
    print("Đã ghi vào", OUT)


if __name__ == "__main__":
    main()
