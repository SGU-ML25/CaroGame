"""Sinh phụ lục phân tích lực lượng thống kê (không dùng scipy)."""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/ban-mo-rong/phu-luc-thong-ke.tex"


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def exact_power(n: int, p_true: float, p0: float = 0.5, alpha: float = 0.05) -> float:
    """Power của kiểm định nhị thức hai phía H0: p=p0, với cỡ mẫu n."""
    # tìm miền bác bỏ: k <= lo hoặc k >= hi
    lo = 0
    while binom_cdf(lo, n, p0) <= alpha / 2 and lo < n:
        lo += 1
    lo -= 1
    hi = n
    while binom_cdf(hi - 1, n, p0) >= 1 - alpha / 2 and hi > 0:
        hi -= 1
    power = binom_cdf(lo, n, p_true) + (1 - binom_cdf(hi - 1, n, p_true))
    return max(0.0, min(1.0, power))


def two_prop_power(n: int, p1: float, p2: float, alpha: float = 0.05) -> float:
    pbar = (p1 + p2) / 2
    se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
    se1 = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    z = 1.96
    if se1 == 0:
        return 1.0
    return 0.5 * (1 + math.erf((abs(p1 - p2) - z * se0) / (math.sqrt(2) * se1)))


def pct(x: float) -> str:
    return f"{100 * x:.1f}\\%".replace(".", ",")


def main() -> None:
    lines = [
        r"\chapter{Phụ lục phân tích lực lượng thống kê}",
        r"\label{app:thong-ke}",
        ("Phụ lục này trả lời câu hỏi: với cỡ mẫu đã dùng, báo cáo có thể phân biệt được những "
         "chênh lệch nào? Các bảng dùng khoảng Wilson, kiểm định nhị thức chính xác và xấp xỉ "
         "chuẩn cho hai tỷ lệ. Mục đích là để người đọc biết giới hạn của kết luận, không phải "
         "để thay thế phép đo."),
    ]

    # 1. Bề rộng Wilson
    ns = [25, 50, 100, 200, 500, 700]
    ps = [0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []
    for n in ns:
        row = [str(n)]
        for p in ps:
            lo, hi = wilson(p, n)
            row.append(f"{100 * (hi - lo):.1f}".replace(".", ","))
        rows.append(row)
    lines += [
        r"\section{Bề rộng khoảng Wilson}",
        (r"Bảng dưới là bề rộng (điểm phần trăm) của khoảng Wilson 95\% theo $n$ và tỷ lệ quan "
         r"sát. Với $n=100$ và $p=0{,}5$, bề rộng khoảng 19,6 điểm; với $n=700$, còn khoảng 7,4 "
         r"điểm. Đây là lý do RR8 dùng 700 ván mỗi tác tử: đủ để phân biệt các nhóm cách nhau "
         r"khoảng 10 điểm, nhưng không đủ để phân biệt các tác tử cách nhau 3--5 điểm."),
        r"\begin{center}\small",
        r"\begin{tabular}{@{}r" + "r" * len(ps) + r"@{}}",
        r"\toprule",
        r"$n$ & " + " & ".join(f"$p={p}$".replace(".", ",") for p in ps) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]

    # 2. Power kiểm định nhị thức
    rows = []
    for n in [50, 100, 200, 500, 700]:
        row = [str(n)]
        for p in [0.55, 0.60, 0.65, 0.70, 0.80, 0.90]:
            row.append(f"{100 * exact_power(n, p):.0f}".replace(".", ","))
        rows.append(row)
    lines += [
        r"\section{Lực lượng của kiểm định nhị thức}",
        (r"Bảng dưới là xác suất phát hiện ($1-\beta$, \%) khi kiểm định hai phía $H_0:p=0{,}5$ "
         r"ở mức $\alpha=0{,}05$, với tỷ lệ thật $p$. Ví dụ, với 100 ván và $p=0{,}7$, lực lượng "
         r"khoảng 97\%; với $p=0{,}6$, chỉ khoảng 52\%. Điều này có nghĩa một kết quả 60\% trên "
         r"100 ván thường chưa đủ để bác bỏ 50\%, trong khi 70\% thì đủ."),
        r"\begin{center}\small",
        r"\begin{tabular}{@{}r" + "r" * 6 + r"@{}}",
        r"\toprule",
        r"$n$ & $p=0,55$ & $p=0,60$ & $p=0,65$ & $p=0,70$ & $p=0,80$ & $p=0,90$ \\",
        r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]

    # 3. So sánh hai tỷ lệ
    rows = []
    for n in [50, 100, 200, 500]:
        row = [str(n)]
        for d in [0.05, 0.10, 0.15, 0.20, 0.30]:
            row.append(f"{100 * two_prop_power(n, 0.5 + d / 2, 0.5 - d / 2):.0f}".replace(".", ","))
        rows.append(row)
    lines += [
        r"\section{Lực lượng so sánh hai tác tử}",
        (r"Bảng dưới là lực lượng xấp xỉ khi so hai tác tử, mỗi bên $n$ ván, chênh lệch thật "
         r"$d$ điểm phần trăm, xung quanh mức 50\%. Với $n=100$ mỗi bên, chỉ chênh lệch từ 20 "
         r"điểm trở lên mới có lực lượng trên 80\%; với $n=500$, chênh lệch 10 điểm đã đủ. Đây "
         r"là lý do các so sánh nhỏ trong báo cáo được ghi là ``chưa phân biệt được''."),
        r"\begin{center}\small",
        r"\begin{tabular}{@{}r" + "r" * 5 + r"@{}}",
        r"\toprule",
        r"$n$ mỗi bên & $d=5$ & $d=10$ & $d=15$ & $d=20$ & $d=30$ \\",
        r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]

    # 4. So sánh nhiều lần
    lines += [
        r"\section{So sánh nhiều lần}",
        (r"RR8 có 28 cặp đấu. Nếu mỗi cặp kiểm định ở $\alpha=0{,}05$ và giả thuyết không đúng, "
         r"xác suất có ít nhất một dương tính giả là $1-0{,}95^{28}\approx 76\%$. Vì vậy không "
         r"thể đọc từng cặp có $p<0{,}05$ như bằng chứng độc lập. Báo cáo không hiệu chỉnh "
         r"Bonferroni vì mục tiêu là mô tả, nhưng chỉ kết luận khi chênh lệch lớn hơn khoảng "
         r"tin cậy và nhất quán với phân tích chiến thuật."),
        (r"Với 8 tác tử, số cặp là 28; nếu chỉ xét 5 cặp chính (L3--L5, L3--L6, L3--L7, L3--L8, "
         r"L7--L8), xác suất có ít nhất một dương tính giả là $1-0{,}95^5\approx 23\%$. Ngay cả "
         r"với 5 cặp, cần thận trọng. Cách tốt hơn là đặt trước các cặp chính và báo cáo khoảng "
         r"tin cậy, thay vì quét toàn bộ rồi chọn cặp có ý nghĩa."),
    ]

    # 5. Áp dụng vào RR8
    lines += [
        r"\section{Áp dụng vào RR8}",
        (r"RR8 có 700 ván mỗi tác tử (100 ván mỗi cặp, 7 cặp). Khoảng Wilson của L8 là "
         r"[87,3; 91,8], của L7 [79,0; 84,7], của L6 [58,2; 65,4], của L3 [54,6; 61,9], của L5 "
         r"[46,5; 53,9], của L2 [39,9; 47,3]. Các khoảng này cho phép kết luận: L8 và L7 tạo "
         r"thành nhóm dẫn đầu tách biệt; L6 và L3 chồng lấn nhẹ; L3 và L5 gần chạm; L2 thấp hơn "
         r"L5 nhưng khoảng tin cậy chạm nhẹ; L4 và L1 tách hẳn ở đáy."),
        (r"Điều không thể kết luận: L6 mạnh hơn L3 (chênh 3,6 điểm, khoảng tin cậy chồng lấn), "
         r"L5 mạnh hơn L2 (chênh 6,6 điểm, khoảng tin cậy chạm nhẹ), L8 mạnh hơn L7 (chênh 7,8 "
         r"điểm, khoảng tin cậy tách rời nhưng chỉ 100 ván trực tiếp). Muốn phân biệt các chênh "
         r"lệch 3--8 điểm cần khoảng 1\,000--3\,000 ván mỗi cặp, vượt ngân sách của đồ án."),
    ]

    OUT.write_text("\n\n".join(lines), encoding="utf-8")
    print("ghi", OUT)


if __name__ == "__main__":
    main()
