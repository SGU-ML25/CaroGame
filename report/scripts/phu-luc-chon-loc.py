"""Sinh hai phụ lục chọn lọc: khai cuộc A3/A4 và 50 vòng huấn luyện L8.

Thay cho các hồ sơ lặp lại từng khai cuộc/từng vòng. Chạy từ gốc repo:

    uv run python report/scripts/phu-luc-chon-loc.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/ban-mo-rong"
RUNS = ROOT / "runs"

A3 = RUNS / "tournament/a3_mcts_sims/20261114-232353/games.jsonl"
A4 = RUNS / "tournament/a4_rollout/20261114-234556/games.jsonl"
L8 = RUNS / "train/alphazero/20261124-160615_gpu-native"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def coord(m: int) -> str:
    return f"{chr(65 + m % 11)}{m // 11 + 1}"


def opening_len(r: dict) -> int:
    return max(0, len(r["moves"]) - len(r.get("time_ms") or []))


def pct(x: float) -> str:
    return f"{100 * x:.1f}\\%".replace(".", ",")


def wdl_table(records: list[dict], agent: str) -> tuple[int, int, int]:
    w = d = l = 0
    for r in records:
        mine = 1 if r["agent_x"] == agent else -1
        if r["result"] == 0:
            d += 1
        elif r["result"] == mine:
            w += 1
        else:
            l += 1
    return w, d, l


def opening_appendix() -> None:
    data = {"A3": rows(A3), "A4": rows(A4)}
    summary_rows = []
    for tag, records in data.items():
        challengers = sorted({a for r in records for a in (r["agent_x"], r["agent_o"]) if a.startswith("MCTS")})
        rivals = sorted({a for r in records for a in (r["agent_x"], r["agent_o"]) if not a.startswith("MCTS")})
        for c in challengers:
            for rival in rivals:
                sub = [r for r in records if {r["agent_x"], r["agent_o"]} == {c, rival}]
                if not sub:
                    continue
                by_op = defaultdict(list)
                for r in sub:
                    mine = 1 if r["agent_x"] == c else -1
                    by_op[r["opening_id"]].append(0.5 if r["result"] == 0 else (1.0 if r["result"] == mine else 0.0))
                vals = np.array([np.mean(v) for v in by_op.values()])
                summary_rows.append((tag, c, rival.split("-")[0], len(vals), pct(vals.mean()),
                                     f"{vals.std():.3f}".replace(".", ","),
                                     pct(vals.min()), pct(np.quantile(vals, .25)),
                                     pct(np.median(vals)), pct(np.quantile(vals, .75)), pct(vals.max()),
                                     int((vals == 0).sum()), int((vals == 1).sum())))
    table = [r"\begin{center}\scriptsize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}llrrrrrrrrrrr@{}}", r"\toprule",
             r"Mã & MCTS & Gặp & KC & TB & Độ lệch & Min & Q1 & Trung vị & Q3 & Max & =0 & =1 \\",
             r"\midrule"]
    table += [" & ".join(map(str, row)) + r" \\" for row in summary_rows]
    table += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]

    # Top/bottom openings của MCTS-6400 vs L3
    sub = [r for r in data["A3"] if {r["agent_x"], r["agent_o"]} == {"MCTS-6400", "L3-AlphaBeta"}]
    by_op = defaultdict(list)
    for r in sub:
        mine = 1 if r["agent_x"] == "MCTS-6400" else -1
        by_op[r["opening_id"]].append(0.5 if r["result"] == 0 else (1.0 if r["result"] == mine else 0.0))
    scored = sorted(((op, float(np.mean(v)), opening_len(next(r for r in sub if r["opening_id"] == op)))
                     for op, v in by_op.items()), key=lambda x: -x[1])
    best, worst = scored[:5], scored[-5:]
    example_rows = []
    for label, group in (("Tốt nhất", best), ("Xấu nhất", worst)):
        for op, sc, _ in group:
            moves = next(r["moves"][:opening_len(next(r for r in sub if r["opening_id"] == op))] for _ in [0])
            example_rows.append((label, f"{op:02d}", " ".join(coord(m) for m in moves), pct(sc)))
    ex_table = [r"\begin{center}\small", r"\begin{tabular}{@{}lllr@{}}", r"\toprule",
                r"Nhóm & Khai cuộc & Chuỗi mở đầu & Điểm MCTS-6400 \\", r"\midrule"]
    ex_table += [" & ".join(map(str, row)) + r" \\" for row in example_rows]
    ex_table += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]

    text = [
        r"\chapter{Phụ lục khai cuộc: phân bố điểm theo 100 thế mở đầu}",
        r"\label{app:khai-cuoc}",
        ("Phụ lục này thay cho hồ sơ từng khai cuộc. Thay vì in 100 trang cho 100 khai cuộc, "
         "bảng dưới tóm tắt phân bố điểm của mỗi cấu hình trên 100 khai cuộc; mỗi khai cuộc có "
         "hai ván đổi màu. Cột ``=0'' và ``=1'' là số khai cuộc mà cấu hình thua cả hai ván hoặc "
         "thắng cả hai ván. Phân bố này quan trọng hơn điểm trung bình: nó cho biết kết quả phụ "
         "thuộc khai cuộc đến mức nào."),
        *table,
        (r"\section{Những khai cuộc tạo khác biệt lớn nhất}"),
        ("Bảng dưới chọn năm khai cuộc MCTS-6400 làm tốt nhất và năm khai cuộc làm xấu nhất "
         "trước L3. Đây là các thế mở đầu mà kết quả không phản ánh điểm trung bình; chúng hữu "
         "ích để kiểm tra lại bằng mắt và để thiết kế tập giữ riêng."),
        *ex_table,
        r"\section{Đọc phân bố}",
        ("Ba quan sát. Thứ nhất, độ lệch chuẩn của điểm theo khai cuộc thường 0,3--0,5, tức "
         "dao động rất lớn so với thang 0--1. Thứ hai, nhiều cấu hình có khai cuộc thắng cả hai "
         "ván và khai cuộc thua cả hai ván; điều này cho thấy khai cuộc không trung tính. Thứ ba, "
         "các cấu hình mạnh hơn có xu hướng ít khai cuộc ``=0'' hơn, nhưng vẫn có một số khai "
         "cuộc xấu. Vì vậy một đánh giá chỉ dùng vài khai cuộc có thể đảo ngược thứ hạng."),
        ("Hệ quả cho thiết kế giải: cần dùng nhiều khai cuộc, đổi màu, và bootstrap theo cặp. "
         "Kết quả trong Chương~4 dùng 50 khai cuộc; phụ lục này cho thấy 50 là mức tối thiểu, "
         "không phải mức lý tưởng. Muốn giảm khoảng tin cậy xuống dưới ±5 điểm phần trăm cần "
         "khoảng 200--400 khai cuộc, vượt ngân sách của đồ án."),
    ]
    (OUT / "ho-so-khai-cuoc.tex").write_text("\n\n".join(text), encoding="utf-8")


def training_appendix() -> None:
    metrics = [r for r in rows(L8 / "metrics.jsonl") if "gating" in r]
    log = rows(L8 / "games.jsonl")
    selfplay = [r for r in log if r["tournament_id"] == "selfplay"]
    gating = defaultdict(list)
    for r in log:
        if r["tournament_id"].startswith("L8-incumbent-"):
            gating[int(r["tournament_id"].rsplit("-", 1)[1])].append(r)

    rows_all = []
    for m in metrics:
        step = m["step"]
        games = selfplay[200 * (step - 1):200 * step]
        gate = gating[step]
        xw = sum(r["result"] == 1 for r in games)
        ow = sum(r["result"] == -1 for r in games)
        plies = [len(r["moves"]) for r in games]
        rows_all.append((step, f"{xw}/{200 - xw - ow}/{ow}", f"{np.median(plies):.0f}",
                         f"{m['loss']:.3f}".replace(".", ","), pct(m["gating"]["score"]),
                         "có" if m["accepted"] else "—"))
    t = [r"\begin{center}\scriptsize\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}rrrrrr@{}}", r"\toprule",
         r"Vòng & Tự đấu X/O & Ply P50 & Loss & Gating & Duyệt \\", r"\midrule"]
    t += [" & ".join(map(str, row)) + r" \\" for row in rows_all]
    t += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]

    selected = [1, 12, 24, 37, 50]
    deep = []
    for step in selected:
        m = next(x for x in metrics if x["step"] == step)
        gate = gating[step]
        games = selfplay[200 * (step - 1):200 * step]
        colors = []
        for color in ("X", "O"):
            sub = [r for r in gate if r[f"agent_{color.lower()}"] == "L8"]
            w = sum(r["result"] == 1 for r in sub)
            d = sum(r["result"] == 0 for r in sub)
            l = len(sub) - w - d
            colors.append((color, str(len(sub)), f"{w}/{d}/{l}", pct((w + 0.5 * d) / len(sub))))
        sp_x = sum(r["result"] == 1 for r in games)
        sp_o = sum(r["result"] == -1 for r in games)
        sp_d = 200 - sp_x - sp_o
        deep += [
            rf"\subsection{{Vòng {step:02d}}}",
            (f"Tự đấu 200 ván: X thắng {sp_x}, O thắng {sp_o}, hoà {sp_d}; độ dài trung vị "
             f"{np.median([len(r['moves']) for r in games]):.0f} nước. Loss tổng "
             f"{m['loss']:.4f}, policy {m['policy_loss']:.4f}, value {m['value_loss']:.4f}. "
             f"Gating {m['gating']['score']*100:.1f}\\%; {'được duyệt' if m['accepted'] else 'không được duyệt'}."),
            r"\begin{center}\small\begin{tabular}{@{}lrrr@{}}\toprule",
            r"Màu của L8 & Ván & W/D/L & Điểm \\", r"\midrule",
            *(" & ".join(row) + r" \\" for row in colors),
            r"\bottomrule\end{tabular}\end{center}",
        ]
    # Hồ sơ từng vòng: 50 khối ngắn, mỗi khối một bảng 4 dòng
    per_round = [r"\section{Hồ sơ từng vòng}"]
    for m in metrics:
        step = m["step"]
        games = selfplay[200 * (step - 1):200 * step]
        gate = gating[step]
        xw = sum(r["result"] == 1 for r in games)
        ow = sum(r["result"] == -1 for r in games)
        dr = len(games) - xw - ow
        plies = [len(r["moves"]) for r in games]
        colors = []
        for color in ("X", "O"):
            sub = [r for r in gate if r[f"agent_{color.lower()}"] == "L8"]
            w = sum(r["result"] == 1 for r in sub)
            d = sum(r["result"] == 0 for r in sub)
            l = len(sub) - w - d
            colors.append((color, str(len(sub)), f"{w}/{d}/{l}", pct((w + .5 * d) / len(sub))))
        per_round += [
            rf"\subsection*{{Vòng {step:02d}}}",
            (f"Tự đấu X/O/hoà: {xw}/{ow}/{dr}; độ dài P50 {np.median(plies):.0f}; "
             f"loss {m['loss']:.4f}; gating {pct(m['gating']['score'])}; "
             f"{'duyệt' if m['accepted'] else 'không duyệt'}."),
            r"\begin{center}\scriptsize\begin{tabular}{@{}lrrr@{}}\toprule",
            r"Màu L8 & Ván & W/D/L & Điểm \\", r"\midrule",
            *(" & ".join(row) + r" \\" for row in colors),
            r"\bottomrule\end{tabular}\end{center}",
        ]

    text = [
        r"\chapter{Phụ lục huấn luyện L8: 50 vòng và năm vòng chọn lọc}",
        r"\label{app:huan-luyen}",
        ("Phụ lục này thay cho hồ sơ từng vòng. Bảng đầu ghi cả 50 vòng; năm mục sau đi sâu vào "
         "các vòng đại diện: vòng 1 (khởi đầu), 12 (duyệt đầu), 24 (giữa), 37 (duyệt cuối) và "
         "50 (kết thúc). Điểm gating so ứng viên với bản đương nhiệm tại thời điểm đó, nên không "
         "so sánh trực tiếp giữa các vòng."),
        *t,
        r"\section{Năm vòng chọn lọc}",
        *deep,
        *per_round,
        r"\section{Đọc lịch sử huấn luyện}",
        ("Hai đặc điểm nổi bật. Thứ nhất, loss tổng giảm chậm và có dao động; loss policy chiếm "
         "phần lớn, loss value nhỏ hơn nhiều. Điều này hợp lý vì value đã được khởi tạo từ SL v3 "
         "và chỉ cần tinh chỉnh, còn policy phải học phân phối thăm của PUCT, khác với phân phối "
         "chuyên gia. Thứ hai, tỷ lệ duyệt thấp (6/50 vòng) nhưng không có nghĩa quá trình thất "
         r"bại: ngưỡng 55\% trên 100 ván là khắt khe, và nhiều ứng viên có điểm 50--55\% vẫn có "
         "thể đã tốt hơn ở một số khía cạnh. Việc giữ bản cũ khi nghi ngờ là lựa chọn thận trọng."),
        ("Một hạn chế: log không lưu phân phối thăm trung bình của PUCT theo vòng, nên không thể "
         "phân tích policy đã dịch chuyển như thế nào. Muốn làm điều đó cần thêm trường thống kê "
         "vào log tự đấu, hoặc chạy lại một tập thế cố định qua từng checkpoint."),
    ]
    (OUT / "ho-so-huan-luyen.tex").write_text("\n\n".join(text), encoding="utf-8")


def main() -> None:
    opening_appendix()
    training_appendix()
    print("xong")


if __name__ == "__main__":
    main()
