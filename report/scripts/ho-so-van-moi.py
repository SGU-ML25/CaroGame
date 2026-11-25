"""Sinh mục phân tích ván cho báo cáo: bàn TikZ, sự kiện chiến thuật, diễn biến.

Chạy từ gốc repo:

    uv run python report/scripts/ho-so-van-moi.py

Ghi report/ban-mo-rong/ho-so-van-moi.tex. Mọi nhận xét định lượng dựa trên engine
(winning_squares, has_five); phần diễn giải do người viết đặt trong NHAN_XET.
"""

from __future__ import annotations

import json
from pathlib import Path

from caro.engine import CaroGame, iter_bits, winning_squares
from caro.engine.bitboard import LINES, WIN

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/ban-mo-rong/ho-so-van-moi.tex"
RUNS = ROOT / "runs"

L8V3 = RUNS / "eval/tournament/l8_vs_l3/20261125-134954_round37/games.jsonl"
L8TRAIN = RUNS / "train/alphazero/20261124-160615_gpu-native/games.jsonl"


def coord(m: int) -> str:
    return f"{chr(65 + m % 11)}{m // 11 + 1}"


def opening_len(r: dict) -> int:
    if r.get("time_ms"):
        return max(0, len(r["moves"]) - len(r["time_ms"]))
    return 0


def winning_cells(bb: int) -> list[int]:
    cells = set()
    for d, start in LINES:
        t = bb & (bb >> d)
        t &= t >> (2 * d)
        t &= bb >> (4 * d)
        for s in iter_bits(t & start):
            cells.update(s + k * d for k in range(WIN))
    return sorted(cells)


def board(record: dict, upto: int | None = None, winning: bool = False) -> str:
    moves = record["moves"][:upto] if upto is not None else record["moves"]
    g = CaroGame.from_moves(moves)
    last = moves[-1] if moves else None
    win_cells = set(winning_cells(g.bb[g.winner])) if winning and g.winner is not None else set()
    s = [r"\begin{center}", r"\begin{tikzpicture}[x=.42cm, y=.42cm]"]
    for k in range(12):
        s.append(rf"\draw[gray!45, line width=.25pt] (0,{-k}) -- (11,{-k});")
        s.append(rf"\draw[gray!45, line width=.25pt] ({k},0) -- ({k},-11);")
    s.append(r"\draw[gray!80, line width=.5pt] (0,0) rectangle (11,-11);")
    for c in range(11):
        s.append(rf"\node[font=\scriptsize, text=gray] at ({c + .5},.42) {{{chr(65 + c)}}};")
    for r in range(11):
        s.append(rf"\node[font=\scriptsize, text=gray] at (-.45,{-r - .5}) {{{r + 1}}};")
    for r in range(11):
        for c in range(11):
            i = r * 11 + c
            if i in win_cells:
                s.append(rf"\fill[green!22] ({c},{-r}) rectangle ++(1,-1);")
            ch = "X" if (g.bb[0] >> i) & 1 else "O" if (g.bb[1] >> i) & 1 else None
            if ch:
                color = "black" if ch == "X" else "red!75!black"
                s.append(rf"\node[text={color}, font=\bfseries\scriptsize] at ({c + .5},{-r - .5}) {{{ch}}};")
            if i == last:
                s.append(rf"\draw[orange, line width=1pt] ({c + .08},{-r - .08}) rectangle ++(.84,-.84);")
    s += [r"\end{tikzpicture}", r"\end{center}"]
    return "\n".join(s)


def events(record: dict) -> list[tuple]:
    g = CaroGame.from_moves(record["moves"][:opening_len(record)])
    out = []
    for i, m in enumerate(record["moves"][opening_len(record):], start=opening_len(record)):
        me, opp = g.me_opp()
        wins = winning_squares(me, opp)
        threats = winning_squares(opp, me)
        color = "X" if g.side == 0 else "O"
        if wins.bit_count():
            if (wins >> m) & 1:
                out.append((i + 1, color, coord(m), "Thắng ngay", "—", 2))
            else:
                out.append((i + 1, color, coord(m), "Bỏ lỡ thắng ngay",
                            ", ".join(coord(x) for x in iter_bits(wins)), 5))
        elif threats.bit_count() == 1:
            if (threats >> m) & 1:
                out.append((i + 1, color, coord(m), "Chặn bắt buộc",
                            ", ".join(coord(x) for x in iter_bits(threats)), 3))
            else:
                out.append((i + 1, color, coord(m), "Bỏ chặn duy nhất",
                            ", ".join(coord(x) for x in iter_bits(threats)), 5))
        elif threats.bit_count() >= 2:
            out.append((i + 1, color, coord(m), "Đã ở thế buộc thua", "—", 1))
        g.step(m)
    return out


def critical(record: dict) -> str:
    """Mô tả cơ chế kết thúc: đòn kép hai đầu hay bỏ chặn đơn."""
    g = CaroGame.from_moves(record["moves"])
    if g.winner is None:
        return "Bàn đầy mà không bên nào có năm quân liên tiếp; kết quả là hoà."
    # vị trí trước nước cuối của bên thua (moves[-2]) — lúc bên thắng đã có đe doạ
    last = record["moves"][-1]
    moves = record["moves"]
    pre = CaroGame.from_moves(moves[:-2]) if len(moves) >= 2 else CaroGame()
    wins = winning_squares(pre.bb[g.winner], pre.bb[g.winner ^ 1])
    squares = [coord(x) for x in iter_bits(wins)]
    if len(squares) >= 2:
        return (f"Trước nước áp chót, bên thắng đã có {len(squares)} ô thắng ngay "
                f"({', '.join(squares)}); đối thủ chỉ chặn được một ô, nên nước "
                f"{coord(last)} hoàn thành dãy năm quân.")
    if len(squares) == 1:
        return (f"Trước nước áp chót, bên thắng có đúng một ô thắng ngay ({squares[0]}); "
                f"đối thủ không chặn, nên nước {coord(last)} hoàn thành dãy.")
    return f"Nước {coord(last)} hoàn thành dãy năm quân sau một chuỗi đe doạ trước đó."


NHAN_XET = {
    "L8–L3 · ván 07": (
        "L8 cầm X xây dựng cột G từ G5 xuống G1. Đến nước 20, X đã có bốn quân G2–G5; "
        "hai đầu G1 và G6 đều là ô thắng. O chặn G6, X đi G1 và thắng. Đây là đòn kép "
        "hai đầu kinh điển: không phải O bỏ sót một ô chặn duy nhất, mà O bị buộc phải "
        "chọn giữa hai ô. Ván cho thấy L8 biết biến lợi thế trung tâm thành đe doạ dọc."
    ),
    "L8–L3 · ván 04": (
        "Ván dài 54 nước, L8 cầm X thua. Ở nước 53, O đã có bốn quân C3, E3, F3, G3; D3 là "
        "ô thắng duy nhất. X đi G10 thay vì chặn D3, O đi D3 và hoàn thành hàng ngang "
        "C3–G3. Đây là lỗi chiến thuật rõ ràng duy nhất trong ván: trước đó hai bên đổi "
        "đe doạ ở nhiều cánh, nhưng đến nước 53 chỉ còn một nước cứu. L8 với 200 mô phỏng "
        "vẫn có thể bỏ sót nước bắt buộc khi cây tìm kiếm bị phân tán."
    ),
    "L3–L8 · ván 24": (
        "L8 cầm O thắng trong 26 nước, một trong bảy ván O thắng trước L3. O xây dựng "
        "đường chéo E7–F8–G9–H10; trước nước 25, hai đầu D6 và I11 đều là ô thắng. X chặn "
        "D6, O đi I11 và thắng. Đây là ví dụ L8 không chỉ phòng thủ khi cầm O mà còn tạo "
        "được đòn kép chéo. So với các ván O thua nhanh, ván này cho thấy chất lượng tìm "
        "kiếm quyết định khả năng lật thế."
    ),
    "L3–L8 · ván 26": (
        "L8 cầm O thua nhanh trong 21 nước. Ở nước 20, X có bốn quân D7, F7, G7, H7; E7 là "
        "ô thắng duy nhất. O đi C8, X đi E7 và hoàn thành hàng ngang D7–H7. Đây là lỗi "
        "bỏ chặn duy nhất điển hình: không có đe doạ thứ hai nào buộc O phải bỏ E7. Ván "
        "này thuộc lớp lỗi mà Bảng~\\ref{tab:tn-chien-thuat} đếm: L8 bỏ 1/82 lần chặn "
        "trong bộ dữ liệu L8–L3."
    ),
    "L3–L8 · ván 23": (
        "L8 cầm O thắng sau 56 nước, ván dài nhất trong nhóm O thắng. O kiên trì xây dựng "
        "hàng ngang E9–I9; trước nước 55, hai đầu D9 và I9 đều là ô thắng. X chặn D9, O đi "
        "I9. Ván này cho thấy L8 có thể chơi phòng thủ dài rồi phản công, không chỉ thắng "
        "nhanh bằng đe doạ sớm."
    ),
    "L8–L3 · ván 03": (
        "Ván hoà duy nhất của cặp L8–L3 khi L8 cầm X, kéo dài đủ 121 nước. Bàn đầy mà không "
        "bên nào có năm quân. Đây là bằng chứng L3 có thể phòng thủ đủ để cầm hoà trước L8 "
        "trong một số khai cuộc, dù L8 mạnh hơn về tổng thể. Ván hoà cũng cho thấy kết quả "
        "không chỉ có thắng/thua; với luật tự do, hoà là trạng thái hiếm nhưng có thật."
    ),
    "L8–L7 · ván 12": (
        "L8 cầm X thắng L7 trong 21 nước bằng hàng ngang C8–G8. Trước nước cuối, X có bốn "
        "quân D8–G8; hai đầu C8 và H8 là ô thắng. O đi J6, X đi C8. Đòn kép hai đầu lại "
        "xuất hiện, lần này ở hàng ngang. Ván minh hoạ L8 chuyển từ trung tâm sang cánh "
        "phải để tạo đe doạ ngang."
    ),
    "L8–L7 · ván 08": (
        "L8 cầm X thua L7 trong 24 nước. O thắng bằng đường chéo G4–K8; trước nước 23, hai "
        "đầu F3 và K8 là ô thắng. X đi E8, O đi K8. Không có lỗi bỏ chặn đơn; X bị buộc "
        "phải chọn giữa hai đầu. Đây là ví dụ thua vì đối thủ tạo đòn kép, không phải vì "
        "bỏ sót một nước cứu."
    ),
    "L7–L8 · ván 19": (
        "L7 cầm X thắng L8 trong 21 nước bằng cột G. Trước nước 20, X có bốn quân G3–G6; "
        "hai đầu G2 và G7 là ô thắng. O chặn G7, X đi G2. Ván này giống ván 07 ở chỗ đòn "
        "kép hai đầu quyết định, nhưng bên thắng là L7 và bên thua là L8. Nó cho thấy "
        "L8 không miễn nhiễm với đe doạ dọc."
    ),
    "L7–L8 · ván 15": (
        "L8 cầm O thắng L7 sau 58 nước bằng hàng ngang F1–J1. Trước nước 58, O có bốn quân "
        "G1–J1; hai đầu F1 và K1 là ô thắng. X đi K10, O đi F1. Ván dài, nhiều nước đổi "
        "cánh, nhưng cơ chế kết thúc vẫn là đòn kép hai đầu. Đây là ví dụ L8 cầm O thắng "
        "L7, góp phần vào 82,5\\% điểm của L8 trước L7."
    ),
    "RR8 · L4 thua L2 · ván 1101": (
        "Ván ngắn nhất trong nhóm L4 thua L2: chỉ 12 nước. Ở nước 11, O có bốn quân B6, C6, "
        "D6, E6; F6 là ô thắng duy nhất. L4 đi nước khác, L2 đi F6 và hoàn thành hàng ngang "
        "B6–F6. Đây là lỗi bỏ chặn duy nhất điển hình, và cũng là lý do L4 chỉ đạt 14,3\% "
        "tổng: với 400 mô phỏng, cây tìm kiếm quá thưa để thấy nước bắt buộc ở gốc."
    ),
    "RR8 · L5 thắng L2 · ván 517": (
        "L5 cầm O thắng L2 trong 22 nước, nhưng bỏ lỡ một nước thắng ngay ở nước 20. O vẫn "
        "thắng bằng đường chéo E4–I8 ở nước 22. Ván này cho thấy một lần bỏ lỡ thắng ngay "
        "không luôn dẫn đến thua, đặc biệt khi đối thủ không có đe doạ tức thì. Tuy nhiên, "
        "trong 402 cơ hội thắng ngay, L5 bỏ 54 lần; tỷ lệ này đủ để giải thích một phần "
        "khoảng cách giữa L5 và L6."
    ),
    "RR8 · L6 thắng L3 · ván 900": (
        "L6 cầm O thắng L3 trong 34 nước bằng hàng ngang E10–I10. Không có lỗi bỏ chặn hay "
        "bỏ thắng ngay trong ván; khác biệt đến từ tích luỹ đe doạ và chọn cánh. Đây là ví "
        "dụ vì sao L6 (61,9\%) nhỉnh hơn L3 (58,3\%) dù khoảng tin cậy chồng lấn: RL học "
        "được một số mẫu tấn công mà Alpha-Beta độ sâu 4 không thấy."
    ),
    "RR8 · L7 thắng L8 · ván 2401": (
        "L7 cầm X thắng L8 trong 21 nước bằng đường chéo G4–C8. Đây là một trong 34 ván L7 "
        "thắng L8 trên 100 ván cặp này; tỷ lệ 34\% là cao hơn nhiều so với các tác tử khác "
        "trước L8. Ván cho thấy L8 không bất khả chiến bại: khi L7 tạo được đe doạ chéo sớm, "
        "L8 cũng bị cuốn vào thế phòng thủ."
    ),
    "RR8 · L8 thắng L7 · ván 2402": (
        "L8 cầm O thắng L7 sau 64 nước bằng hàng ngang E2–J2 (sáu quân). Ván dài, nhiều lần "
        "đổi cánh, kết thúc bằng một hàng ngang sát biên. Đây là ví dụ L8 không chỉ thắng "
        "nhanh ở trung tâm mà còn biết kéo dài và chuyển cánh khi cần. Trong cặp L7–L8, L8 "
        "thắng 66/100 ván, nhưng 34 ván thua cho thấy khoảng cách không tuyệt đối."
    ),
    "RR8 · hoà L7–L8 · ván 2400": (
        "Một trong số ít ván hoà giữa L7 và L8, kéo dài đủ 121 nước. Bàn đầy mà không bên "
        "nào có năm quân. Ván hoà hiếm vì cả hai tác tử đều có 200 mô phỏng và đủ mạnh để "
        "trừng phạt sai lầm; khi cả hai phòng thủ đúng, bàn có thể đầy. Đây là lý do điểm "
        "không nên chỉ báo thắng/thua; hoà là kết quả có thật và cần được tính nửa điểm."
    ),
    "E3 · L4 tự đấu · ván 311": (
        "L4 tự đấu, bên O thắng trong 14 nước bằng đường chéo E1–I5. Ở nước 13, bên X bỏ "
        "chặn một ô thắng duy nhất; bên O đi vào và thắng. Ván này cô đọng điểm yếu của "
        "MCTS rollout ngẫu nhiên: nó không có tri thức bắt buộc, nên trong tự đấu vẫn bỏ "
        "chặn. Đây là lý do L4 tự đấu cân bằng 50–50 nhưng tuyệt đối không có nghĩa là "
        "L4 mạnh."
    ),
    "E3 · L5 tự đấu · ván 24": (
        "L5 tự đấu, bên X thắng trong 27 nước bằng cột D (D4–D8). Bên X bỏ lỡ một nước "
        "thắng ngay ở nước 25 nhưng vẫn thắng sau đó. Ván minh hoạ policy-only có thể sai "
        "ở nước quyết định mà không bị trừng phạt nếu đối thủ cũng là chính nó. Trong đấu "
        "với tác tử mạnh hơn, cùng lỗi đó thường bị trừng phạt ngay."
    ),
    "E3 · L7 tự đấu · ván 232": (
        "L7 tự đấu, bên O thắng trong 28 nước bằng hàng ngang D6–H6. Ở nước 27, bên X bỏ "
        "chặn một ô thắng duy nhất; bên O đi vào và thắng. Ngay cả PUCT với 200 mô phỏng "
        "vẫn có thể bỏ chặn trong tự đấu, dù tỷ lệ thấp hơn nhiều so với L4. Đây là bằng "
        "chứng định lượng rằng tìm kiếm giảm nhưng không xoá lỗi chiến thuật."
    ),
    "E3 · L8 tự đấu · ván 370": (
        "L8 tự đấu, bên X thắng trong 45 nước bằng hàng ngang C7–G7. Ở nước 44, bên O bỏ "
        "chặn một ô thắng duy nhất nhưng bên X không chuyển ngay thành thắng; đến nước 45 "
        "mới hoàn thành. Ván cho thấy ngay cả L8 cũng có lúc bỏ chặn, nhưng đối thủ không "
        "phải lúc nào cũng trừng phạt được. Đây là lý do phân tích lỗi phải đếm cả cơ hội "
        "và kết quả, không chỉ kết quả."
    ),
}


RR8 = RUNS / "eval/tournament/rr8_mode_b/latest/games.jsonl"
E3_RUNS = sorted((RUNS / "eval/tournament/e3_selfplay").glob("2026*"))


def load(path: Path, gid: int, tid: str | None = None) -> dict:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if tid:
        rows = [r for r in rows if r["tournament_id"] == tid]
    return next(r for r in rows if r["game_id"] == gid)


def load_rr8(gid: int) -> dict:
    return load(RR8, gid)


def load_e3(level: str, gid: int) -> dict:
    for d in E3_RUNS:
        rows = [json.loads(line) for line in (d / "games.jsonl").open(encoding="utf-8")]
        for r in rows:
            if r["agent_x"] == level and r["game_id"] == gid:
                return r
    raise KeyError((level, gid))


def section(label: str, r: dict) -> str:
    op = opening_len(r)
    outcome = {1: "X thắng", 0: "hoà", -1: "O thắng"}[r["result"]]
    ev = events(r)
    chosen = sorted(sorted(ev, key=lambda e: e[-1], reverse=True)[:6])
    lines = [
        rf"\subsection{{{label}}}",
        (f"Khai cuộc gồm {op} nước: " + " → ".join(coord(m) for m in r["moves"][:op]) +
         f". X: \\texttt{{{r['agent_x']}}}, O: \\texttt{{{r['agent_o']}}}. "
         f"Kết quả: {outcome} sau {len(r['moves'])} nước."),
    ]
    if chosen:
        rows = [(e[0], e[1], e[2], e[3], e[4]) for e in chosen]
        lines += [
            r"\begin{center}\small",
            r"\begin{tabular}{@{}rccll@{}}", r"\toprule",
            r"Ply & Màu & Nước & Nhận xét & Ô liên quan \\", r"\midrule",
            *(" & ".join(map(str, row)) + r" \\" for row in rows),
            r"\bottomrule", r"\end{tabular}", r"\end{center}",
        ]
    key = chosen[-1][0] if chosen else len(r["moves"])
    lines += [
        rf"\paragraph{{Thế cờ trước nước quyết định (sau nước {max(0, key - 1)})}}",
        board(r, upto=max(1, key - 1)),
        rf"\paragraph{{Thế kết thúc}}",
        board(r, winning=True),
        rf"\paragraph{{Cơ chế kết thúc}} {critical(r)}",
        rf"\paragraph{{Nhận xét}} {NHAN_XET.get(label, '')}",
        rf"\paragraph{{Diễn biến}} {{\footnotesize " +
        " ".join(f"{i + 1}.{coord(m)}" for i, m in enumerate(r["moves"])) + r"\par}",
    ]
    return "\n".join(lines)


def main() -> None:
    games = [
        ("L8–L3 · ván 07", load(L8V3, 7)),
        ("L8–L3 · ván 04", load(L8V3, 4)),
        ("L3–L8 · ván 24", load(L8V3, 24)),
        ("L3–L8 · ván 26", load(L8V3, 26)),
        ("L3–L8 · ván 23", load(L8V3, 23)),
        ("L8–L3 · ván 03", load(L8V3, 3)),
        ("L8–L7 · ván 12", load(L8TRAIN, 12, "L7-PUCT[l5]-37")),
        ("L8–L7 · ván 08", load(L8TRAIN, 8, "L7-PUCT[l5]-37")),
        ("L7–L8 · ván 19", load(L8TRAIN, 19, "L7-PUCT[l5]-37")),
        ("L7–L8 · ván 15", load(L8TRAIN, 15, "L7-PUCT[l5]-37")),
        ("RR8 · L4 thua L2 · ván 1101", load_rr8(1101)),
        ("RR8 · L5 thắng L2 · ván 517", load_rr8(517)),
        ("RR8 · L6 thắng L3 · ván 900", load_rr8(900)),
        ("RR8 · L7 thắng L8 · ván 2401", load_rr8(2401)),
        ("RR8 · L8 thắng L7 · ván 2402", load_rr8(2402)),
        ("RR8 · hoà L7–L8 · ván 2400", load_rr8(2400)),
        ("E3 · L4 tự đấu · ván 311", load_e3("L4-MCTS", 311)),
        ("E3 · L5 tự đấu · ván 24", load_e3("L5-PolicySL", 24)),
        ("E3 · L7 tự đấu · ván 232", load_e3("L7-PUCT", 232)),
        ("E3 · L8 tự đấu · ván 370", load_e3("L8-AlphaZero", 370)),
    ]
    intro = (
        "Chương này phát lại hai mươi ván tiêu biểu. Mỗi ván được chọn vì minh hoạ một cơ chế "
        "khác nhau: đòn kép hai đầu, bỏ chặn duy nhất, thắng bằng đường chéo, thắng dài bằng "
        "hàng ngang, hoà trên bàn đầy, và khác biệt giữa các cặp tác tử. Bàn cờ dùng chữ X/O; "
        "ô cam là nước vừa đi, nền xanh là dãy năm quân thắng. Các nhận xét định lượng lấy từ "
        "engine, không từ cảm nhận."
    )
    body = [r"\chapter{Phân tích ván: hai mươi ván tiêu biểu}", r"\label{ch:phan-tich-van}", intro] + [
        section(label, record) for label, record in games
    ]
    OUT.write_text("\n\n".join(body), encoding="utf-8")
    print("ghi", OUT)


if __name__ == "__main__":
    main()
