"""Sinh bàn cờ, bảng diễn biến và hình tỷ lệ thắng cho mục phân tích sâu bốn ván.

    uv run --no-sync --with matplotlib python report/scripts/chuyen_sau/bang_van_sau.py
"""

from __future__ import annotations

import chung
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chung import cell_name, num

KIND = {"five": "năm", "open_four": "tứ hở", "four": "tứ", "open_three": "ba hở"}


def x_view(entry: dict) -> float:
    """Điểm kỳ vọng KataGomo báo (thắng + nửa hoà, theo góc nhìn bên đi) đổi về góc nhìn X."""
    return entry["winrate"] if entry["ply"] % 2 == 0 else 1 - entry["winrate"]


def shape_text(ply: dict) -> str:
    parts = [f"{KIND[s['kind']]} {s['dir']}" for s in ply["shapes"]]
    if ply["blocked"] == "five":
        parts.insert(0, "chặn năm")
    elif ply["blocked"] == "double":
        parts.insert(0, "chặn đòn kép")
    return ", ".join(parts) or "---"


SAVING_SCORE = 0.2      # KataGomo phải chấm ít nhất 20% thì một nước thay thế mới được coi là giữ được ván
DROP = 0.05             # mức mất điểm nhỏ nhất đáng ghi ở ván không có sai lầm quyết định


def saving_moves(game: dict) -> list[int]:
    """Nước thay thế vừa không bị bác bỏ khi tìm sâu hơn, vừa được KataGomo chấm đủ cao."""
    if not (chung.DATA / "kiem-tra-thay-the.json").is_file():
        return []
    return [r["move"] for r in chung.load_json("kiem-tra-thay-the")
            if r["game_id"] == game["game_id"] and r["refuted"] is None and r["engine_score"] >= SAVING_SCORE]


def largest_drop(game: dict) -> int | None:
    """Nước của bên thua (trước nước khoá ván) làm điểm KataGomo của họ giảm nhiều nhất."""
    engine = {e["ply"]: e for e in game["engine"]}
    loser = 1 if game["result"] == 1 else 0
    drops = [(e["winrate"] - (1 - engine[p + 1]["winrate"]), p) for p, e in engine.items()
             if p % 2 == loser and p + 1 in engine and p < game["turning"]["first_proven"]]
    value, ply = max(drops, default=(0, None))
    return ply if value >= DROP else None


def move_table(game: dict) -> None:
    engine = {e["ply"]: e for e in game["engine"]}
    turning = game["turning"]
    decisive = turning["mistake"]["ply"] if saving_moves(game) else None
    costly = None if decisive is not None else largest_drop(game)
    rows = []
    for p in game["plies"]:
        e, proof = engine[p["ply"]], p["proof"]
        tag = ""
        if p["ply"] == decisive:
            tag = r"\textbf{sai lầm}"
        elif p["ply"] == costly:
            tag = "mất điểm"
        elif p["ply"] == turning["first_proven"]:
            tag = r"\textbf{khoá ván}"
        rows.append([str(p["ply"] + 1), f"{'XO'[p['side']]} {cell_name(p['move'])}", shape_text(p),
                     f"{proof['kind']} ({cell_name(proof['move'])})" if proof else "---",
                     num(100 * x_view(e), 0), cell_name(e["best"]), tag])
    chung.longtable(f"vs-{game['game_id']}", f"Diễn biến ván {game['game_id']} sau khai cuộc",
                    r"rlp{4.4cm}lrll",
                    ["Nước", "Đi", "Hình tạo / đỡ", "Thắng ép", "X \\%", "KG", ""], rows, size=r"\scriptsize")


def board(game: dict, upto: int, marks: dict[int, str], caption: str) -> list[str]:
    x, o = chung.replay(game["moves"], upto)
    numbers = {m: i + 1 for i, m in enumerate(game["moves"][:upto])}
    return [r"  \begin{subfigure}[t]{.48\linewidth}", r"    \centering",
            chung.board_tikz(x, o, numbers, marks, unit=0.40), rf"    \caption{{{caption}}}",
            r"  \end{subfigure}"]


def figures(game: dict) -> None:
    turning, moves = game["turning"], game["moves"]
    plies = {p["ply"]: p for p in game["plies"]}
    k = turning["first_proven"]
    panels = []
    mistake = turning["mistake"]
    if mistake:
        j = mistake["ply"]
        saving = saving_moves(game)
        marks = {m: "green!35" if m in saving else "gray!35" for m in mistake["safe"]}
        marks[moves[j]] = "red!40"
        text = (f"{len(saving)} ô xanh giữ được ván" if saving
                else f"{len(mistake['safe'])} ô xám chỉ hoãn thua, không cứu được")
        panels.append(board(game, j, marks, f"Trước nước {j + 1}: ô đỏ là nước đã đi; {text}"))
    key = plies[k]
    marks = {moves[k]: "orange!60"}
    for shape in key["shapes"]:
        for cell in shape["cells"]:
            marks[cell] = "yellow!60"
    panels.append(board(game, k + 1, marks, f"Sau nước {k + 1} (ô cam); ô vàng là điểm đe doạ nó tạo ra"))
    end = {moves[-1]: "green!35"}
    panels.append(board(game, len(moves), end, "Thế cuối (ô xanh: nước kết thúc)"))

    lines = [r"\begin{figure}[!htbp]", r"  \centering"]
    for i, panel in enumerate(panels):
        lines += panel
        lines.append(r"  \hfill" if i % 2 == 0 and i < len(panels) - 1 else r"  \par\medskip")
    lines += [(rf"  \caption{{Ván {game['game_id']}: X là {game['x']}, O là {game['o']}, {game['winner']} thắng. "
              r"Số trên quân là thứ tự nước.}"), rf"  \label{{fig:cs-vs-{game['game_id']}}}", r"\end{figure}", ""]
    (chung.TABLES / f"hinh-vs-{game['game_id']}.tex").write_text("\n".join(lines), encoding="utf-8")


def curves(games: list[dict]) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"],
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "savefig.dpi": 220, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 4.8), sharey=True)
    for ax, g in zip(axes.flat, games, strict=True):
        ax.plot([e["ply"] + 1 for e in g["engine"]], [100 * x_view(e) for e in g["engine"]], "-o",
                color="#257de0", markersize=3.5, linewidth=1.6)
        ax.axhline(50, color="#888888", linewidth=0.8)
        turning = g["turning"]
        if saving_moves(g):
            ax.axvline(turning["mistake"]["ply"] + 1, color="#ec4647", linestyle="--", linewidth=1.2)
        ax.axvline(turning["first_proven"] + 1, color="#f47230", linestyle=":", linewidth=1.6)
        ax.set_title(f"Ván {g['game_id']}: X={g['x']}, O={g['o']}, {g['winner']} thắng", fontsize=9)
        ax.set_ylim(-3, 103)
    for ax in axes[1]:
        ax.set_xlabel("nước thứ")
    for ax in axes[:, 0]:
        ax.set_ylabel("điểm của X theo KataGomo (%)")
    fig.tight_layout()
    fig.savefig(chung.FIGURES / "vs-ty-le-thang.png")
    plt.close(fig)


def alternatives_table(games: list[dict]) -> None:
    """Các nước lẽ ra nên đi: còn đứng vững khi tìm sâu hơn không, và KataGomo chấm bao nhiêu."""
    checked = chung.load_json("kiem-tra-thay-the")
    rows = []
    for g in games:
        mistake = g["turning"]["mistake"]
        if not mistake:
            continue
        engine = {e["ply"]: e for e in g["engine"]}
        after = engine.get(mistake["ply"] + 1)
        played = 1 - after["winrate"] if after else None
        if rows:
            rows.append([])
        loser = "XO"[mistake["ply"] % 2]
        rows.append([str(g["game_id"]), f"{mistake['ply'] + 1} ({loser})", f"{cell_name(mistake['played'])} (đã đi)",
                     "bị thắng ép", "---" if played is None else num(100 * played, 0)])
        for r in (r for r in checked if r["game_id"] == g["game_id"]):
            refuted = r["refuted"]
            verdict = (f"vẫn bị thắng ép, mở đầu {cell_name(refuted['move'])}" if refuted
                       else "không tìm thấy đường thắng ép")
            rows.append(["", "", cell_name(r["move"]), verdict, num(100 * r["engine_score"], 0)])
    chung.table("vs-thay-the", "Các nước thay thế cho sai lầm quyết định của bốn ván", "rlllr",
                ["Ván", "Nước", "Ô", "Tìm sâu hơn một bậc (3 đe doạ)", "KG \\%"], rows, size=r"\footnotesize",
                note="``KG'' là điểm KataGomo của bên vừa đi sau khi chọn ô đó (1\\,600 lượt thăm).")


def main() -> None:
    games = chung.load_json("phan-tich-van-sau")["games"]
    for game in games:
        move_table(game)
        figures(game)
    curves(games)
    if (chung.DATA / "kiem-tra-thay-the.json").is_file():
        alternatives_table(games)
    print(len(games), "ván")


if __name__ == "__main__":
    main()
