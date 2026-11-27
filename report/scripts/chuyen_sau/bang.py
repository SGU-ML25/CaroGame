"""Sinh mọi bảng report/chuyen-sau/bang/*.tex từ report/chuyen-sau/data/*.json.

    .venv/bin/python report/scripts/chuyen_sau/bang.py
"""

from __future__ import annotations

import collections
import statistics

import chung
from chung import AGENT_ORDER, SHORT, count, num, percent, table

CATEGORY = {"C1": "C1 thắng ngay", "C2": "C2 chặn", "C3": "C3 đòn kép", "C4": "C4 chuỗi tứ",
            "C5": "C5 phòng thủ"}
CATS = tuple(CATEGORY)


def p_value(p: float) -> str:
    return "$<0{,}001$" if p < 0.001 else num(p, 3)


# ---------------------------------------------------------------- bộ chiến thuật
def suite_tables() -> None:
    suite = chung.load_json("bo-chien-thuat")
    positions = suite["positions"]
    rows = []
    for c in CATS:
        group = [p for p in positions if p["category"] == c]
        answers = [len(p["answers"]) for p in group]
        rows.append([CATEGORY[c], str(len(group)), num(statistics.mean(p["ply"] for p in group)),
                     num(statistics.mean(p["candidates"] for p in group)), num(statistics.mean(answers)),
                     str(max(answers)),
                     num(100 * statistics.mean(len(p["answers"]) / p["candidates"] for p in group)),
                     num(statistics.mean(p["vcf_attacks"] for p in group), 2)])
    table("bo-mo-ta", "Đặc trưng năm loại thế của bộ kiểm tra chiến thuật", "lrrrrrrr",
          ["Loại", "Số thế", "Nước thứ", "Ứng viên", "Đáp án", "Max", "Ngẫu nhiên (\\%)", "Độ dài VCF"], rows,
          size=r"\footnotesize",
          note="Các cột 3--5 và 7--8 là trung bình. ``Ngẫu nhiên'' là xác suất trúng khi chọn đều trong tập ứng "
               "viên bán kính 2 (ước lượng bằng số đáp án chia số ứng viên). Với C5, độ dài VCF là của đối thủ.")


def accuracy_rows(results: dict, labels: list[tuple[str, str]], positions: list[dict]) -> list[list[str]]:
    category = {p["id"]: p["category"] for p in positions}
    rows = []
    for key, label in labels:
        ok = collections.Counter(category[r["id"]] for r in results[key] if r["ok"])
        total = collections.Counter(category[r["id"]] for r in results[key])
        rows.append([label, *[percent(ok[c], total[c], 0) for c in CATS],
                     percent(sum(ok.values()), sum(total.values()))])
    return rows


HEADER = ["Tác tử / cấu hình", *CATS, "Tổng"]


def tactics_tables() -> None:
    positions = chung.load_json("bo-chien-thuat")["positions"]
    base = chung.load_json("chien-thuat-base")
    table("ct-goc", "Tỷ lệ chọn đúng (\\%) trên bộ chiến thuật, cấu hình giải đấu (chế độ B)", "lrrrrrr",
          HEADER, accuracy_rows(base, [(n, n.replace("-", " ")) for n in AGENT_ORDER], positions))

    scale = chung.load_json("chien-thuat-scale")

    def pick(prefix: str, label: str) -> list[tuple[str, str]]:
        return [(k, f"{label}, {k.split('=')[1]}") for k in scale if k.startswith(prefix + "|")]

    table("ct-l3", "L3: tỷ lệ chọn đúng (\\%) theo độ sâu tìm kiếm", "lrrrrrr",
          HEADER, accuracy_rows(scale, pick("L3", "độ sâu"), positions))
    rows = []
    for i, (prefix, label) in enumerate((("L4-random", "ngẫu nhiên"), ("L4-light", "light"),
                                         ("L4-tactics", "light + gốc"))):
        if i:
            rows.append([])
        rows += accuracy_rows(scale, pick(prefix, label), positions)
    table("ct-l4", "MCTS: tỷ lệ chọn đúng (\\%) theo kiểu rollout và số mô phỏng", "lrrrrrr", HEADER, rows,
          size=r"\footnotesize")
    rows = accuracy_rows(scale, pick("L7", "L7, sims"), positions)
    rows.append([])
    rows += accuracy_rows(scale, pick("L8", "L8, sims"), positions)
    table("ct-puct", "PUCT: tỷ lệ chọn đúng (\\%) theo số mô phỏng", "lrrrrrr", HEADER, rows)

    variants = chung.load_json("chien-thuat-variants")

    def group(prefixes: list[str]) -> list[tuple[str, str]]:
        return [(k, k.replace("|", ", ").replace("_", r"\_")) for k in variants
                if any(k.startswith(p) for p in prefixes)]

    for name, caption, prefixes in (
        ("ct-cpuct", "Ảnh hưởng của $c_{puct}$ (200 mô phỏng), tỷ lệ đúng (\\%)", ["L7|c_puct", "L8|c_puct"]),
        ("ct-batch", "Ảnh hưởng của số lá gom mỗi lô (200 mô phỏng), tỷ lệ đúng (\\%)", ["L7|batch", "L8|batch"]),
        ("ct-mix", "L7: trộn giá trị mạng với rollout, tỷ lệ đúng (\\%)", ["L7|mix"]),
        ("ct-cuct", "L4 rollout light, 400 mô phỏng: ảnh hưởng của $c_{uct}$", ["L4|c_uct"]),
        ("ct-l2", "L2: tỷ lệ đúng (\\%) khi tắt lần lượt từng luật", ["L2|"]),
        ("ct-l3bien", "L3 độ sâu 4: tỷ lệ đúng (\\%) khi đổi từng tham số", ["L3|"]),
    ):
        table(name, caption, "lrrrrrr", HEADER, accuracy_rows(variants, group(prefixes), positions))

    # Độ khó theo thế: có bao nhiêu trong 8 tác tử giải đúng
    solved = collections.Counter()
    for name in AGENT_ORDER:
        for r in base[name]:
            solved[r["id"]] += r["ok"]
    rows = []
    for k in range(9):
        by_cat = collections.Counter(p["category"] for p in positions if solved[p["id"]] == k)
        rows.append([str(k), *[str(by_cat[c]) for c in CATS], str(sum(by_cat.values()))])
    table("ct-dokho", "Số thế theo số tác tử (trong tám) chọn đúng", "crrrrrr",
          ["Số tác tử đúng", *CATS, "Tổng"], rows)

    # Hai tác tử cùng đúng / một đúng một sai (McNemar)
    rows = []
    for a, b in (("L5-PolicySL", "L6-PolicyRL"), ("L3-AlphaBeta", "L7-PUCT"), ("L7-PUCT", "L8-AlphaZero"),
                 ("L5-PolicySL", "L7-PUCT"), ("L2-Rule", "L5-PolicySL")):
        x = {r["id"]: r["ok"] for r in base[a]}
        y = {r["id"]: r["ok"] for r in base[b]}
        only_a = sum(x[i] and not y[i] for i in x)
        only_b = sum(y[i] and not x[i] for i in x)
        from phan_tich_thong_ke import sign_test

        rows.append([f"{SHORT[a]} / {SHORT[b]}", str(sum(x[i] and y[i] for i in x)), str(only_a), str(only_b),
                     str(sum(not x[i] and not y[i] for i in x)), p_value(sign_test(only_a, only_b))])
    table("ct-mcnemar", "So sánh ghép cặp trên cùng 500 thế (kiểm định McNemar chính xác)", "lrrrrr",
          ["Cặp A / B", "Cả hai đúng", "Chỉ A", "Chỉ B", "Cả hai sai", "$p$"], rows)

    appendix_positions(positions, base)


def appendix_positions(positions: list[dict], base: dict) -> None:
    move = {name: {r["id"]: r for r in base[name]} for name in AGENT_ORDER}
    rows = []
    for p in positions:
        answers = [chung.cell_name(a) for a in p["answers"]]
        text = ", ".join(answers[:4]) + (f" (+{len(answers) - 4})" if len(answers) > 4 else "")
        marks = [r"$\bullet$" if move[n][p["id"]]["ok"] else r"$\circ$" for n in AGENT_ORDER]
        source = "E3" if "e3_" in p["source"] else "RR8"
        rows.append([p["id"], f"{source}/{p['game_id']}", str(p["ply"]), text, *marks])
    chung.longtable("ct-tatca", "Toàn bộ 500 thế: nguồn, đáp án và kết quả của tám tác tử "
                    r"($\bullet$ đúng, $\circ$ sai)", "llrp{3.9cm}cccccccc",
                    ["Mã thế", "Ván", "Nước", "Đáp án", *[SHORT[n] for n in AGENT_ORDER]], rows,
                    size=r"\scriptsize")


def sample_boards() -> None:
    suite = chung.load_json("bo-chien-thuat")["positions"]
    base = chung.load_json("chien-thuat-base")
    move = {name: {r["id"]: r for r in base[name]} for name in AGENT_ORDER}
    solved = {p["id"]: sum(move[n][p["id"]]["ok"] for n in AGENT_ORDER) for p in suite}
    for c in CATS:
        group = sorted((p for p in suite if p["category"] == c), key=lambda p: (solved[p["id"]], p["id"]))
        chosen = [group[0], group[len(group) // 4]]
        blocks = [r"\begin{figure}[H]", r"  \centering"]
        notes = []
        for p in chosen:
            x, o = chung.replay(p["moves"])
            numbers = {m: i + 1 for i, m in enumerate(p["moves"])}
            marks = {a: "green!30" for a in p["answers"]}
            side = "X" if p["ply"] % 2 == 0 else "O"
            blocks += [r"  \begin{subfigure}[t]{.48\linewidth}", r"    \centering",
                       chung.board_tikz(x, o, numbers, marks, unit=0.44),
                       rf"    \caption{{{p['id']}: {side} đi; {solved[p['id']]}/8 tác tử đúng}}",
                       r"  \end{subfigure}\hfill" if p is chosen[0] else r"  \end{subfigure}"]
            picks = collections.defaultdict(list)
            for n in AGENT_ORDER:
                if not move[n][p["id"]]["ok"]:
                    picks[chung.cell_name(move[n][p["id"]]["move"])].append(SHORT[n])
            wrong = "; ".join(f"{', '.join(v)} chọn {k}" for k, v in picks.items()) or "không tác tử nào sai"
            notes.append(f"{p['id']}: {wrong}")
        blocks += [(rf"  \caption{{Hai thế loại {c} (ô xanh là đáp án, số trên quân là thứ tự nước). "
                   f"{'. '.join(notes)}.}}"), rf"  \label{{fig:cs-the-{c.lower()}}}", r"\end{figure}", ""]
        (chung.TABLES / f"hinh-the-{c.lower()}.tex").write_text("\n".join(blocks), encoding="utf-8")


# ---------------------------------------------------------------- phân tích từng nước
def move_tables() -> None:
    labels = chung.load_json("phan-tich-van")["games"]
    records = {r["game_id"]: r for r in chung.games(chung.RUNS["rr8"])}
    counts = {n: collections.Counter() for n in AGENT_ORDER}
    ends = {n: collections.Counter() for n in AGENT_ORDER}
    missed_len = {n: collections.Counter() for n in AGENT_ORDER}
    seen_len = {n: collections.Counter() for n in AGENT_ORDER}
    first_vcf = collections.defaultdict(list)
    converted = {n: [0, 0] for n in AGENT_ORDER}
    by_ply = {n: collections.defaultdict(collections.Counter) for n in AGENT_ORDER}
    for g in labels:
        r = records[g["game_id"]]
        owner = (r["agent_x"], r["agent_o"])
        first = None
        for k, (label, length) in enumerate(zip(g["labels"], g["vcf_len"], strict=True)):
            ply = r["opening_plies"] + k
            name = owner[ply % 2]
            counts[name][label] += 1
            by_ply[name][min(ply // 8 * 8, 32)][label] += 1
            if label.startswith("vcf_"):
                bucket = min(length, 5)
                seen_len[name][bucket] += 1
                missed_len[name][bucket] += label == "vcf_missed"
            if first is None and (label.startswith(("vcf_", "win1_"))):
                first = (ply, ply % 2)
        if first is not None:
            name = owner[first[1]]
            winner = None if r["result"] == 0 else (0 if r["result"] == 1 else 1)
            converted[name][0] += winner == first[1]
            converted[name][1] += 1
            first_vcf[name].append((first[0], len(r["moves"]) - first[0]))
        if r["result"] != 0:
            loser = 1 if r["result"] == 1 else 0
            own = [lab for k, lab in enumerate(g["labels"]) if (r["opening_plies"] + k) % 2 == loser]
            ends[owner[loser]][own[-1] if own else "none"] += 1

    rows = []
    for n in AGENT_ORDER:
        c = counts[n]
        win1 = c["win1_taken"] + c["win1_missed"]
        block = c["block_done"] + c["block_missed"]
        forced = c["vcf_kept"] + c["vcf_missed"]
        quiet = c["safe_kept"] + c["open_blunder"] + c["open_forced"]
        rows.append([n.replace("-", " "), count(sum(c.values())),
                     f"{c['win1_missed']}/{win1}", f"{c['block_missed']}/{block}",
                     f"{c['vcf_missed']}/{forced}", f"{c['open_blunder']}/{quiet}",
                     num(100 * (c["win1_missed"] + c["block_missed"] + c["vcf_missed"] + c["open_blunder"])
                         / sum(c.values()), 2)])
    table("nuoc-loi", "Lỗi chiến thuật từng nước trong 2.800 ván RR8 giữ riêng", "lrrrrrr",
          ["Tác tử", "Nước", "Bỏ thắng", "Sót chặn", "Bỏ VCF", "Để hở VCF", "Lỗi/100"], rows,
          size=r"\footnotesize",
          note="Mẫu số của mỗi cột là số lượt mà tình huống đó xuất hiện. ``Để hở VCF'': không có đường thắng ép, "
               "còn nước an toàn nhưng chọn nước cho đối thủ có VCF; mẫu số là mọi lượt yên tĩnh.")

    rows = [[n.replace("-", " "), *[f"{missed_len[n][k]}/{seen_len[n][k]}" for k in (2, 3, 4, 5)]]
            for n in AGENT_ORDER]
    table("nuoc-vcf", "Số lần bỏ lỡ VCF theo độ dài (số nước tấn công)", "lrrrr",
          ["Tác tử", "2", "3", "4", "$\\ge 5$"], rows)

    rows = []
    for n in AGENT_ORDER:
        values = first_vcf[n]
        rows.append([n.replace("-", " "), str(len(values)),
                     percent(converted[n][0], converted[n][1]),
                     num(statistics.median(v[0] for v in values), 0) if values else "---",
                     num(statistics.median(v[1] for v in values), 0) if values else "---"])
    table("nuoc-dautien", "Ván mà tác tử là bên đầu tiên có đường thắng ép", "lrrrr",
          ["Tác tử", "Số ván", "Thắng (\\%)", "Nước (trung vị)", "Còn lại (trung vị)"], rows,
          note="Đường thắng ép gồm thắng ngay và VCF. ``Còn lại'' là số nước từ lúc đó đến hết ván.")

    keys = (("lost", "Đòn kép"), ("block_missed", "Sót chặn"),
            ("win1_missed", "Bỏ thắng"), ("open_forced", "Hết nước đỡ"))
    rows = []
    for n in AGENT_ORDER:
        total = sum(ends[n].values())
        known = sum(ends[n][k] for k, _ in keys)
        rows.append([n.replace("-", " "), str(total), *[str(ends[n][k]) for k, _ in keys], str(total - known)])
    table("nuoc-ketthuc", "Nước cuối cùng của bên thua trong các ván có thắng thua", "lrrrrrr",
          ["Bên thua", "Ván", *[k for _, k in keys], "Khác"], rows, size=r"\footnotesize",
          note="``Đòn kép'': trước nước cuối của bên thua, đối thủ đã có từ hai ô thắng. ``Sót chặn'': còn một ô "
               "thắng mà không chặn. ``Bỏ thắng'': bên thua có nước thắng ngay nhưng đi chỗ khác. ``Hết nước đỡ'': "
               "không nước nào ngăn được VCF.")

    rows = []
    for n in AGENT_ORDER:
        cells = []
        for bucket in (0, 8, 16, 24, 32):
            c = by_ply[n][bucket]
            errors = c["win1_missed"] + c["block_missed"] + c["vcf_missed"] + c["open_blunder"]
            cells.append(percent(errors, sum(c.values())))
        rows.append([n.replace("-", " "), *cells])
    table("nuoc-giaidoan", "Tỷ lệ nước mắc lỗi chiến thuật (\\%) theo giai đoạn ván", "lrrrrr",
          ["Tác tử", "Nước 1--8", "9--16", "17--24", "25--32", "$\\ge 33$"], rows)


# ---------------------------------------------------------------- thống kê
def stat_tables() -> None:
    data = chung.load_json("phan-tich-thong-ke")
    rows = []
    for p in data["pairs"]:
        lo, hi = p["boot"]
        rows.append([f"{SHORT[p['a']]}--{SHORT[p['b']]}", f"{p['w']}/{p['d']}/{p['l']}", num(100 * p["score"]),
                     f"[{num(100 * lo, 0)}; {num(100 * hi, 0)}]", num(100 * p["as_x"], 0), num(100 * p["as_o"], 0),
                     f"{p['openings_better']}/{p['openings_even']}/{p['openings_worse']}", p_value(p["p_holm"]),
                     num(p["length_median"], 0)])
    chung.longtable("tk-capdau", "28 cặp đấu của RR8 giữ riêng, nhìn từ tác tử đứng trước", "lcrcrrcrr",
                    ["Cặp", "T/H/B", "Điểm", "KTC 95\\%", "Khi X", "Khi O", "KC", "$p$ Holm", "Dài"], rows)

    ratings = data["ratings"]
    priors = ratings["priors"]
    rows = [[n.replace("-", " "), *[num(priors[k][n], 0) for k in priors]] for n in AGENT_ORDER]
    table("tk-prior", "Điểm Bradley-Terry theo số ván hoà ảo mỗi cặp", "lrrrrr",
          ["Tác tử", *[num(float(k), 1) for k in priors]], rows)

    rows = []
    for n in AGENT_ORDER:
        s16, s32 = ratings["elo_shuffled"]["16"][n], ratings["elo_shuffled"]["32"][n]
        rows.append([n.replace("-", " "), num(ratings["base"][n], 0), num(ratings["elo_sequential"][n], 0),
                     f"{num(s16['mean'], 0)} $\\pm$ {num(s16['std'], 0)}",
                     f"{num(s32['mean'], 0)} $\\pm$ {num(s32['std'], 0)}",
                     f"{num(s32['min'], 0)}--{num(s32['max'], 0)}"])
    table("tk-elo", "Elo tuần tự phụ thuộc thứ tự ván; Bradley-Terry thì không", "lrrrrr",
          ["Tác tử", "B-T", "Elo theo log", "K=16 (200 hoán vị)", "K=32", "Khoảng K=32"], rows,
          size=r"\footnotesize")

    rows = []
    for c in ratings["convergence"]:
        rows.append([str(c["openings"]), str(c["openings"] * 2),
                     *[num(c["agents"][n][2] - c["agents"][n][0], 0) for n in AGENT_ORDER]])
    table("tk-hoitu", "Độ rộng khoảng bootstrap 95\\% của điểm Bradley-Terry theo số khai cuộc", "rrrrrrrrrr",
          ["Khai cuộc", "Ván/cặp", *[SHORT[n] for n in AGENT_ORDER]], rows, size=r"\footnotesize")

    fit = data["fit"]
    worst = sorted(fit["pairs"], key=lambda r: -abs(r["residual"]))[:10]
    rows = [[f"{SHORT[r['a']]}--{SHORT[r['b']]}", num(100 * r["observed"]), num(100 * r["predicted"]),
             num(100 * r["residual"])] for r in worst]
    table("tk-phandu", "Mười cặp lệch nhiều nhất so với mô hình Bradley-Terry (điểm phần trăm)", "lrrr",
          ["Cặp", "Quan sát", "Mô hình", "Phần dư"], rows)

    rows = [[num(r["p1"], 2), num(400 * __import__("math").log10(r["p1"] / (1 - r["p1"])), 0),
             count(r["0.8"]), count(r["0.9"])] for r in data["power"]]
    table("tk-luc", "Số ván độc lập cần để phát hiện chênh lệch ($\\alpha = 0{,}05$ hai phía)", "rrrr",
          ["Điểm thật", "Chênh Elo", "Lực 80\\%", "Lực 90\\%"], rows)

    rows = []
    for o in data["openings"]:
        rows.append([str(o["id"]), " ".join(chung.cell_name(m) for m in o["moves"]), str(o["n"]),
                     num(100 * o["x_score"]), str(o["draws"]), num(o["length"])])
    chung.longtable("tk-khaicuoc", "Kết quả theo từng khai cuộc giữ riêng được dùng trong RR8", "rlrrrr",
                    ["Mã", "Các nước", "Ván", "Điểm X (\\%)", "Hoà", "Độ dài TB"], rows)

    board = data["board"]["by_ply"]
    plies = (0, 8, 16, 24, 32, 40)
    rows = []
    for n in AGENT_ORDER:
        lookup = {b["ply"]: b for b in board[n]}
        rows.append([n.replace("-", " "), *[num(lookup[p]["time_median"], 1) if p in lookup else "---"
                                            for p in plies]])
    table("tk-thoigian", "Trung vị thời gian mỗi nước (ms) theo giai đoạn trong RR8 giữ riêng", "lrrrrrr",
          ["Tác tử", *[f"từ {p}" for p in plies]], rows)

    rows = []
    for n in AGENT_ORDER:
        total = sum(b["n"] for b in board[n])
        distance = sum(b["distance"] * b["n"] for b in board[n]) / total
        adjacent = sum(b["adjacent"] * b["n"] for b in board[n]) / total
        usage = data["board"]["every"][n]
        center = sum(usage[r * 11 + c] for r in range(3, 8) for c in range(3, 8)) / sum(usage)
        edge = sum(usage[r * 11 + c] for r in range(11) for c in range(11) if r in (0, 10) or c in (0, 10))
        rows.append([n.replace("-", " "), num(distance, 2), num(100 * adjacent), num(100 * center),
                     num(100 * edge / sum(usage))])
    table("tk-khonggian", "Phân bố không gian của nước đi", "lrrrr",
          ["Tác tử", "Cách nước trước", "Kề nước trước (\\%)", "Vùng 5$\\times$5 (\\%)", "Ô biên (\\%)"], rows,
          note="Khoảng cách Chebyshev tới nước vừa đi của đối thủ.")


# ---------------------------------------------------------------- mạng
def net_tables() -> None:
    data = chung.load_json("phan-tich-mang")
    names = list(data["models"])
    d = data["dataset"]
    table("mang-dulieu", "Thống kê tập dữ liệu expert\\_v2 sau khi loại trùng", "lr", ["Đại lượng", "Giá trị"], [
        ["Số thế", count(d["samples"])], ["Số ván", count(d["games"])],
        ["Train / validation / test", " / ".join(count(d["split"][k]) for k in ("train", "val", "test"))],
        ["Thế X đi", percent(d["x_to_move"], d["samples"]) + "\\%"],
        ["Thế thuộc ván X thắng / hoà / O thắng (nhãn theo bên đi)",
         " / ".join(percent(d["value_counts"][k], d["samples"]) for k in ("1", "0", "-1")) + "\\%"],
        ["Số nước có xác suất khác 0 trong nhãn policy (TB)", num(d["target_support_mean"], 2)],
        ["Xác suất lớn nhất trong nhãn policy (TB)", num(d["target_max_mean"], 3)],
        ["Entropy nhãn policy (nat, TB)", num(d["target_entropy_mean"], 3)]])

    rows = []
    for n in names:
        m = data["models"][n]["all"]
        rows.append([n, num(100 * m["top1"]), num(100 * m["top3"]), num(100 * m["top5"]), num(m["entropy"], 2),
                     num(m["mse"], 3), num(100 * m["sign"])])
    table("mang-tong", "Bốn mạng trên cùng tập test", "lrrrrrr",
          ["Mạng", "Top-1", "Top-3", "Top-5", "Entropy", "MSE $v$", "Đúng dấu"], rows, size=r"\footnotesize",
          note="Top-$k$ đo mức trùng với nước tốt nhất của expert L3; L6 và L8 không được huấn luyện để bắt "
               "chước expert nên cột này không phải thước đo sức mạnh của chúng.")

    for key, name, caption, fmt in (
        ("top1", "mang-giaidoan-top1", "Top-1 (\\%) theo số quân trên bàn", lambda v: num(100 * v)),
        ("mse", "mang-giaidoan-mse", "MSE của value theo số quân trên bàn", lambda v: num(v, 3)),
        ("entropy", "mang-giaidoan-entropy", "Entropy policy (nat) theo số quân trên bàn", lambda v: num(v, 2)),
    ):
        buckets = data["models"][names[0]]["by_stones"]
        header = ["Số quân", "Số thế", *[n.split(" (")[0] for n in names]]
        rows = []
        for i, b in enumerate(buckets):
            label = f"{b['lo']}--{b['hi'] - 1}" if b["hi"] < 122 else f"$\\ge {b['lo']}$"
            rows.append([label, count(b["n"]), *[fmt(data["models"][n]["by_stones"][i][key]) for n in names]])
        table(name, caption, "lrrrrr", header, rows, size=r"\footnotesize")

    rows = []
    for i, b in enumerate(data["models"][names[0]]["calibration"]):
        cells = []
        for n in names:
            c = data["models"][n]["calibration"][i]
            cells.append("---" if not c["n"] else f"{num(c['z'], 2)} ({count(c['n'])})")
        rows.append([f"[{num(b['lo'], 1)}; {num(b['hi'], 1)})", *cells])
    table("mang-calib", "Kết quả trung bình thực tế $\\bar z$ (số thế) theo khoảng giá trị mạng dự đoán", "lrrrr",
          ["Khoảng $v$", *[n.split(" (")[0] for n in names]], rows, size=r"\footnotesize")

    rows = []
    for n in names:
        s = data["models"][n]["symmetry"]
        rows.append([n, num(s["tv"], 3), num(100 * s["top1_agree"]), num(s["value_std"], 3),
                     num(s["value_range"], 3), num(100 * s["top1_single"]), num(100 * s["top1_ensemble"])])
    table("mang-doixung", "Độ nhất quán dưới 8 phép đối xứng D4 (4.000 thế test)", "lrrrrrr",
          ["Mạng", "TV", "Trùng top-1", "$\\sigma_v$", "Biên độ $v$", "Top-1 đơn", "Top-1 TB 8"], rows,
          size=r"\footnotesize",
          note="TV: khoảng cách biến phân toàn phần giữa policy gốc và policy của thế đã biến đổi (đưa về hệ toạ độ "
               "gốc). Mạng đẳng biến hoàn hảo có TV bằng 0 và trùng top-1 100\\%.")

    for key, name, caption in (
        ("mass", "mang-ct-khoiluong", "Khối lượng policy (\\%) đặt lên các đáp án của bộ chiến thuật"),
        ("top1", "mang-ct-top1", "Nước policy cao nhất đúng đáp án (\\%) trên bộ chiến thuật"),
        ("value", "mang-ct-value", "Giá trị $v$ trung bình của mạng trên từng loại thế"),
    ):
        rows = []
        for n in names:
            t = data["models"][n]["tactics"]
            rows.append([n, *[num(t[c]["value"], 2) if key == "value" else num(100 * t[c][key]) for c in CATS]])
        table(name, caption, "lrrrrr", ["Mạng", *CATS], rows)

    groups = collections.OrderedDict()
    for layer in data["layers"]:
        parts = layer["name"].split(".")
        head = f"{parts[0]}.{parts[1]}" if parts[0] == "res_blocks" else parts[0]
        groups[head] = groups.get(head, 0) + layer["params"]
    total = sum(groups.values())
    rows = [[rf"\texttt{{{k.replace('_', chr(92) + '_')}}}", count(v), percent(v, total)] for k, v in groups.items()]
    rows += [[], ["Tổng", count(total), "100,0"]]
    table("mang-thamso", "Phân bố tham số của CaroResNet (64 kênh, 6 khối)", "lrr",
          ["Khối", "Tham số", "\\%"], rows, size=r"\footnotesize")


# ---------------------------------------------------------------- dữ liệu huấn luyện
def data_tables() -> None:
    data = chung.load_json("phan-tich-du-lieu")
    v1, v2 = data["expert_v1"], data["expert_v2"]

    def pct(v: float) -> str:
        return num(100 * v) + r"\%"

    table("dl-hoadiem", "Nhãn policy hoà điểm trong hai tập dữ liệu chuyên gia", "lrr",
          ["Đại lượng", "expert\\_v1", "expert\\_v2"], [
              [label, fmt(v1[key]), fmt(v2[key])] for label, key, fmt in (
                  ("Số mẫu", "samples", count), ("Mẫu có nhãn hoà điểm", "tie", count),
                  ("Mẫu ở thế đã thua", "lost", count), ("Thế đã thua có nhãn hoà", "tie_given_lost", pct),
                  ("Nước nhãn cao nhất ở hàng 1: toàn tập", "row1_all", pct),
                  ("\\quad trong mẫu hoà", "row1_tie", pct), ("\\quad trong mẫu không hoà", "row1_clean", pct),
                  ("\\quad trong thế đã thua", "row1_lost", pct),
                  ("Nhãn hàng 1 đến từ mẫu hoà", "tie_given_row1", pct),
                  ("Ván không còn mẫu sau loại trùng", "games_empty", count),
                  ("Ván còn không quá 2 mẫu", "games_le2", count))],
          note="Hoà điểm: nhãn đặt xác suất bằng nhau lên từ hai nước trở lên. Thế đã thua: đối thủ có từ hai ô "
               "thắng, bên đi không có ô thắng.")
    rows = []
    for a, b in zip(v1["by_stones"], v2["by_stones"], strict=True):
        label = f"{a['lo']}--{a['hi'] - 1}" if a["hi"] < 122 else f"$\\ge {a['lo']}$"
        rows.append([label, count(a["n"]), pct(a["tie"]), pct(a["draw"]), count(b["n"]), pct(b["tie"]),
                     pct(b["lost"]), pct(b["draw"])])
    table("dl-theoquan", "Tỷ lệ nhãn hoà theo số quân trên bàn", "lrrrrrrr",
          ["Quân", "v1: mẫu", "hoà", "ván hoà", "v2: mẫu", "hoà", "đã thua", "ván hoà"], rows,
          size=r"\footnotesize")
    w = data["immediate_win"]
    table("dl-thangngay", "Nhãn ở các thế bên đi thắng ngay được (expert\\_v2)", "lr", ["Đại lượng", "Giá trị"], [
        ["Số mẫu (tỷ lệ trong tập)", f"{count(w['n'])} ({pct(w['share'])})"],
        ["Xác suất nhãn đặt trên các ô thắng (trung bình)", pct(w["label_mass"])],
        ["Mẫu đặt dưới 50\\% lên các ô thắng", pct(w["mass_below_half"])],
        ["Nước nhãn cao nhất là ô thắng", pct(w["label_argmax_wins"])],
        ["Mẫu chỉ có đúng một ô thắng", pct(w["single_winner"])]])
    rows = []
    for name, m in data["models"].items():
        rows.append([name, *[num(100 * m[k]["top1"]) for k in ("all", "clean", "tie", "lost")],
                     *[num(100 * m[k]["row1_mass"]) for k in ("clean", "tie", "lost")]])
    n = next(iter(data["models"].values()))
    table("dl-mang", "Top-1 (\\%) và khối lượng policy ở hàng 1 (\\%) theo nhóm mẫu test", "lrrrrrrr",
          ["Mạng", "Tất cả", "Không hoà", "Hoà", "Đã thua", "H1: không hoà", "H1: hoà", "H1: thua"], rows,
          size=r"\scriptsize",
          note=f"Cỡ nhóm: tất cả {count(n['all']['n'])}, không hoà {count(n['clean']['n'])}, hoà "
               f"{count(n['tie']['n'])}, đã thua {count(n['lost']['n'])}.")
    rows = [[name, count(m["immediate_win"]["n"]), num(100 * m["immediate_win"]["top1"]),
             num(100 * m["immediate_win"]["mass"])] for name, m in data["models"].items()]
    table("dl-mang-thang", "Mạng trên các thế test thắng ngay được", "lrrr",
          ["Mạng", "Mẫu", "Argmax là ô thắng (\\%)", "Khối lượng ô thắng (\\%)"], rows)


def root_table() -> None:
    rows = []
    for r in chung.load_json("kiem-tra-goc"):
        saved = 1 - r["nodes_fixed"] / r["nodes_current"]
        rows.append([str(r["depth"]), f"{r['previous_first']}/{r['passes']}", (f"{r['rank_median'] + 1} / "
                     f"{r['rank_max'] + 1}"), count(r["nodes_current"]), count(r["nodes_fixed"]), num(100 * saved),
                     f"{r['same_score']}/{r['positions']}", f"{r['same_move']}/{r['positions']}"])
    table("goc", "Duyệt nước tốt nhất của độ sâu trước ở gốc: mã hiện tại so với bản đưa nước đó lên đầu",
          "rrrrrrrr", ["Độ sâu", "Đứng đầu", "Vị trí TV / max", "Nút hiện tại", "Nút sửa", "Giảm (\\%)",
                       "Cùng điểm", "Cùng nước"], rows, size=r"\footnotesize",
          note="50 thế trung cuộc, backend Python. ``Đứng đầu'' đếm trên mọi lượt tìm ở gốc từ độ sâu 2.")


def rollout_table() -> None:
    s = chung.load_json("kiem-tra-rollout")["summary"]
    budgets = list(s["budgets"])
    rows = [[label, *[fmt(s["budgets"][b]) for b in budgets]] for label, fmt in (
        ("Chọn đúng nước chặn", lambda r: f"{r['blocked']}/{s['positions']}"),
        ("Điểm $Q$ của nước sai được chọn (TB)", lambda r: num(r["missed_score"], 2)),
        ("Lượt thăm nút đó (TB)", lambda r: num(r["missed_visits"])),
        ("Đáp trả đã mở rộng / tổng (TB)", lambda r: f"{num(r['missed_expanded'])} / {num(r['missed_replies'])}"),
        ("Đã mở rộng nước thắng của đối thủ", lambda r: f"{r['missed_found_win']}/{r['missed']}"))]
    table("ct-rollout", "MCTS rollout light ở 100 thế phải chặn: cây thấy gì khi chọn sai", "l" + "r" * len(budgets),
          ["Đại lượng", *[f"{count(int(b))} mô phỏng" for b in budgets]], rows,
          note=f"Rollout light chạy trực tiếp ({s['rollouts']} lần mỗi nước): sau nước chặn trung bình "
               f"{num(s['block_value'], 2)}, sau một nước không chặn trung bình {num(s['other_value'], 2)}.")


def sensitivity_tables() -> None:
    import numpy as np
    from phan_tich_thong_ke import sign_test

    sims = [r for r in chung.games(chung.RUNS["l8_sims"]) if r["game_id"] < 400]
    cpuct = chung.games(chung.RUNS["cpuct"])
    assert len(sims) == 400 and len(cpuct) == 400, (len(sims), len(cpuct))

    def row(records: list[dict], agent: str, label: str) -> list[str]:
        games = [r for r in records if agent in (r["agent_x"], r["agent_o"])]
        points, by_opening = [], collections.defaultdict(list)
        for r in games:
            x = r["agent_x"] == agent
            s = (r["result"] * (1 if x else -1) + 1) / 2
            points.append((s, x))
            by_opening[r["opening_id"]].append(s)
        per = np.array([np.mean(v) for _, v in sorted(by_opening.items())])
        rng = np.random.default_rng(chung.SEED)
        boot = per[rng.integers(0, len(per), (10_000, len(per)))].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        as_x = [s for s, x in points if x]
        as_o = [s for s, x in points if not x]
        return [label, (f"{sum(s == 1 for s, _ in points)}/{sum(s == .5 for s, _ in points)}/"
                f"{sum(s == 0 for s, _ in points)}"), num(100 * np.mean([s for s, _ in points])),
                f"[{num(100 * lo, 0)}; {num(100 * hi, 0)}]", num(100 * np.mean(as_x), 0),
                num(100 * np.mean(as_o), 0), p_value(sign_test(int((per > .5).sum()), int((per < .5).sum()))),
                num(statistics.median(len(r["moves"]) for r in games), 0)]

    header = ["Cấu hình L8", "T/H/B", "Điểm", "KTC 95\\%", "Khi X", "Khi O", "$p$", "Dài"]
    note = ("Mỗi dòng 100 ván gặp L7 (200 mô phỏng, $c_{puct}=1{,}5$) trên 50 khai cuộc giữ riêng đổi màu. $p$: "
            "kiểm định dấu theo khai cuộc, chưa hiệu chỉnh so sánh nhiều lần.")
    table("dn-sims", "Sức chơi của L8 theo số mô phỏng, đối thủ cố định L7", "lcrcrrrr", header,
          [row(sims, f"L8-sims{s}", f"{count(s)} mô phỏng") for s in (1, 50, 200, 800)],
          size=r"\footnotesize", note=note)
    rows = [row(cpuct, f"L8-c{c}", f"$c_{{puct}}={c.replace('.', '{,}')}$") for c in ("0.5", "1.0")]
    rows.append(row(sims, "L8-sims200", "$c_{puct}=1{,}5$"))
    rows += [row(cpuct, f"L8-c{c}", f"$c_{{puct}}={c.replace('.', '{,}')}$") for c in ("3.0", "5.0")]
    table("dn-cpuct", "Sức chơi của L8 (200 mô phỏng) theo $c_{puct}$, đối thủ cố định L7", "lcrcrrrr", header,
          rows, size=r"\footnotesize", note=note)

    reference = {(r["opening_id"], r["agent_x"][:2]): r for r in chung.games(chung.RUNS["rr8"])
                 if {r["agent_x"], r["agent_o"]} == {"L7-PUCT", "L8-AlphaZero"}}
    repeat = [r for r in sims if "L8-sims200" in (r["agent_x"], r["agent_o"])]
    same = sum(reference[r["opening_id"], r["agent_x"][:2]]["moves"] == r["moves"] for r in repeat)
    chung.save_json("do-nhay-tai-lap", {"games": len(repeat), "same_moves": same})
    table("dn-tailap", "Tái lập cặp L7--L8 của RR8 giữ riêng bằng giải thử độ nhạy", "lr", ["Đại lượng", "Giá trị"],
          [["Ván của mức 200 mô phỏng", str(len(repeat))], ["Trùng từng nước với RR8", f"{same}/{len(repeat)}"]])


# ---------------------------------------------------------------- tìm kiếm
def search_tables() -> None:
    data = chung.load_json("phan-tich-tim-kiem")
    table("tk-vi-mo", "Thời gian một lần gọi các phép toán cốt lõi (Python thuần, một luồng)", "lr",
          ["Phép toán", "µs"],
          [[m["name"] if "\\" in m["name"] else m["name"].replace("_", r"\_"), num(m["us"], 2)]
           for m in data["micro"]],
          note=f"Lấy lần lặp nhanh nhất trong 5 lần, trung bình trên {data['positions']} thế trung cuộc.")

    depths = data["alphabeta"]
    final = [r["move"] for r in depths[-1]["rows"]]
    rows, previous = [], None
    for d in depths:
        nodes = [r["nodes"] for r in d["rows"]]
        times = sorted(r["time_ms"] for r in d["rows"])
        mean = statistics.mean(nodes)
        same = sum(r["move"] == f for r, f in zip(d["rows"], final, strict=True))
        rows.append([str(d["depth"]), count(mean), count(statistics.median(nodes)), count(max(nodes)),
                     "---" if previous is None else num(mean / previous, 2), num(statistics.median(times)),
                     num(times[int(0.95 * (len(times) - 1))]), percent(same, len(final), 0)])
        previous = mean
    table("tk-dosau", "L3 (lõi Cython, đủ tính năng) theo độ sâu trên 50 thế trung cuộc", "rrrrrrrr",
          ["Độ sâu", "Nút TB", "Trung vị", "Max", "Tỷ số", "ms P50", "ms P95", "Trùng d=6"], rows,
          size=r"\footnotesize", note="Nút và thời gian tính cho cả chuỗi lặp sâu dần tới độ sâu đó.")

    ab = {(r["backend"], r["depth"]): r for r in data["backends"] if r["kind"] == "alphabeta"}
    rows = []
    for depth in (2, 3, 4):
        py, nat = ab["python", depth], ab["native", depth]
        rows.append([str(depth), num(statistics.median(py["time_ms"])), num(statistics.median(nat["time_ms"]), 2),
                     num(sum(py["time_ms"]) / sum(nat["time_ms"])),
                     f"{sum(a == b for a, b in zip(py['nodes'], nat['nodes'], strict=True))}/20",
                     f"{sum(a == b for a, b in zip(py['moves'], nat['moves'], strict=True))}/20"])
    table("tk-backend-ab", "L3: Python thuần so với lõi Cython trên 20 thế", "rrrrrr",
          ["Độ sâu", "Python P50 (ms)", "Cython P50 (ms)", "Tăng tốc", "Cùng số nút", "Cùng nước"], rows)

    puct = [r for r in data["backends"] if r["kind"] == "puct"]
    rows = []
    for name in ("L7-PUCT", "L8-AlphaZero"):
        reference = next(r for r in puct if r["agent"] == name and r["backend"] == "native" and r["batch"] == 1)
        for batch in (1, 4, 8, 16, 32):
            py = next(r for r in puct if r["agent"] == name and r["backend"] == "python" and r["batch"] == batch)
            nat = next(r for r in puct if r["agent"] == name and r["backend"] == "native" and r["batch"] == batch)
            rows.append([SHORT[name], str(batch), num(statistics.median(py["time_ms"])),
                         num(statistics.median(nat["time_ms"])), num(sum(py["time_ms"]) / sum(nat["time_ms"]), 2),
                         f"{sum(a == b for a, b in zip(nat['moves'], reference['moves'], strict=True))}/20"])
        if name == "L7-PUCT":
            rows.append([])
    table("tk-backend-puct", "PUCT 200 mô phỏng: backend và số lá mỗi lô, 20 thế", "lrrrrr",
          ["Tác tử", "Lô", "Python P50 (ms)", "Cython P50 (ms)", "Tăng tốc", "Cùng nước với lô 1"], rows,
          size=r"\footnotesize")

    rows = []
    for rollout in ("random", "light"):
        for r in (r for r in data["backends"] if r["kind"] == "mcts" and r["rollout"] == rollout):
            median = statistics.median(r["time_ms"])
            rows.append([rollout, count(r["sims"]), num(median), count(r["sims"] / median * 1000),
                         num(median / r["sims"] * 1000, 0)])
    table("tk-mcts", "Thông lượng MCTS (Python) theo kiểu rollout", "lrrrr",
          ["Rollout", "Mô phỏng", "P50 (ms)", "Mô phỏng/s", "µs/mô phỏng"], rows)

    batches = sorted({r["batch"] for r in data["throughput"]})
    threads = sorted({r["threads"] for r in data["throughput"]})
    look = {(r["threads"], r["batch"]): r for r in data["throughput"]}
    rows = [[str(b), *[f"{num(look[t, b]['ms'], 2)} / {count(look[t, b]['per_second'])}" for t in threads]]
            for b in batches]
    table("tk-thongluong", "Suy luận mạng trên CPU: ms mỗi lô / số thế mỗi giây", "r" + "r" * len(threads),
          ["Lô", *[f"{t} luồng" for t in threads]], rows)

    for entry in data["heatmaps"]:
        pid = entry["position"]["id"]
        rows = [[m["label"], count(m["sims"]), chung.cell_name(m["move"]),
                 percent(m["answer_share"], 1), "---" if m["score"] is None else num(m["score"], 2),
                 " ".join(chung.cell_name(c) for c in m["pv"][:5])] for m in entry["maps"]]
        table(f"tk-nhiet-{pid.lower()}", f"Tìm kiếm tại thế {pid}", "lrcrrl",
              ["Cấu hình", "Mô phỏng", "Chọn", "Thăm đáp án (\\%)", "Q", "Biến chính"], rows, size=r"\footnotesize")


GROUPS = (("bo-chien-thuat", suite_tables), ("chien-thuat-variants", tactics_tables),
          ("chien-thuat-variants", sample_boards), ("phan-tich-van", move_tables),
          ("phan-tich-thong-ke", stat_tables), ("phan-tich-mang", net_tables),
          ("phan-tich-du-lieu", data_tables), ("kiem-tra-goc", root_table), ("kiem-tra-rollout", rollout_table), ("phan-tich-van", sensitivity_tables),
          ("phan-tich-tim-kiem", search_tables))


def evidence_tables() -> None:
    import json

    rows, inputs = [], {}
    for meta_path in sorted((chung.ROOT / "runs" / "chuyen_sau").glob("*/*/meta.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        stamp = meta_path.parent.name
        rows.append([rf"\texttt{{{m['step']}}}", stamp, num(m["duration_s"]), str(m["exit_code"]),
                     (m["git"]["commit"] or "")[:7] + ("*" if m["git"]["dirty"] else "")])
        for item in m.get("inputs", []):
            if "sha256" in item:
                inputs[item["path"]] = (item["bytes"], item["sha256"])
    chung.longtable("minh-chung", "Các lượt chạy được ghi trong runs/chuyen\\_sau (* là cây làm việc chưa commit)",
                    "llrrl", ["Bước", "Thời điểm", "Giây", "Mã thoát", "Commit"], rows)
    rows = [[rf"\texttt{{{path.replace('_', chr(92) + '_')}}}", count(size), rf"\texttt{{{sha[:16]}}}"]
            for path, (size, sha) in sorted(inputs.items())]
    chung.longtable("minh-chung-vao", "Tệp đầu vào và 16 ký tự đầu của SHA-256", "p{9cm}rl",
                    ["Tệp", "Byte", "SHA-256"], rows)


def main() -> None:
    for required, build in GROUPS:
        if (chung.DATA / f"{required}.json").is_file():
            build()
            print("xong", build.__name__)
        else:
            print("chưa có dữ liệu", required, "→ bỏ qua", build.__name__)
    evidence_tables()


if __name__ == "__main__":
    main()
