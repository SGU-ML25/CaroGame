"""Vẽ mọi hình report/chuyen-sau/hinh/* từ report/chuyen-sau/data/*.json.

    uv run --no-sync --with matplotlib python report/scripts/chuyen_sau/ve_hinh.py

Mỗi hình dùng tối đa bốn chuỗi trên một trục; chuỗi phân biệt bằng cả màu, nét và dấu để in trắng đen vẫn đọc được.
"""

from __future__ import annotations

import collections
import statistics

import chung
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from chung import AGENT_ORDER, SHORT
from matplotlib.colors import LinearSegmentedColormap

COLORS = ("#257de0", "#f47230", "#00b894", "#6a4ce2")
STYLES = (("-", "o"), ("--", "s"), ("-.", "^"), (":", "D"))
BLUES = LinearSegmentedColormap.from_list("xanh", ["#f4f8fd", "#9cc3f0", "#257de0", "#0d3b70"])
CATS = ("C1", "C2", "C3", "C4", "C5")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.edgecolor": "#888888",
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    fig.savefig(chung.FIGURES / f"{name}.png")
    plt.close(fig)


def series(ax, x, y, i: int, label: str, **kwargs) -> None:
    line, marker = STYLES[i]
    ax.plot(x, y, color=COLORS[i], linestyle=line, marker=marker, markersize=4, linewidth=1.8, label=label,
            **kwargs)


def board_heat(ax, values, title: str, vmax=None, annotate: float | None = None):
    grid = np.asarray(values, dtype=float).reshape(11, 11)
    image = ax.imshow(grid, cmap=BLUES, vmin=0, vmax=vmax)
    ax.set_xticks(range(11), "ABCDEFGHIJK", fontsize=7)
    ax.set_yticks(range(11), range(1, 12), fontsize=7)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10)
    return image


# ---------------------------------------------------------------- bộ chiến thuật
def tactics_figures() -> None:
    positions = chung.load_json("bo-chien-thuat")["positions"]
    category = {p["id"]: p["category"] for p in positions}
    scale = chung.load_json("chien-thuat-scale")

    def accuracy(key: str, c: str | None = None) -> float:
        rows = [r for r in scale[key] if c is None or category[r["id"]] == c]
        return 100 * sum(r["ok"] for r in rows) / len(rows)

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.0), sharey=True)
    panels = [*CATS, None]
    for ax, c in zip(axes.flat, panels, strict=True):
        for i, (prefix, label, sims) in enumerate((
                ("L4-random", "L4 ngẫu nhiên", (100, 400, 800, 1600, 6400)),
                ("L4-light", "L4 light", (100, 400, 800, 1600, 6400)),
                ("L7", "L7", (1, 50, 200, 800, 3200)), ("L8", "L8", (1, 50, 200, 800, 3200)))):
            series(ax, sims, [accuracy(f"{prefix}|sims={s}", c) for s in sims], i, label)
        ax.set_xscale("log")
        ax.set_title(c or "Cả 500 thế", fontsize=10)
        ax.set_ylim(-3, 103)
    for ax in axes[1]:
        ax.set_xlabel("số mô phỏng (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("đúng (%)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    save(fig, "ct-scale")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    depths = (1, 2, 3, 4, 5, 6)
    for i, c in enumerate(("C3", "C4", "C5")):
        values = [accuracy(f"L3|depth={d}", c) for d in depths]
        series(ax, depths, values, i, c)
        ax.annotate(c, (6, values[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    series(ax, depths, [accuracy(f"L3|depth={d}") for d in depths], 3, "Tổng")
    ax.set_xlabel("độ sâu L3")
    ax.set_ylabel("đúng (%)")
    ax.legend(ncol=4, loc="lower right")
    save(fig, "ct-l3-dosau")


# ---------------------------------------------------------------- phân tích ván
def game_figures() -> None:
    data = chung.load_json("phan-tich-thong-ke")
    board = data["board"]
    for key, name, label in (("first", "van-nuoc-dau", "nước đầu tiên"), ("every", "van-moi-nuoc", "mọi nước")):
        fig, axes = plt.subplots(2, 4, figsize=(8.0, 4.4))
        for ax, agent in zip(axes.flat, AGENT_ORDER, strict=True):
            counts = np.array(board[key][agent], dtype=float)
            share = 100 * counts / counts.sum()
            image = board_heat(ax, share, f"{SHORT[agent]} (max {share.max():.1f}%)")
            image.set_clim(0, share.max())
        fig.suptitle(f"Phân bố {label} của từng tác tử (mỗi ô vẽ theo giá trị lớn nhất của tác tử đó)", fontsize=10)
        fig.tight_layout()
        save(fig, name)

    records = chung.games(chung.RUNS["rr8"])
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 3.8), sharex=True, sharey=True)
    for ax, agent in zip(axes.flat, AGENT_ORDER, strict=True):
        lengths = [len(r["moves"]) for r in records if agent in (r["agent_x"], r["agent_o"])]
        ax.hist(lengths, bins=range(5, 126, 4), color=COLORS[0], edgecolor="white", linewidth=0.5)
        ax.set_title(f"{SHORT[agent]} (trung vị {statistics.median(lengths):.0f})", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("số nước")
    for ax in axes[:, 0]:
        ax.set_ylabel("số ván")
    fig.tight_layout()
    save(fig, "van-do-dai")

    length = collections.defaultdict(list)
    for r in records:
        length[r["agent_x"], r["agent_o"]].append(len(r["moves"]))
    grid = np.full((8, 8), np.nan)
    for i, a in enumerate(AGENT_ORDER):
        for j, b in enumerate(AGENT_ORDER):
            if (a, b) in length:
                grid[i, j] = statistics.median(length[a, b])
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    image = ax.imshow(grid, cmap=BLUES, vmin=0)
    ax.set_xticks(range(8), [SHORT[a] for a in AGENT_ORDER])
    ax.set_yticks(range(8), [SHORT[a] for a in AGENT_ORDER])
    ax.set_xlabel("cầm O")
    ax.set_ylabel("cầm X")
    ax.grid(False)
    for i in range(8):
        for j in range(8):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if grid[i, j] > np.nanmax(grid) * 0.55 else "#191919")
    fig.colorbar(image, ax=ax, shrink=0.8, label="trung vị số nước")
    save(fig, "van-do-dai-ma-tran")

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0))
    for ax, group in zip(axes, (("L2-Rule", "L3-AlphaBeta", "L4-MCTS", "L7-PUCT"),
                                ("L5-PolicySL", "L6-PolicyRL", "L8-AlphaZero", "L1-Random")), strict=True):
        for i, agent in enumerate(group):
            rows = [b for b in board["by_ply"][agent] if b["n"] >= 30]
            series(ax, [b["ply"] for b in rows], [b["time_median"] for b in rows], i, SHORT[agent])
        ax.set_yscale("log")
        ax.set_xlabel("nước thứ (gom 4)")
        ax.legend(ncol=2, fontsize=8)
    axes[0].set_ylabel("trung vị ms/nước (log)")
    fig.tight_layout()
    save(fig, "van-thoi-gian")


# ---------------------------------------------------------------- thống kê
def stat_figures() -> None:
    data = chung.load_json("phan-tich-thong-ke")
    pairs = sorted(data["pairs"], key=lambda p: p["score"])
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    for y, p in enumerate(pairs):
        lo, hi = p["boot"]
        ax.plot([100 * lo, 100 * hi], [y, y], color=COLORS[0], linewidth=2)
        ax.plot(100 * p["score"], y, "o", color=COLORS[0], markersize=5, markeredgecolor="white")
    ax.axvline(50, color="#666666", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(pairs)), [f"{SHORT[p['a']]}–{SHORT[p['b']]}" for p in pairs], fontsize=8)
    ax.set_xlabel("điểm của tác tử đứng trước (%), bootstrap 95% theo khai cuộc")
    save(fig, "tk-khoang")

    conv = data["ratings"]["convergence"]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), sharey=True)
    for ax, group in zip(axes, (AGENT_ORDER[:4], AGENT_ORDER[4:]), strict=True):
        for i, agent in enumerate(group):
            series(ax, [c["openings"] for c in conv],
                   [c["agents"][agent][2] - c["agents"][agent][0] for c in conv], i, SHORT[agent])
        ax.set_xlabel("số khai cuộc")
        ax.legend(ncol=2, fontsize=8)
    axes[0].set_ylabel("độ rộng khoảng 95% (Elo)")
    fig.tight_layout()
    save(fig, "tk-hoi-tu")

    fit = data["fit"]["pairs"]
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 100], [0, 100], color="#666666", linewidth=1, linestyle="--")
    ax.scatter([100 * r["predicted"] for r in fit], [100 * r["observed"] for r in fit], s=28, color=COLORS[0],
               edgecolor="white", linewidth=0.8, zorder=3)
    for r in sorted(fit, key=lambda r: -abs(r["residual"]))[:4]:
        ax.annotate(f"{SHORT[r['a']]}–{SHORT[r['b']]}", (100 * r["predicted"], 100 * r["observed"]), fontsize=8,
                    xytext=(5, -4), textcoords="offset points")
    ax.set_xlabel("điểm Bradley-Terry dự đoán (%)")
    ax.set_ylabel("điểm quan sát (%)")
    ax.set_aspect("equal")
    save(fig, "tk-phan-du")


# ---------------------------------------------------------------- mạng
def net_figures() -> None:
    data = chung.load_json("phan-tich-mang")
    names = list(data["models"])
    short = ["SL v3", "SL không aug", "L6 (RL)", "L8 (AZ)"]
    d = data["dataset"]

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.7), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    hist = np.array(d["stones_hist"][:80])
    axes[0].bar(range(len(hist)), hist, color=COLORS[0], width=0.8)
    axes[0].set_xlabel("số quân trên bàn")
    axes[0].set_ylabel("số thế")
    lengths = np.array(d["game_length_hist"][:80])
    axes[1].bar(range(len(lengths)), lengths, color=COLORS[0], width=0.8)
    axes[1].set_xlabel("số thế giữ lại mỗi ván")
    axes[1].set_ylabel("số ván")
    share = 100 * np.array(d["policy_heat"])
    board_heat(axes[2], share, f"nhãn policy TB (max {share.max():.1f}%)")
    fig.tight_layout()
    save(fig, "mang-du-lieu")

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.2))
    axes[0].plot([-1, 1], [-1, 1], color="#666666", linewidth=1, linestyle="--")
    for i, n in enumerate(names):
        rows = [c for c in data["models"][n]["calibration"] if c["n"] >= 30]
        series(axes[0], [c["v"] for c in rows], [c["z"] for c in rows], i, short[i])
        total = sum(c["n"] for c in data["models"][n]["calibration"])
        series(axes[1], [(c["lo"] + c["hi"]) / 2 for c in data["models"][n]["calibration"]],
               [100 * c["n"] / total for c in data["models"][n]["calibration"]], i, short[i])
    axes[0].set_xlabel("v dự đoán (TB trong khoảng)")
    axes[0].set_ylabel("kết quả thực tế z (TB)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("v dự đoán")
    axes[1].set_ylabel("tỷ lệ thế (%)")
    fig.tight_layout()
    save(fig, "mang-calib")

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8))
    buckets = data["models"][names[0]]["by_stones"]
    labels = [f"{b['lo']}–{b['hi'] - 1}" if b["hi"] < 122 else f"≥{b['lo']}" for b in buckets]
    for i, n in enumerate(names):
        series(axes[0], range(1, 21), [100 * v for v in data["models"][n]["rank_cdf"]], i, short[i])
        series(axes[1], range(len(buckets)), [100 * b["top1"] for b in data["models"][n]["by_stones"]], i, short[i])
        series(axes[2], range(len(buckets)), [b["mse"] for b in data["models"][n]["by_stones"]], i, short[i])
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("top-k (%)")
    for ax, label in zip(axes[1:], ("top-1 (%)", "MSE value"), strict=True):
        ax.set_xticks(range(len(buckets)), labels, rotation=45, fontsize=8)
        ax.set_xlabel("số quân")
        ax.set_ylabel(label)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07),
               fontsize=9)
    fig.tight_layout()
    save(fig, "mang-giai-doan")

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.0), sharey=True)
    for ax, t in zip(axes.flat, data["trajectories"], strict=True):
        plies = range(t["opening_plies"], len(t["moves"]))
        for i, (name, values) in enumerate(t["values"].items()):
            series(ax, plies, values, i, "mạng L5/L7" if i == 0 else "mạng L8")
        ax.axhline(0, color="#666666", linewidth=0.8)
        if t["first_vcf"]:
            ax.axvline(t["first_vcf"]["ply"], color="#ec4647", linewidth=1, linestyle="--")
        outcome = {1: "X thắng", -1: "O thắng", 0: "hoà"}[t["result"]]
        ax.set_title(f"Ván {t['game_id']}: X={SHORT[t['agent_x']]}, O={SHORT[t['agent_o']]}, {outcome}", fontsize=9)
        ax.set_ylim(-1.05, 1.05)
    for ax in axes[1]:
        ax.set_xlabel("số nước đã đi")
    for ax in axes[:, 0]:
        ax.set_ylabel("v (góc nhìn X)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    save(fig, "mang-quy-dao")

    suite = {p["id"]: p for p in chung.load_json("bo-chien-thuat")["positions"]}
    for entry in data["heat"]:
        p = suite[entry["id"]]
        x, o = chung.replay(p["moves"])
        side = "X" if p["ply"] % 2 == 0 else "O"
        blocks = [r"\begin{figure}[H]", r"  \centering"]
        for k, n in enumerate(names):
            policy = entry["maps"][n]["policy"]
            top = max(policy) or 1
            heat = {c: v / top for c, v in enumerate(policy) if v > 0.005}
            mass = sum(policy[a] for a in p["answers"])
            blocks += [r"  \begin{subfigure}[t]{.48\linewidth}", r"    \centering",
                       chung.board_tikz(x, o, heat=heat, unit=0.42),
                       (rf"    \caption{{{short[k]}: $v={chung.num(entry['maps'][n]['value'], 2).replace(',', '{,}')}$; đáp án "
                       rf"{chung.num(100 * mass)}\%; cao nhất {chung.cell_name(int(np.argmax(policy)))} "
                       rf"({chung.num(100 * top)}\%)}}"),
                       r"  \end{subfigure}" + (r"\hfill" if k % 2 == 0 else r"\par\medskip")]
        answers = ", ".join(chung.cell_name(a) for a in p["answers"][:6])
        blocks += [(rf"  \caption{{Policy của bốn mạng tại thế {p['id']} ({side} đi). Ô càng đậm xác suất càng "
                   rf"cao (chuẩn hoá theo ô lớn nhất của từng mạng). Đáp án: {answers}.}}"),
                   rf"  \label{{fig:cs-policy-{p['category'].lower()}}}", r"\end{figure}", ""]
        (chung.TABLES / f"hinh-policy-{p['category'].lower()}.tex").write_text("\n".join(blocks), encoding="utf-8")


# ---------------------------------------------------------------- tìm kiếm
def search_figures() -> None:
    data = chung.load_json("phan-tich-tim-kiem")
    depths = data["alphabeta"]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9))
    x = [d["depth"] for d in depths]
    for ax, key, label in ((axes[0], "nodes", "số nút (log)"), (axes[1], "time_ms", "ms mỗi nước (log)")):
        for i, (q, name) in enumerate(((50, "P50"), (90, "P90"), (100, "max"))):
            series(ax, x, [np.percentile([r[key] for r in d["rows"]], q) for d in depths], i, name)
        ax.set_yscale("log")
        ax.set_xlabel("độ sâu")
        ax.set_ylabel(label)
    axes[0].legend()
    fig.tight_layout()
    save(fig, "tk-do-sau")

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9))
    for i, threads in enumerate(sorted({r["threads"] for r in data["throughput"]})):
        rows = [r for r in data["throughput"] if r["threads"] == threads]
        series(axes[0], [r["batch"] for r in rows], [r["ms"] for r in rows], i, f"{threads} luồng")
        series(axes[1], [r["batch"] for r in rows], [r["per_second"] for r in rows], i, f"{threads} luồng")
    for ax, label in zip(axes, ("ms mỗi lô", "thế mỗi giây"), strict=True):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("kích thước lô")
        ax.set_ylabel(label)
    axes[0].legend()
    fig.tight_layout()
    save(fig, "tk-thong-luong")

    for entry in data["heatmaps"]:
        p = entry["position"]
        x_bb, o_bb = chung.replay(p["moves"])
        labels = list(dict.fromkeys(m["label"] for m in entry["maps"]))
        blocks = [r"\begin{figure}[H]", r"  \centering"]
        for label in labels:
            for k, m in enumerate(m for m in entry["maps"] if m["label"] == label):
                visits = {int(c): v for c, v in m["visits"].items()}
                top = max(visits.values(), default=1)
                heat = {c: v / top for c, v in visits.items() if v / top > 0.02}
                blocks += [r"  \begin{subfigure}[t]{.32\linewidth}", r"    \centering",
                           chung.board_tikz(x_bb, o_bb, heat=heat, marks={}, unit=0.29),
                           (rf"    \caption{{\footnotesize {label}, {chung.count(m['sims'])}: "
                           rf"{chung.cell_name(m['move'])}}}"),
                           r"  \end{subfigure}" + (r"\hfill" if k < 2 else r"\par\medskip")]
        side = "X" if p["ply"] % 2 == 0 else "O"
        answers = ", ".join(chung.cell_name(a) for a in p["answers"][:6])
        blocks += [(rf"  \caption{{Phân bố lượt thăm ở gốc tại thế {p['id']} ({side} đi) khi tăng ngân sách. "
                   rf"Đáp án: {answers}. Chú thích con ghi nước được chọn.}}"),
                   rf"  \label{{fig:cs-tham-{p['id'].lower()}}}", r"\end{figure}", ""]
        (chung.TABLES / f"hinh-tham-{p['id'].lower()}.tex").write_text("\n".join(blocks), encoding="utf-8")


GROUPS = (("chien-thuat-scale", tactics_figures), ("phan-tich-thong-ke", game_figures),
          ("phan-tich-thong-ke", stat_figures), ("phan-tich-mang", net_figures),
          ("phan-tich-tim-kiem", search_figures))


def main() -> None:
    for required, draw in GROUPS:
        if (chung.DATA / f"{required}.json").is_file():
            draw()
            print("xong", draw.__name__)
        else:
            print("chưa có dữ liệu", required, "→ bỏ qua", draw.__name__)


if __name__ == "__main__":
    main()
