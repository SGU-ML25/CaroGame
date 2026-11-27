"""Phân tích từng nước của 2.800 ván RR8 giữ riêng bằng bộ giải VCF. Ghi chuyen-sau/data/phan-tich-van.json.

    .venv/bin/python report/scripts/chuyen_sau/phan_tich_van.py [--workers 4]

Mỗi nước (sau khai cuộc) được gắn tối đa một nhãn, theo thứ tự ưu tiên:
    win1_*   bên đi có ô thắng ngay: đã đi vào (taken) hay bỏ lỡ (missed)
    block_*  đối thủ có đúng một ô thắng: đã chặn (done) hay bỏ sót (missed)
    lost     đối thủ có từ hai ô thắng: không còn cách chặn
    vcf_*    bên đi có VCF: nước chọn giữ được VCF (kept) hay bỏ lỡ (missed)
    safe_* / open_*   không có VCF: sau nước đi, đối thủ không có VCF (safe_kept); có VCF dù còn nước
             an toàn (open_blunder); hoặc không nước nào an toàn (open_forced)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp

import chung
import vcf

from caro.engine.bitboard import FULL, iter_bits, popcount, winning_squares

MAX_ATTACKS = 8


def label_move(me: int, opp: int, move: int) -> str:
    wins = winning_squares(me, opp)
    if wins:
        return "win1_taken" if wins >> move & 1 else "win1_missed"
    threats = winning_squares(opp, me)
    if threats:
        if popcount(threats) >= 2:
            return "lost"
        return "block_done" if threats >> move & 1 else "block_missed"
    if vcf.has_vcf(me, opp, MAX_ATTACKS):
        kept = vcf.winning_first_moves(me, opp, MAX_ATTACKS)
        return "vcf_kept" if kept >> move & 1 else "vcf_missed"
    after = me | 1 << move
    if not vcf.has_vcf(opp, after, MAX_ATTACKS):
        return "safe_kept"
    empty = ~(me | opp) & FULL
    for other in iter_bits(empty & ~(1 << move)):
        if not vcf.has_vcf(opp, me | 1 << other, MAX_ATTACKS):
            return "open_blunder"
    return "open_forced"


def analyse(record: dict) -> dict:
    moves, start = record["moves"], record["opening_plies"]
    bb = [0, 0]
    for k in range(start):
        bb[k % 2] |= 1 << moves[k]
    labels, vcf_len = [], []
    for ply in range(start, len(moves)):
        side = ply % 2
        me, opp = bb[side], bb[side ^ 1]
        labels.append(label_move(me, opp, moves[ply]))
        if labels[-1].startswith("vcf_"):
            vcf_len.append(vcf.attacks_in(vcf.solve(me, opp, MAX_ATTACKS)))
        else:
            vcf_len.append(0)
        bb[side] |= 1 << moves[ply]
    return {"game_id": record["game_id"], "labels": labels, "vcf_len": vcf_len}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    records = chung.games(chung.RUNS["rr8"])
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(analyse, records, chunksize=8)
    chung.save_json("phan-tich-van", {"max_attacks": MAX_ATTACKS, "games": results})
    print(len(results), "ván,", sum(len(r["labels"]) for r in results), "nước")


if __name__ == "__main__":
    main()
