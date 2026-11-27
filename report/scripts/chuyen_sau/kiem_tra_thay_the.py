"""Các nước thay thế cho sai lầm quyết định có thật sự giữ được ván không.

    .venv/bin/python report/scripts/chuyen_sau/kiem_tra_thay_the.py

Với mỗi nước thay thế (không để lộ đường thắng ép ở độ sâu của phan_tich_van_sau): (1) tìm lại đường thắng ép của
đối thủ ở độ sâu 3; (2) hỏi KataGomo điểm của bên vừa đi sau nước đó.
"""

from __future__ import annotations

import multiprocessing as mp

import chung
from phan_tich_van_sau import DEEP, VISITS, position, prove

from caro.agents.base import Limits
from caro.agents.katagomo import KataGomoAgent
from caro.engine import CaroGame


def main() -> None:
    games = chung.load_json("phan-tich-van-sau")["games"]
    rows, jobs = [], []
    engine = KataGomoAgent(max_visits=VISITS, threads=8)
    try:
        for g in games:
            mistake = g["turning"]["mistake"]
            if not mistake:
                continue
            ply = mistake["ply"]
            me, opp = position(g["moves"], ply)
            for move in mistake["safe"]:
                engine.reset(chung.SEED)
                engine.select_action(CaroGame.from_moves([*g["moves"][:ply], move]), Limits(max_sims=VISITS))
                rows.append({"game_id": g["game_id"], "ply": ply, "move": move,
                             "engine_score": 1 - engine.last_stats()["value"]})
                jobs.append((opp, me | 1 << move, DEEP))
    finally:
        engine.close()
    print("KataGomo xong", len(rows), "nước", flush=True)
    with mp.get_context("spawn").Pool(14) as pool:
        for k, proof in enumerate(pool.imap(prove, jobs)):
            rows[k]["refuted"] = proof
            r = rows[k]
            print(f"ván {r['game_id']} nước {r['ply'] + 1} thay bằng {chung.cell_name(r['move'])}: "
                  f"KataGomo {100 * r['engine_score']:.0f}%, độ sâu 3 "
                  f"{'bị bác bỏ ' + proof['kind'] if proof else 'không tìm thấy đường thắng'}", flush=True)
            chung.save_json("kiem-tra-thay-the", rows)


if __name__ == "__main__":
    main()
