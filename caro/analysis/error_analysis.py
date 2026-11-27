"""Phân tích lỗi tác tử từ games.jsonl của giải đấu chính thức (issue #12).

Với mỗi nước đi, dùng AlphaBeta độ sâu cao đánh giá lại;
nước khiến thế thành "thua ngay sau đó" = lỗi.

Chạy:
    python -m caro.analysis.error_analysis runs/tournament/main_mode_a_l1_l6
    python -m caro.analysis.error_analysis runs/tournament/main_mode_b_l1_l6 --depth 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from caro.agents.alphabeta import AlphaBetaAgent
from caro.agents.base import Limits
from caro.engine import CaroGame
from caro.engine.bitboard import popcount, winning_squares

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

WIN_SCORE   = 1_000_000
EVAL_DEPTH  = 3          # độ sâu đánh giá lại (tăng lên cho chính xác hơn, chậm hơn)
EVAL_LIMITS = Limits(mode="fixed", max_depth=EVAL_DEPTH)

# Ngưỡng để xác định "nước sai"
BLUNDER_THRESHOLD = -5_000   # score giảm nhiều so với nước tốt nhất


# ---------------------------------------------------------------------------
# Phân loại lỗi
# ---------------------------------------------------------------------------

def classify_error(game_before: CaroGame, move: int) -> str | None:
    """Phân loại lỗi của nước đi `move` trong thế `game_before`.

    Trả tên loại lỗi hoặc None nếu không phát hiện lỗi rõ ràng.
    """
    me, opp = game_before.me_opp()
    bit_m = 1 << move

    # 1. Bỏ sót thắng ngay
    win_sqs = winning_squares(me, opp)
    if win_sqs and not (bit_m & win_sqs):
        return "bỏ_sót_thắng_ngay"

    # 2. Bỏ sót chặn tứ (đối thủ có ô thắng ngay mà không chặn)
    opp_win = winning_squares(opp, me)
    if opp_win and not (bit_m & opp_win):
        return "bỏ_sót_chặn_tứ"

    # 3. Bỏ sót chặn đòn kép (đối thủ có ≥2 ô thắng)
    if popcount(opp_win) >= 2 and not (bit_m & opp_win):
        return "bỏ_sót_chặn_đòn_kép"

    # 4. Bỏ sót tạo đòn kép cho mình
    future_me = me | bit_m
    if popcount(winning_squares(future_me, opp)) >= 2:
        return None   # nước tốt, không phải lỗi

    # 5. Đi xa vùng giao tranh (nước đi không gần quân nào)
    occ = me | opp
    if occ:
        from caro.engine.bitboard import dilate
        nearby = dilate(dilate(occ))   # vùng bán kính 2
        if not (bit_m & nearby):
            return "đi_xa_vùng_giao_tranh"

    return "lỗi_chiến_lược"   # lỗi chung, không rõ loại


# ---------------------------------------------------------------------------
# Phân tích 1 ván
# ---------------------------------------------------------------------------

def analyze_game(record: dict, agent: AlphaBetaAgent) -> list[dict]:
    """Phân tích 1 ván, trả danh sách các nước bị đánh giá là lỗi."""
    moves   = record["moves"]
    agent_x = record.get("agent_x", "?")
    agent_o = record.get("agent_o", "?")
    result  = record.get("result", 0)   # +1 X thắng, -1 O thắng, 0 hoà
    game_id = record.get("game_id", -1)

    errors: list[dict] = []
    game = CaroGame()

    for ply, move in enumerate(moves):
        side = game.side                        # 0=X, 1=O
        agent_name = agent_x if side == 0 else agent_o

       

        # Đánh giá nước tốt nhất tại thế này
        agent.reset(0)
        best_move = agent.select_action(game, EVAL_LIMITS)
        best_score = agent.last_stats().get("score", 0)

        # Đánh giá nước thực tế được đi
        game_copy = game.clone()
        game_copy.step(move)
        me_after, opp_after = game_copy.me_opp()

        # Score sau khi đi nước thực tế (theo góc nhìn đối thủ → đảo dấu)
        from caro.engine.bitboard import evaluate
        actual_score = -evaluate(opp_after, me_after)

        # Phát hiện lỗi
        error_type = None
        if move != best_move and best_score - actual_score > abs(BLUNDER_THRESHOLD):
            error_type = classify_error(game, move)

        if error_type:
            # Xác định bên thua
            
            
            errors.append({
                "game_id":    game_id,
                "agent_x":    agent_x,
                "agent_o":    agent_o,
                "result":     result,
                "ply":        ply,
                "side":       side,
                "agent":      agent_name,
                "move":       move,
                "best_move":  best_move,
                "best_score": best_score,
                "actual_score": actual_score,
                "score_drop": best_score - actual_score,
                "error_type": error_type,
            })

        game.step(move)

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(tournament_dir: str, depth: int, max_games: int, agent_filter: str) -> None:
    path = Path(tournament_dir)
    jsonl_files = list(path.rglob("games.jsonl"))
    if not jsonl_files:
        print(f"Không tìm thấy games.jsonl trong {tournament_dir}")
        sys.exit(1)

    # Nạp tất cả ván
    all_games: list[dict] = []
    for f in jsonl_files:
        records = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        all_games.extend(records)

    # Lọc theo agent nếu có
    if agent_filter:
        all_games = [g for g in all_games
                     if agent_filter in g.get("agent_x", "") or agent_filter in g.get("agent_o", "")]

    if max_games:
        all_games = all_games[:max_games]

    print(f"Phân tích {len(all_games)} ván từ {tournament_dir}")

    # Dùng AlphaBeta để đánh giá lại
    global EVAL_DEPTH, EVAL_LIMITS
    EVAL_DEPTH  = depth
    EVAL_LIMITS = Limits(mode="fixed", max_depth=depth)
    agent = AlphaBetaAgent(
        pruning=True, move_ordering=True, tt=True,
        iterative_deepening=False,
    )

    all_errors: list[dict] = []
    for i, record in enumerate(all_games):
        if i % 50 == 0:
            print(f"  [{i}/{len(all_games)}] ván {record.get('game_id')}...")
        errors = analyze_game(record, agent)
        all_errors.extend(errors)

    # Thống kê
    print(f"\nTổng lỗi phát hiện: {len(all_errors)}")

    # Thống kê theo loại lỗi
    by_type: dict[str, int] = defaultdict(int)
    for e in all_errors:
        by_type[e["error_type"]] += 1

    print("\n--- Phân loại lỗi ---")
    for etype, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count} ({count/len(all_errors)*100:.1f}%)")

    # Thống kê theo agent
    by_agent: dict[str, int] = defaultdict(int)
    for e in all_errors:
        by_agent[e["agent"]] += 1

    print("\n--- Lỗi theo cấp ---")
    for agent_name, count in sorted(by_agent.items(), key=lambda x: -x[1]):
        print(f"  {agent_name}: {count}")

    # Lưu kết quả
    out_dir = path / "error_analysis"
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / "errors.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for e in all_errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\nĐã lưu {len(all_errors)} lỗi vào {out_path}")

    # Lưu thống kê tổng hợp
    summary = {
        "tournament_dir": tournament_dir,
        "total_games":    len(all_games),
        "total_errors":   len(all_errors),
        "by_type":        dict(by_type),
        "by_agent":       dict(by_agent),
        "eval_depth":     depth,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu tổng hợp vào {summary_path}")

    # In ván minh họa (ván có nhiều lỗi nhất)
    games_with_errors: dict[int, list] = defaultdict(list)
    for e in all_errors:
        games_with_errors[e["game_id"]].append(e)

    if games_with_errors:
        worst_game_id = max(games_with_errors, key=lambda k: len(games_with_errors[k]))
        worst_errors  = games_with_errors[worst_game_id]
        print(f"\n--- Ván minh họa (game_id={worst_game_id}, {len(worst_errors)} lỗi) ---")
        for e in worst_errors[:3]:
            print(f"  ply={e['ply']} agent={e['agent']} lỗi={e['error_type']} "
                  f"move={e['move']} best={e['best_move']} drop={e['score_drop']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phân tích lỗi tác tử từ games.jsonl")
    ap.add_argument("tournament_dir", help="thư mục tournament chứa games.jsonl")
    ap.add_argument("--depth",      type=int, default=3,  help="độ sâu AlphaBeta đánh giá lại")
    ap.add_argument("--max-games",  type=int, default=0,  help="giới hạn số ván (0=tất cả)")
    ap.add_argument("--agent",      type=str, default="", help="lọc theo tên agent")
    args = ap.parse_args()
    run_analysis(args.tournament_dir, args.depth, args.max_games, args.agent)


if __name__ == "__main__":
    main()