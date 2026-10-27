"""Test AlphaBetaAgent — Cấp 3 issue #10.

Chạy: pytest tests/test_alphabeta.py -v
"""
from __future__ import annotations

import pytest
from caro.agents.alphabeta import AlphaBetaAgent
from caro.agents.base import Limits
from caro.engine import CaroGame

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def build_game(*pairs: tuple[int, int]) -> CaroGame:
    """Dựng thế cờ từ danh sách (x_move, o_move)."""
    game = CaroGame()
    for x, o in pairs:
        game.step(x)
        game.step(o)
    return game


WIN_SCORE = 1_000_000
LIMITS_D2 = Limits(mode="fixed", max_depth=2)
LIMITS_D3 = Limits(mode="fixed", max_depth=3)
LIMITS_D5 = Limits(mode="fixed", max_depth=5)


# ---------------------------------------------------------------------------
# Test 1: Thắng trong 1 nước
# ---------------------------------------------------------------------------

class TestWinInOne:
    def test_horizontal(self):
        """X có 4 quân liên tiếp hàng 5 → đặt ô thắng ngay."""
        game = build_game((60,10),(61,11),(62,12),(63,13))
        agent = AlphaBetaAgent(iterative_deepening=False)
        move = agent.select_action(game, LIMITS_D2)
        assert move in (64, 59)
        assert agent.last_stats()["score"] >= WIN_SCORE - 10

    def test_vertical(self):
        """X có 4 quân cột → đặt ô thắng ngay."""
        game = build_game((5,0),(16,1),(27,2),(38,3))
        agent = AlphaBetaAgent(iterative_deepening=False)
        move = agent.select_action(game, LIMITS_D2)
        assert move in (49, 0-11) or move == 49
        assert agent.last_stats()["score"] >= WIN_SCORE - 10


# ---------------------------------------------------------------------------
# Test 2: Thắng trong 3 nước
# ---------------------------------------------------------------------------

class TestWinInThree:
    def test_finds_win_in_3(self):
        """X có 3 quân liên tiếp → depth=5 tìm được thắng trong 3 nước."""
        game = build_game((60,10),(61,11),(62,12))
        agent = AlphaBetaAgent(iterative_deepening=False)
        agent.select_action(game, LIMITS_D5)
        stats = agent.last_stats()
        # Score WIN_SCORE - ply: thắng trong 3 nước → ply=5 → score=999995
        assert stats["score"] >= WIN_SCORE - 10
        assert len(stats["pv"]) >= 1


# ---------------------------------------------------------------------------
# Test 3: Tính nhất quán pruning
# ---------------------------------------------------------------------------

class TestPruningConsistency:
    def test_same_score_with_and_without_pruning(self):
        """Cùng depth, cùng candidates → score gốc có/không cắt tỉa phải giống nhau."""
        game = build_game((60,10),(61,11),(62,12))
        ab = AlphaBetaAgent(
            pruning=True, move_ordering=False, tt=False, iterative_deepening=False
        )
        plain = AlphaBetaAgent(
            pruning=False, move_ordering=False, tt=False, iterative_deepening=False
        )
        ab.select_action(game, LIMITS_D3)
        plain.select_action(game, LIMITS_D3)
        assert ab.last_stats()["score"] == plain.last_stats()["score"]

    def test_pruning_reduces_nodes(self):
        """Pruning phải duyệt ít node hơn no-pruning."""
        game = build_game((60,10),(61,11),(62,12))
        ab = AlphaBetaAgent(
            pruning=True, move_ordering=False, tt=False, iterative_deepening=False
        )
        plain = AlphaBetaAgent(
            pruning=False, move_ordering=False, tt=False, iterative_deepening=False
        )
        ab.select_action(game, LIMITS_D3)
        plain.select_action(game, LIMITS_D3)
        assert ab.last_stats()["nodes"] < plain.last_stats()["nodes"]


# ---------------------------------------------------------------------------
# Test 4: Tính nhất quán TT
# ---------------------------------------------------------------------------

class TestTTConsistency:
    def test_tt_same_score(self):
        """TT bật/tắt → score giống nhau."""
        game = build_game((60,10),(61,11),(62,12))
        with_tt = AlphaBetaAgent(
            tt=True, move_ordering=False, pruning=True, iterative_deepening=False
        )
        without_tt = AlphaBetaAgent(
            tt=False, move_ordering=False, pruning=True, iterative_deepening=False
        )
        with_tt.select_action(game, LIMITS_D3)
        without_tt.select_action(game, LIMITS_D3)
        assert with_tt.last_stats()["score"] == without_tt.last_stats()["score"]


# ---------------------------------------------------------------------------
# Test 5: Forced moves
# ---------------------------------------------------------------------------

class TestForcedMoves:
    def test_blocks_opponent_win(self):
        """O có 4 quân liên tiếp → X phải chặn ngay (forced_moves)."""
        # O: 0,1,2,3 hàng 0 → X phải chặn 4 hoặc 0-1=-1(invalid) → chặn 4
        game = CaroGame()
        # X đi trước: đặt xa, O đặt 4 liên tiếp
        game.step(60)  # X
        game.step(0)   # O
        game.step(61)  # X
        game.step(1)   # O
        game.step(62)  # X
        game.step(2)   # O
        game.step(50)  # X
        game.step(3)   # O — giờ X phải chặn 4
        agent = AlphaBetaAgent(forced_moves=True, iterative_deepening=False)
        move = agent.select_action(game, LIMITS_D2)
        assert move == 4

    def test_wins_immediately(self):
        """forced_moves ưu tiên thắng ngay trước khi chặn."""
        game = build_game((60,0),(61,1),(62,2),(63,3))
        agent = AlphaBetaAgent(forced_moves=True, iterative_deepening=False)
        move = agent.select_action(game, LIMITS_D2)
        assert move in (64, 59)


# ---------------------------------------------------------------------------
# Test 6: Move ordering cải thiện tốc độ
# ---------------------------------------------------------------------------

class TestMoveOrdering:
    def test_ordering_reduces_nodes(self):
        """Move ordering bật → ít node hơn no-ordering (cùng pruning=True)."""
        game = build_game((60,10),(61,11),(62,12))
        ordered = AlphaBetaAgent(
            pruning=True, move_ordering=True, tt=False, iterative_deepening=False
        )
        unordered = AlphaBetaAgent(
            pruning=True, move_ordering=False, tt=False, iterative_deepening=False
        )
        ordered.select_action(game, LIMITS_D3)
        unordered.select_action(game, LIMITS_D3)
        assert ordered.last_stats()["nodes"] <= unordered.last_stats()["nodes"]


# ---------------------------------------------------------------------------
# Test 7: Iterative deepening
# ---------------------------------------------------------------------------

class TestIterativeDeepening:
    def test_id_returns_valid_move(self):
        """Iterative deepening trả nước hợp lệ."""
        game = build_game((60,10),(61,11))
        agent = AlphaBetaAgent(iterative_deepening=True)
        move = agent.select_action(game, LIMITS_D3)
        assert 0 <= move <= 120

    def test_id_depth_increases(self):
        """ID đạt depth ≥ 1."""
        game = build_game((60,10),(61,11))
        agent = AlphaBetaAgent(iterative_deepening=True)
        agent.select_action(game, LIMITS_D3)
        assert agent.last_stats()["depth"] >= 1


# ---------------------------------------------------------------------------
# Test 8: last_stats có đủ fields
# ---------------------------------------------------------------------------

class TestLastStats:
    def test_stats_keys(self):
        """last_stats() phải có đủ nodes, depth, time_ms, pv, score."""
        game = build_game((60,10))
        agent = AlphaBetaAgent(iterative_deepening=False)
        agent.select_action(game, LIMITS_D2)
        stats = agent.last_stats()
        for key in ("nodes", "depth", "time_ms", "pv", "score"):
            assert key in stats, f"thiếu key '{key}'"

    def test_pv_is_list(self):
        """pv phải là list."""
        game = build_game((60,10))
        agent = AlphaBetaAgent(iterative_deepening=False)
        agent.select_action(game, LIMITS_D2)
        assert isinstance(agent.last_stats()["pv"], list)


# ---------------------------------------------------------------------------
# Test 9: scored_moves (sinh dữ liệu expert)
# ---------------------------------------------------------------------------

class TestScoredMoves:
    def test_returns_top_k(self):
        """scored_moves trả đúng top-k nước."""
        game = build_game((60,10),(61,11))
        agent = AlphaBetaAgent(iterative_deepening=False)
        results = agent.scored_moves(game, LIMITS_D2, top_k=3)
        assert len(results) <= 3
        assert all(0 <= m <= 120 for m, _ in results)

    def test_sorted_by_score(self):
        """Kết quả phải được sắp xếp giảm dần theo score."""
        game = build_game((60,10),(61,11))
        agent = AlphaBetaAgent(iterative_deepening=False)
        results = agent.scored_moves(game, LIMITS_D2, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_win_move_is_first(self):
        """Nước thắng ngay phải đứng đầu danh sách."""
        game = build_game((60,10),(61,11),(62,12),(63,13))
        agent = AlphaBetaAgent(iterative_deepening=False)
        results = agent.scored_moves(game, LIMITS_D2, top_k=5)
        assert results[0][0] in (64, 59)