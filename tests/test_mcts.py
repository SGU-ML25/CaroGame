"""Unit tests cho tác tử cấp 4 — MCTS (UCT) (issue #13).

Kiểm tra:
1. Đăng ký kind='mcts' và khởi tạo từ config configs/agents/l4_mcts.yaml
2. 4 bước: Selection - Expansion - Simulation - Backpropagation
3. Công thức UCT và chọn nước có N lớn nhất
4. Mở rộng chỉ trên candidates(candidate_radius)
5. Rollout random và light (thắng ngay → chặn → ngẫu nhiên)
6. Dừng theo max_sims (chế độ B) và time_ms (chế độ A)
7. Tái sử dụng cây (reuse_tree) và reset
8. last_stats(): sims, time_ms, visits (heatmap cho UI), pv, score
9. Nhận diện thắng ngay và chặn thua ngay với số mô phỏng nhỏ
10. Tính tất định: cùng seed → cùng nước
11. Thắng Cấp 1 (RandomAgent) gần tuyệt đối; đo mô phỏng/giây
"""

from __future__ import annotations

import random
import time

import pytest

from caro.agents import make_agent
from caro.agents.base import Limits
from caro.agents.mcts import MCTSAgent, first_winning_square, rollout_fast
from caro.agents.random_agent import RandomAgent
from caro.engine import CaroGame
from caro.engine.bitboard import candidates, index, iter_bits

LIMITS_50 = Limits(mode="fixed", max_sims=50)
LIMITS_100 = Limits(mode="fixed", max_sims=100)
LIMITS_200 = Limits(mode="fixed", max_sims=200)


def build_game(*plies: tuple[int, int]) -> CaroGame:
    """Tạo game từ các cặp (ô, lượt). Thứ tự theo tham số truyền vào."""
    g = CaroGame()
    for move, _ in plies:
        g.step(move)
    return g


# ---------------------------------------------------------------------------
# 1. Config & Khởi tạo
# ---------------------------------------------------------------------------
class TestMCTSInit:
    def test_make_agent_from_l4_config(self):
        agent = make_agent("agents/l4_mcts")
        assert isinstance(agent, MCTSAgent)
        assert agent.name == "L4-MCTS"
        assert agent.c_uct == 1.4
        assert agent.rollout_mode == "random"
        assert agent.rollout_radius == 1
        assert agent.candidate_radius == 2
        assert agent.reuse_tree is False

    def test_invalid_rollout_raises(self):
        with pytest.raises(ValueError, match="rollout phải là"):
            MCTSAgent(rollout="unknown")


# ---------------------------------------------------------------------------
# 2. Bốn bước và cấu trúc cây (Selection, Expansion, Simulation, Backprop)
# ---------------------------------------------------------------------------
class TestMCTSSearchTree:
    def test_tree_expansion_and_visits(self):
        agent = MCTSAgent(rollout="random", candidate_radius=2)
        agent.reset(42)
        game = CaroGame()
        # Nước đầu tiên: chỉ có 1 ô ứng viên là CENTER (60)
        action = agent.select_action(game, Limits(mode="fixed", max_sims=20))
        assert action == 60

        stats = agent.last_stats()
        assert stats["sims"] == 20
        assert stats["visits"] == {60: 20}
        assert stats["score"] is not None

    def test_expansion_only_on_candidates(self):
        agent = MCTSAgent(rollout="random", candidate_radius=2)
        agent.reset(42)
        game = CaroGame()
        game.step(60)  # X đi ở giữa bàn

        expected_cands = set(iter_bits(candidates(*game.me_opp(), radius=2)))
        agent.select_action(game, Limits(mode="fixed", max_sims=30))
        stats = agent.last_stats()

        for move in stats["visits"]:
            assert move in expected_cands

    def test_final_move_is_most_visited_child(self):
        agent = MCTSAgent(rollout="random")
        agent.reset(123)
        game = CaroGame()
        game.step(60)
        action = agent.select_action(game, LIMITS_100)
        stats = agent.last_stats()
        visits = stats["visits"]
        # Nước được chọn phải có số lượt thăm bằng max trong visits
        assert visits[action] == max(visits.values())


# ---------------------------------------------------------------------------
# 3. Rollout Fast & First Winning Square
# ---------------------------------------------------------------------------
class TestRollout:
    def test_first_winning_square_detects_win(self):
        # 4 quân hàng 3: (3, 2), (3, 3), (3, 4), (3, 5)
        me = sum(1 << index(3, c) for c in (2, 3, 4, 5))
        opp = 0
        sq = first_winning_square(me, opp)
        assert sq in (index(3, 1), index(3, 6))

    def test_rollout_terminates_and_valid_result(self):
        rng = random.Random(42)
        for mode in ("random", "light"):
            res = rollout_fast(1 << 60, 0, 1, rng, mode, 1)
            assert res in (-1, 0, 1)


# ---------------------------------------------------------------------------
# 4. Ngân sách: Chế độ B (fixed max_sims) và Chế độ A (time_ms)
# ---------------------------------------------------------------------------
class TestBudgets:
    def test_fixed_sims_stops_exactly(self):
        agent = MCTSAgent()
        agent.reset(0)
        game = CaroGame()
        game.step(60)
        agent.select_action(game, Limits(mode="fixed", max_sims=120))
        assert agent.last_stats()["sims"] == 120

    def test_time_mode_respects_deadline(self):
        agent = MCTSAgent()
        agent.reset(0)
        game = CaroGame()
        game.step(60)
        t0 = time.perf_counter()
        agent.select_action(game, Limits(mode="time", time_ms=80))
        elapsed = (time.perf_counter() - t0) * 1000
        stats = agent.last_stats()
        assert stats["sims"] > 0
        # Không chạy lố quá nhiều so với time_ms
        assert elapsed < 150


# ---------------------------------------------------------------------------
# 5. Tái sử dụng cây (reuse_tree)
# ---------------------------------------------------------------------------
class TestReuseTree:
    def test_reuse_tree_preserves_subtree(self):
        agent = MCTSAgent(reuse_tree=True)
        agent.reset(42)
        game = CaroGame()
        game.step(60)  # X đi 60

        # Lượt 1 của O
        m1 = agent.select_action(game, Limits(mode="fixed", max_sims=80))
        game.step(m1)
        assert agent._root is not None

        # X đi tiếp một nước gần đó
        cands_x = list(iter_bits(candidates(*game.me_opp(), 2)))
        x_move = cands_x[0] if cands_x[0] != m1 else cands_x[1]
        game.step(x_move)

        # Lượt 2 của O
        m2 = agent.select_action(game, Limits(mode="fixed", max_sims=80))
        assert m2 in game.legal_actions()

    def test_reuse_tree_disabled_by_default(self):
        agent = MCTSAgent(reuse_tree=False)
        agent.reset(42)
        game = CaroGame()
        game.step(60)
        agent.select_action(game, Limits(mode="fixed", max_sims=50))
        assert agent._root is None

    def test_reset_clears_tree(self):
        agent = MCTSAgent(reuse_tree=True)
        agent.reset(42)
        game = CaroGame()
        game.step(60)
        agent.select_action(game, Limits(mode="fixed", max_sims=50))
        assert agent._root is not None
        agent.reset(99)
        assert agent._root is None


# ---------------------------------------------------------------------------
# 6. Thống kê last_stats() & tương thích UI Heatmap
# ---------------------------------------------------------------------------
class TestStats:
    def test_last_stats_keys(self):
        agent = MCTSAgent()
        agent.reset(1)
        game = CaroGame()
        game.step(60)
        agent.select_action(game, LIMITS_50)
        s = agent.last_stats()
        assert "sims" in s
        assert "time_ms" in s
        assert "visits" in s
        assert "pv" in s
        assert isinstance(s["visits"], dict)
        assert sum(s["visits"].values()) == s["sims"]


# ---------------------------------------------------------------------------
# 7. Tính tất định: cùng seed → cùng nước
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_same_seed_same_moves(self):
        for seed in (7, 42, 999):
            game1 = build_game((60, 0), (61, 1), (50, 0))
            game2 = build_game((60, 0), (61, 1), (50, 0))

            a1 = MCTSAgent(rollout="random")
            a1.reset(seed)
            mv1 = a1.select_action(game1, LIMITS_100)

            a2 = MCTSAgent(rollout="random")
            a2.reset(seed)
            mv2 = a2.select_action(game2, LIMITS_100)

            assert mv1 == mv2
            assert a1.last_stats()["visits"] == a2.last_stats()["visits"]


# ---------------------------------------------------------------------------
# 8. Nhận diện thắng ngay và chặn thua ngay
# ---------------------------------------------------------------------------
class TestTactics:
    def test_win_in_one_horizontal(self):
        """X có 4 quân liên tiếp hàng 5 → đặt ô thắng ngay."""
        # X: (5, 3), (5, 4), (5, 5), (5, 6) -> 58, 59, 60, 61
        game = CaroGame()
        # X và O xen kẽ: X đặt 4 quân hàng 5, O đặt nơi khác
        x_cells = [index(5, 3), index(5, 4), index(5, 5), index(5, 6)]
        o_cells = [index(1, 1), index(1, 2), index(1, 3), index(1, 4)]
        for i in range(4):
            game.step(x_cells[i])
            game.step(o_cells[i])
        # Lượt của X đi nước thứ 5 để thắng ngay
        agent = MCTSAgent(rollout="light")
        agent.reset(42)
        mv = agent.select_action(game, Limits(mode="fixed", max_sims=60))
        assert mv in (index(5, 2), index(5, 7))

    def test_block_loss_in_one(self):
        """O có 4 quân liên tiếp hàng 5 → X phải chặn ngay ở 2 đầu."""
        game = CaroGame()
        # X: (4, 4), (4, 5), (6, 4), (6, 5)
        # O: (5, 3), (5, 4), (5, 5), (5, 6)
        x_cells = [index(4, 4), index(4, 5), index(6, 4), index(6, 5)]
        o_cells = [index(5, 3), index(5, 4), index(5, 5), index(5, 6)]
        for i in range(4):
            game.step(x_cells[i])
            game.step(o_cells[i])
        # Lượt của X, O đang đe doạ thắng ngay ở (5, 2) hoặc (5, 7)
        agent = MCTSAgent(rollout="light")
        agent.reset(42)
        mv = agent.select_action(game, Limits(mode="fixed", max_sims=100))
        assert mv in (index(5, 2), index(5, 7))


# ---------------------------------------------------------------------------
# 9. Thắng Cấp 1 (RandomAgent) gần tuyệt đối & đo mô phỏng/giây
# ---------------------------------------------------------------------------
class TestVsRandomAgent:
    def test_mcts_defeats_random_agent(self):
        mcts = MCTSAgent(rollout="light")
        rnd = RandomAgent()

        wins = 0
        total_games = 10
        total_sims = 0
        t0 = time.perf_counter()

        for g_idx in range(total_games):
            game = CaroGame()
            mcts.reset(100 + g_idx)
            rnd.reset(200 + g_idx)
            mcts_side = 0 if g_idx % 2 == 0 else 1

            while not game.is_terminal():
                if game.side == mcts_side:
                    a = mcts.select_action(game, Limits(mode="fixed", max_sims=150))
                    total_sims += mcts.last_stats()["sims"]
                else:
                    a = rnd.select_action(game, Limits())
                game.step(a)

            won = (game.result() == 1 and mcts_side == 0) or (game.result() == -1 and mcts_side == 1)
            if won:
                wins += 1

        t1 = time.perf_counter()
        duration = t1 - t0
        sims_per_sec = total_sims / duration if duration > 0 else 0
        print(f"\nMCTS vs Random: {wins}/{total_games} thắng ({wins/total_games*100:.1f}%), {sims_per_sec:.0f} mô phỏng/giây")
        assert wins >= 9  # Thắng gần tuyệt đối (≥ 90%)
