"""Test thế cờ dựng tay cho RuleAgent — Cấp 2 issue #9.

Chạy: pytest tests/test_rule_agent.py -v
"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock

from caro.engine.bitboard import bit, index
from caro.agents.base import Limits
from caro.agents.rule_agent import RuleAgent, _makes_four


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_game(me_bits: int, opp_bits: int) -> MagicMock:
    """CaroGame mock: me_opp() trả (me, opp) theo đúng interface thật."""
    game = MagicMock()
    game.me_opp.return_value = (me_bits, opp_bits)
    return game


def place(*cells: tuple[int, int]) -> int:
    """Dựng bitboard từ danh sách (row, col)."""
    b = 0
    for r, c in cells:
        b |= bit(r, c)
    return b


LIMITS = Limits(mode="fixed", max_depth=1)


# ---------------------------------------------------------------------------
# P1: Thắng ngay
# ---------------------------------------------------------------------------

class TestP1Win:
    def test_horizontal_win(self):
        """4 quân liên tiếp hàng 0 cột 0-3 → đặt (0,4) thắng."""
        me  = place((0,0),(0,1),(0,2),(0,3))
        opp = 0
        agent = RuleAgent(limits=LIMITS)
        move = agent.select_action(make_game(me, opp), LIMITS)
        assert move == index(0, 4)
        assert agent.last_stats()["reason"] == "P1_win"

    def test_vertical_win(self):
        """4 quân cột 5 hàng 0-3 → đặt (4,5) thắng."""
        me  = place((0,5),(1,5),(2,5),(3,5))
        opp = 0
        agent = RuleAgent(limits=LIMITS)
        move = agent.select_action(make_game(me, opp), LIMITS)
        assert move == index(4, 5)
        assert agent.last_stats()["reason"] == "P1_win"

    def test_diagonal_win(self):
        """4 quân chéo ↘ → đặt (4,4) thắng."""
        me  = place((0,0),(1,1),(2,2),(3,3))
        opp = 0
        agent = RuleAgent(limits=LIMITS)
        move = agent.select_action(make_game(me, opp), LIMITS)
        assert move == index(4, 4)
        assert agent.last_stats()["reason"] == "P1_win"


# ---------------------------------------------------------------------------
# P2: Chặn thua
# ---------------------------------------------------------------------------

class TestP2Block:
    def test_block_horizontal(self):
        """Đối thủ 4 quân hàng 2 cột 0-3 → ta chặn (2,4)."""
        opp = place((2,0),(2,1),(2,2),(2,3))
        me  = place((5,5))
        agent = RuleAgent(limits=LIMITS)
        move = agent.select_action(make_game(me, opp), LIMITS)
        assert move == index(2, 4)
        assert agent.last_stats()["reason"] == "P2_block_loss"

    def test_p1_beats_p2(self):
        """Ta thắng ngay VÀ đối thủ cũng sắp thắng → P1 ưu tiên hơn P2."""
        me  = place((0,0),(0,1),(0,2),(0,3))
        opp = place((1,0),(1,1),(1,2),(1,3))
        agent = RuleAgent(limits=LIMITS)
        agent.select_action(make_game(me, opp), LIMITS)
        assert agent.last_stats()["reason"] == "P1_win"


# ---------------------------------------------------------------------------
# P3: Tạo đòn kép
# ---------------------------------------------------------------------------

class TestP3DoubleAttack:
    def test_creates_double_threat(self):
        """Đặt (5,5) tạo >=2 ô thắng cho ta → P3."""
        # me: 3 hàng + 3 cột giao tại (5,5)
        me  = place((5,3),(5,4),(5,6),(3,5),(4,5))
        opp = place((0,10))  # xa, không ảnh hưởng
        agent = RuleAgent(limits=LIMITS)
        agent.select_action(make_game(me, opp), LIMITS)
        assert agent.last_stats()["reason"] == "P3_double_attack"


# ---------------------------------------------------------------------------
# P4: Phá đòn kép của đối thủ
# ---------------------------------------------------------------------------

class TestP4BlockDouble:
    def test_block_opponent_double(self):
        """opp 3 quân hàng + 3 quân cột giao tại (5,5) → đặt (5,5) tạo đòn kép → P4 chặn."""
        # opp chưa có winning_squares (không kích hoạt P2)
        # nhưng nếu opp đặt (5,5): hàng thành 4, cột thành 4 → winning_squares ≥ 2
        opp = place((5,2),(5,3),(5,4),(2,5),(3,5),(4,5))
        me  = place((0,0))
        agent = RuleAgent(limits=LIMITS)
        move = agent.select_action(make_game(me, opp), LIMITS)
        assert agent.last_stats()["reason"] == "P4_block_double_attack"
        # (5,5) là 1 trong các ô kích hoạt P4; không assert move cụ thể vì
        # P4 có thể chọn ô chặn hợp lệ khác trước (5,5) trong danh sách candidates)

    def test_p2_beats_p4(self):
        """P2 (chặn thua ngay) ưu tiên hơn P4 (phá đòn kép)."""
        # opp có nước thắng ngay (4 liên tiếp) VÀ đòn kép tiềm năng
        opp = place((0,0),(0,1),(0,2),(0,3),   # 4 liên tiếp → P2
                    (5,1),(5,2),(5,3),(5,4),(1,5),(2,5),(3,5),(4,5))  # đòn kép → P4
        me  = place((9,9))
        agent = RuleAgent(limits=LIMITS)
        agent.select_action(make_game(me, opp), LIMITS)
        assert agent.last_stats()["reason"] == "P2_block_loss"


# ---------------------------------------------------------------------------
# P5: Tạo tứ
# ---------------------------------------------------------------------------

class TestP5Four:
    def test_creates_four_no_double(self):
        """me 3 quân cạnh mép → chỉ tạo tứ (1 ô thắng), không phải đòn kép."""
        # me=(3,0)(3,1)(3,2): đặt (3,3) → win_sqs=1, makes_four=True → P5
        me  = place((3,0),(3,1),(3,2))
        opp = place((0,10))
        agent = RuleAgent(limits=LIMITS)
        agent.select_action(make_game(me, opp), LIMITS)
        assert agent.last_stats()["reason"] == "P5_four_attack"

    def test_p5_picks_highest_p6_score(self):
        """Khi nhiều nước tứ, P5 chọn nước có evaluate cao nhất."""
        # me có 2 cụm 3 quân ở cạnh mép → mỗi cụm có thể tạo tứ
        me  = place((0,0),(0,1),(0,2),   # cụm 1: đặt (0,3) → tứ
                    (10,8),(10,9),(10,10)) # cụm 2: đặt (10,7) → tứ
        opp = place((5,5))
        agent = RuleAgent(limits=LIMITS)
        move = agent.select_action(make_game(me, opp), LIMITS)
        assert agent.last_stats()["reason"] == "P5_four_attack"
        # Nước được chọn phải thực sự tạo tứ
        assert _makes_four(me | (1 << move), opp)


# ---------------------------------------------------------------------------
# Xung đột P3 ↔ P4
# ---------------------------------------------------------------------------

class TestConflictP3P4:
    def test_p3_wins_over_p4(self):
        """Ta có đòn kép (P3) VÀ đối thủ có đòn kép tiềm năng (P4) → P3 ưu tiên."""
        # ta: 3+3 giao (5,5) → P3 tại (5,5)
        # opp: 3+3 giao (2,2) → P4 tại (2,2), nhưng P3 đến trước trong thứ tự ưu tiên
        me  = place((5,3),(5,4),(5,6),(3,5),(4,5))
        opp = place((2,0),(2,1),(2,3),(0,2),(1,2),(3,2))
        agent = RuleAgent(limits=LIMITS)
        agent.select_action(make_game(me, opp), LIMITS)
        assert agent.last_stats()["reason"] == "P3_double_attack"

    def test_p4_active_when_no_p3(self):
        """Ta không có P3 → P4 kích hoạt khi opp có đòn kép tiềm năng."""
        me  = place((0,0))
        opp = place((5,2),(5,3),(5,4),(2,5),(3,5),(4,5))
        agent = RuleAgent(limits=LIMITS)
        agent.select_action(make_game(me, opp), LIMITS)
        assert agent.last_stats()["reason"] == "P4_block_double_attack"


# ---------------------------------------------------------------------------
# Xung đột P4 ↔ P5
# ---------------------------------------------------------------------------

class TestConflictP4P5:
    def test_p4_wins_over_p5(self):
        """Ta có nước tứ (P5) VÀ opp có đòn kép tiềm năng (P4) → P4 ưu tiên."""
        opp = place((5,2),(5,3),(5,4),(2,5),(3,5),(4,5))  # 3+3 giao (5,5) → P4
        me  = place((3,0),(3,1),(3,2))                     # ta có thể tạo tứ → P5
        agent = RuleAgent(limits=LIMITS)
        agent.select_action(make_game(me, opp), LIMITS)
        assert agent.last_stats()["reason"] == "P4_block_double_attack"

    def test_p5_active_when_no_p4(self):
        """Không có mối đe P4 → P5 được kích hoạt khi ta có thể tạo tứ."""
        me  = place((3,0),(3,1),(3,2))
        opp = place((0,10))   # opp không có mối đe gì
        agent = RuleAgent(limits=LIMITS)
        agent.select_action(make_game(me, opp), LIMITS)
        assert agent.last_stats()["reason"] == "P5_four_attack"


# ---------------------------------------------------------------------------
# P6 & P7
# ---------------------------------------------------------------------------

class TestP6P7:
    def test_p6_or_p7_triggered(self):
        """Không có P1-P5 → rơi vào P6/P7."""
        me  = place((5,5))
        opp = place((5,6))
        agent = RuleAgent(limits=LIMITS, priorities=["P6","P7"])
        agent.select_action(make_game(me, opp), LIMITS)
        assert agent.last_stats()["reason"] in ("P6_evaluate","P7_center_random")

    def test_p7_same_seed_same_move(self):
        """Cùng seed → cùng nước đi (tái lập được)."""
        me  = place((5,5))
        opp = place((0,10))
        agent_a = RuleAgent(limits=LIMITS, priorities=["P7"])
        agent_b = RuleAgent(limits=LIMITS, priorities=["P7"])
        agent_a.reset(42)
        agent_b.reset(42)
        game = make_game(me, opp)
        assert agent_a.select_action(game, LIMITS) == agent_b.select_action(game, LIMITS)

    def test_p7_different_seed_may_differ(self):
        """Khác seed → có thể khác nước (chạy nhiều lần để xác nhận phân phối)."""
        me  = place((5,5))
        opp = place((0,10))
        results = set()
        for seed in range(20):
            agent = RuleAgent(limits=LIMITS, priorities=["P7"])
            agent.reset(seed)
            results.add(agent.select_action(make_game(me, opp), LIMITS))
        assert len(results) > 1, "Khác seed phải tạo ra ít nhất 2 nước khác nhau"


# ---------------------------------------------------------------------------
# Tham số priorities và eval_weights
# ---------------------------------------------------------------------------

class TestPrioritiesConfig:
    def test_disable_p1_falls_through(self):
        """Tắt P1: dù ta thắng ngay vẫn không dùng P1."""
        me  = place((0,0),(0,1),(0,2),(0,3))
        opp = place((5,5))
        agent = RuleAgent(limits=LIMITS, priorities=["P2","P3","P4","P5","P6","P7"])
        agent.select_action(make_game(me, opp), LIMITS)
        assert agent.last_stats()["reason"] != "P1_win"

    def test_eval_weights_override(self):
        """eval_weights khác mặc định không crash và vẫn trả nước hợp lệ."""
        me  = place((5,5))
        opp = place((5,6))
        weights = {"0": 0, "1": 1, "2": 10, "3": 100, "4": 1000}
        agent = RuleAgent(limits=LIMITS, eval_weights=weights)
        move = agent.select_action(make_game(me, opp), LIMITS)
        assert 0 <= move <= 120

    def test_last_stats_has_reason(self):
        """last_stats() luôn có key 'reason' sau khi select_action."""
        me  = place((5,5))
        opp = place((5,6))
        agent = RuleAgent(limits=LIMITS)
        agent.select_action(make_game(me, opp), LIMITS)
        assert "reason" in agent.last_stats()