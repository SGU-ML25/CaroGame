"""Cấp 1 — Random (mục 04.1): mốc dưới cùng, mọi tác tử khác phải thắng gần như tuyệt đối."""

from __future__ import annotations

from caro.agents.base import BaseAgent, Limits
from caro.engine import CaroGame


class RandomAgent(BaseAgent):
    def select_action(self, game: CaroGame, limits: Limits) -> int:
        return self.rng.choice(game.legal_actions())
