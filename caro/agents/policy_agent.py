"""Cấp 5–6: đi theo policy của mạng, không tìm kiếm (argmax, hoặc lấy mẫu khi temperature > 0)."""

from __future__ import annotations

import numpy as np
import torch

from caro.agents.base import BaseAgent, Limits
from caro.engine import CaroGame
from caro.hub import resolve_checkpoint
from caro.nn.features import board_to_tensor
from caro.nn.model import CaroResNet


class PolicyAgent(BaseAgent):
    """checkpoint: đường dẫn .pt cục bộ, `model:<tên>` (Hugging Face) hoặc `hf://...`."""

    def __init__(self, name=None, limits=None, checkpoint=None, temperature=0.0, channels=64, blocks=6):
        super().__init__(name, limits)
        self.temperature = float(temperature)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CaroResNet({"num_channels": channels, "num_blocks": blocks}).to(self.device)
        if checkpoint:
            state = torch.load(resolve_checkpoint(checkpoint), map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state)
        self.model.eval()

    def select_action(self, game: CaroGame, limits: Limits | None = None) -> int:
        legal = game.legal_actions()
        mask = torch.zeros(1, 121, dtype=torch.bool, device=self.device)
        mask[0, legal] = True
        x = board_to_tensor(game).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(x, legal_moves_mask=mask)
        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
        if self.temperature > 0:
            weights = probs ** (1.0 / self.temperature)
            action = int(self.rng.choices(range(121), weights=weights.tolist())[0])
        else:
            action = int(np.argmax(probs))
        self._stats = {
            "policy": {int(a): float(probs[a]) for a in legal if probs[a] > 1e-4},
            "value": float(value.item()),
        }
        return action
