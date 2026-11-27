"""Kiểm chuẩn cây trò chơi (perft): đếm chuỗi nước hợp lệ qua step / undo.

Với d ≤ 8 chưa bên nào có thể có 5 quân, nên số chuỗi nước đúng bằng 121!/(121−d)!.
"""

import math

import pytest

from caro.engine import CELLS, CaroGame


def perft(game: CaroGame, depth: int) -> int:
    if depth == 0:
        return 1
    if game.is_terminal():
        return 0
    actions = game.legal_actions()
    if depth == 1:
        return len(actions)
    total = 0
    for action in actions:
        game.step(action)
        total += perft(game, depth - 1)
        game.undo()
    return total


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_perft_matches_falling_factorial(depth):
    game = CaroGame()

    assert perft(game, depth) == math.perm(CELLS, depth)
    assert game.history == [] and game.key == 0
