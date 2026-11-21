import numpy as np

from caro.engine import CaroGame
from caro.nn.features import board_to_tensor
from caro.training.train_sl import mover_relative_values


def test_mover_relative_values_flips_sign_for_o_to_move():
    x_to_move = board_to_tensor(CaroGame()).numpy()                 # X đi: kênh 2 = 1
    o_to_move = board_to_tensor(CaroGame.from_moves([60])).numpy()  # O đi: kênh 2 = 0
    states = np.stack([x_to_move, o_to_move, x_to_move, o_to_move])
    x_wins = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)     # nhãn tuyệt đối

    rel = mover_relative_values(states, x_wins)

    # X đi: giữ nguyên; O đi: đảo dấu (X thắng = người đang đi (O) thua)
    assert rel.tolist() == [1.0, -1.0, -1.0, 1.0]
    assert rel.dtype == x_wins.dtype


def test_mover_relative_values_keeps_draws_zero():
    states = np.stack([board_to_tensor(CaroGame()).numpy(), board_to_tensor(CaroGame.from_moves([60])).numpy()])
    assert mover_relative_values(states, np.zeros(2, dtype=np.float32)).tolist() == [0.0, 0.0]
