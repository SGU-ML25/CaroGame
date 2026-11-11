from __future__ import annotations

import numpy as np
import torch

from caro.engine.game import CaroGame, N, X


def board_to_tensor(game_state: CaroGame) -> torch.Tensor:
    """
    Chuyển đổi trạng thái CaroGame thành tensor đặc trưng đầu vào cho mô hình Neural Network.
    Kích thước trả về: (4, 11, 11) hoặc (4, N, N)
    Các kênh (channels):
        0: Quân cờ của người chơi hiện tại (Me)
        1: Quân cờ của đối thủ (Opponent)
        2: Hằng số 1 nếu là lượt của X, toàn 0 nếu là lượt của O (hoặc ngược lại tùy quy ước)
        3: Lịch sử nước đi hoặc bias khác
    """
    # Lấy thông tin lượt đi hiện tại từ thuộc tính 'side' của CaroGame
    current_side = game_state.side
    
    # Tạo tensor 4 x N x N
    tensor = np.zeros((4, N, N), dtype=np.float32)
    
    x_bb, o_bb = game_state.bb
    
    # Xác định bitboard của bản thân (Me) và đối thủ (Opponent)
    me_bb = x_bb if current_side == X else o_bb
    opp_bb = o_bb if current_side == X else x_bb
    
    for r in range(N):
        for c in range(N):
            idx = r * N + c
            if (me_bb >> idx) & 1:
                tensor[0, r, c] = 1.0
            if (opp_bb >> idx) & 1:
                tensor[1, r, c] = 1.0
                
    # Kênh thứ 3 lưu thông tin lượt đi (1.0 nếu X đi, 0.0 nếu O đi)
    if current_side == X:
        tensor[2, :, :] = 1.0
        
    # Kênh thứ 4 có thể dùng cho bias tổng quát hoặc để trống (mặc định 0.0)
    
    return torch.tensor(tensor, dtype=torch.float32)