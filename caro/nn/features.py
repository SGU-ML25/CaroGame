import numpy as np
import torch


def board_to_tensor(game_state):
    """
    Chuyển đổi trạng thái bàn cờ thành tensor 4x11x11 từ góc nhìn bên đang đi.
    Các lớp:
      0: Quân của người chơi hiện tại (mình)
      1: Quân của đối thủ
      2: Vị trí nước vừa đi (last move)
      3: Lớp hằng số toàn 1
    """
    # Kích thước bàn cờ Caro 11x11
    board_size = 11
    tensor = np.zeros((4, board_size, board_size), dtype=np.float32)
    
    current_player = game_state.current_player
    opponent = 3 - current_player # Giả định ký hiệu player là 1 và 2
    
    # 1. Điền lớp quân mình và quân đối thủ từ board
    board = game_state.board
    tensor[0] = (board == current_player).astype(np.float32)
    tensor[1] = (board == opponent).astype(np.float32)
    
    # 2. Lớp nước vừa đi
    if hasattr(game_state, 'last_move') and game_state.last_move:
        r, c = game_state.last_move
        if 0 <= r < board_size and 0 <= c < board_size:
            tensor[2, r, c] = 1.0
            
    # 3. Lớp toàn 1
    tensor[3] = 1.0
    
    return torch.tensor(tensor, dtype=torch.float32)

def batch_board_to_tensor(game_states):
    """Batch hoá cho nhiều trạng thái."""
    return torch.stack([board_to_tensor(state) for state in game_states])