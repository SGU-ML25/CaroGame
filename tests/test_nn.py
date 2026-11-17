import torch

from caro.agents.policy_agent import PolicyAgent
from caro.engine import CaroGame
from caro.nn.features import board_to_tensor
from caro.nn.model import CaroResNet


class MockGameState:
    def __init__(self):
        self.side = 0  # 0 cho X, 1 cho O
        self.bb = (0, 0)  # tuple chứa bitboard (me_bb, opp_bb) giả lập
        
        # Thêm các thuộc tính hỗ trợ cho các bài test board numpy & legal moves
        self.board = torch.zeros((11, 11), dtype=torch.long).numpy()
        self.board[5, 5] = 1  # Quân của player 1
        self.board[5, 6] = 2  # Quân của player 2
        self.current_player = 1
        self.last_move = (5, 6)

    def get_legal_moves(self):
        # Trả về danh sách nước đi hợp lệ dưới dạng các cặp tọa độ (r, c)
        legal = []
        for r in range(11):
            for c in range(11):
                if self.board[r, c] == 0:
                    legal.append((r, c))
        return legal


def test_features_tensor():
    """Kiểm tra shape của tensor đặc trưng đầu ra."""
    state = MockGameState()
    tensor_state = board_to_tensor(state)
    assert tensor_state.shape == (4, 11, 11), "Lỗi kích thước tensor đặc trưng!"


def test_resnet_model():
    """Kiểm tra mô hình ResNet và policy logits / value."""
    state = MockGameState()
    tensor_state = board_to_tensor(state)
    model = CaroResNet()
    model.eval()
    
    input_batch = tensor_state.unsqueeze(0)
    legal_mask = torch.zeros(1, 121, dtype=torch.bool)
    for r, c in state.get_legal_moves():
        legal_mask[0, r * 11 + c] = True
        
    policy_logits, value = model(input_batch, legal_moves_mask=legal_mask)
    assert policy_logits.shape == (1, 121), "Lỗi kích thước Policy logits!"
    assert -1.0 <= value.item() <= 1.0, "Giá trị Value nằm ngoài khoảng [-1, 1]!"


def test_policy_agent():
    """PolicyAgent (mạng khởi tạo ngẫu nhiên, không checkpoint) đi một nước hợp lệ và ghi policy/value."""
    game = CaroGame.from_moves([60, 61])
    agent = PolicyAgent()
    action = agent.select_action(game)
    stats = agent.last_stats()

    assert isinstance(action, int)
    assert action in game.legal_actions()
    assert "policy" in stats
    assert "value" in stats
