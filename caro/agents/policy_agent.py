import torch

from caro.agents.base import Agent
from caro.nn.features import board_to_tensor
from caro.nn.model import CaroResNet


class PolicyAgent(Agent):
    def __init__(self, model_path=None, config=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CaroResNet(config).to(self.device)
        self.model.eval()
        
        self.last_policy = None
        self.last_value = None
        
        if model_path:
            self.resolve_checkpoint(model_path)

    def resolve_checkpoint(self, path):
        """Nạp trọng số checkpoint kèm config mạng."""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

    def select_action(self, game_state, limits=None):
        """Chọn nước đi tối ưu bằng argmax (temperature = 0, không tìm kiếm)."""
        state_tensor = board_to_tensor(game_state).unsqueeze(0).to(self.device)
        
        # Lấy danh sách nước đi hợp lệ dưới dạng mask (121 ô)
        legal_moves = game_state.get_legal_moves()
        mask = torch.zeros(1, 121, dtype=torch.bool, device=self.device)
        for r, c in legal_moves:
            mask[0, r * 11 + c] = True
            
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor, legal_moves_mask=mask)
            policy_probs = torch.softmax(policy_logits, dim=-1)
            
        self.last_policy = policy_probs.cpu().numpy()[0]
        self.last_value = value.item()
        
        # Chọn nước đi có xác suất cao nhất (argmax)
        best_flat_idx = torch.argmax(policy_logits).item()
        action = (best_flat_idx // 11, best_flat_idx % 11)
        
        return action

    def last_stats(self):
        """Trả về phân phối policy và giá trị value để hiển thị UI/debug."""
        return {
            "policy": self.last_policy,
            "value": self.last_value
        }