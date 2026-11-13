import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class CaroResNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        # Cấu hình mặc định hoặc load từ config YAML
        self.in_channels = 4
        self.num_channels = 64
        self.num_blocks = 6
        board_size = 11
        
        if config is not None:
            self.num_channels = config.get("num_channels", 64)
            self.num_blocks = config.get("num_blocks", 6)

        # Thân mạng Convolution ban đầu
        self.conv_in = nn.Conv2d(self.in_channels, self.num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(self.num_channels)
        
        # Các khối Residual
        self.res_blocks = nn.ModuleList([ResidualBlock(self.num_channels) for _ in range(self.num_blocks)])
        
        # Policy Head (Dự đoán nước đi trên bàn cờ 11x11 = 121 ô)
        self.policy_conv = nn.Conv2d(self.num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value Head (Đánh giá trạng thái ván đấu trong khoảng [-1, 1])
        self.value_conv = nn.Conv2d(self.num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 32)
        self.value_fc2 = nn.Linear(32, 1)

    def forward(self, x, legal_moves_mask=None):
        """
        x: tensor kích thước (Batch, 4, 11, 11)
        legal_moves_mask: tensor boolean/mask các ô hợp lệ (Batch, 121)
        """
        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)
            
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)
        
        # Mask các ô đã có quân (gán thành -infinity)
        if legal_moves_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_moves_mask, float('-inf'))
            
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy_logits, value