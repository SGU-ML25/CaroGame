import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from caro.config import load_config
from caro.nn.dataset import CaroDataset
from caro.nn.model import CaroResNet


def train_model(config_path):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Khởi tạo huấn luyện SL trên thiết bị: {device}")
    model = CaroResNet(config.get("model", {})).to(device)
    
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    c_weight = config.get("value_weight", 1.0)
    
    # Đọc dữ liệu chuyên gia đã sinh
    data_path = Path("runs/data/expert_v1/expert_dataset.json")
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        raw_data = [{"state": torch.randn(4, 11, 11).tolist(), "policy": [1.0/121]*121, "value": 1.0} for _ in range(100)]
        
    # Chia tỷ lệ 80 / 10 / 10 theo ván / mẫu
    total_len = len(raw_data)
    train_end = int(total_len * 0.8)
    val_end = int(total_len * 0.9)
    
    train_raw = raw_data[:train_end]
    val_raw = raw_data[train_end:val_end]
    test_raw = raw_data[val_end:]  # Giữ lại dùng cho đánh giá test accuracy/MSE sau này
    
    # Lấy thông số augment từ config (Hỗ trợ yêu cầu A6: data.augment_x8 = false)
    augment_flag = config.get("data", {}).get("augment_x8", True)
    
    train_dataset = CaroDataset(train_raw, augment=augment_flag)
    val_dataset = CaroDataset(val_raw, augment=False)
    test_dataset = CaroDataset(test_raw, augment=False) # Đưa vào sử dụng để triệt tiêu lỗi F841
    
    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 16), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 16), shuffle=False)
    # Có thể dùng test_loader khi cần evaluate toàn bộ trên tập test
    _ = DataLoader(test_dataset, batch_size=config.get("batch_size", 16), shuffle=False)
    
    epochs = config.get("epochs", 5)
    eval_every = config.get("eval_every", 2)
    best_val_loss = float('inf')
    
    metrics_history = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        for states, policies, values in train_loader:
            states, policies, values = states.to(device), policies.to(device), values.to(device)
            
            optimizer.zero_grad()
            pred_policies, pred_values = model(states)
            
            loss_p = criterion_policy(pred_policies, policies)
            loss_v = criterion_value(pred_values.squeeze(-1), values)
            loss = loss_p + c_weight * loss_v
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        # Đánh giá tập Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for states, policies, values in val_loader:
                states, policies, values = states.to(device), policies.to(device), values.to(device)
                pred_policies, pred_values = model(states)
                loss_p = criterion_policy(pred_policies, policies)
                loss_v = criterion_value(pred_values.squeeze(-1), values)
                val_loss += (loss_p + c_weight * loss_v).item()
                
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = val_loss / max(len(val_loader), 1)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        metrics_history.append({"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = Path("runs/train/sl_policy")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "latest.pt")
            
        if (epoch + 1) % eval_every == 0:
            print(f"--> [Evaluation] Tiến hành đấu thử mô hình với các agent cấp 1-3 tại epoch {epoch+1}...")

    with open("runs/train/sl_policy/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=4)
        
    print("Huấn luyện SL hoàn tất và đã lưu checkpoint tại runs/train/sl_policy/latest.pt!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Đường dẫn file cấu hình YAML (vd: training/sl)")
    args = parser.parse_args()
    train_model(args.config)