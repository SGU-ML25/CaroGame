import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from caro.config import load_config
from caro.nn.dataset import ArrayDataset
from caro.nn.model import CaroResNet


def train_model(config_path):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Khởi tạo huấn luyện SL trên thiết bị: {device}")
    model = CaroResNet(config.get("model", {})).to(device)
    
    tcfg = {**config, **(config.get("train") or {})}      # `train:` ưu tiên, vẫn đọc khoá ở gốc (sl_test.yaml)
    lr = tcfg.get("lr", 1e-3)
    weight_decay = tcfg.get("weight_decay", 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    c_weight = tcfg.get("value_coef", tcfg.get("value_weight", 1.0))
    # Mỗi thí nghiệm một thư mục (runs/train/<experiment.name>) để không ghi đè lên nhau
    out_dir = Path("runs/train") / (config.get("experiment") or {}).get("name", "sl_policy")
    batch_size = int(tcfg.get("batch_size", 16))

    data_dir = Path((config.get("data") or {}).get("dir", "runs/data/expert_v1"))
    data_path = data_dir / "expert_dataset.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Chưa có {data_path}. Chạy trước: python -m caro.training.gen_expert_data training/gen_expert_data"
        )
    with np.load(data_path) as npz:
        states, policies = npz["states"], npz["policies"]
        values, game_ids = npz["values"], npz["game_ids"]

    # Chia 80/10/10 THEO VÁN (các nước của cùng một ván không rơi vào hai tập khác nhau)
    rng = np.random.default_rng(int(config.get("seed", 42)))
    unique_games = rng.permutation(np.unique(game_ids))
    n_train, n_val = int(len(unique_games) * 0.8), int(len(unique_games) * 0.9)
    group = np.zeros(len(game_ids), dtype=np.int8)
    group[np.isin(game_ids, unique_games[n_train:n_val])] = 1
    group[np.isin(game_ids, unique_games[n_val:])] = 2
    print(f"Dữ liệu: {len(values):,} mẫu / {len(unique_games):,} ván "
          f"(train {np.sum(group == 0):,} | val {np.sum(group == 1):,} | test {np.sum(group == 2):,})")

    def subset(g, augment):
        m = group == g
        return ArrayDataset(states[m], policies[m], values[m], augment=augment)

    augment_flag = (config.get("data") or {}).get("augment_x8", True)
    train_loader = DataLoader(subset(0, augment_flag), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(subset(1, False), batch_size=batch_size, shuffle=False)

    epochs = int(tcfg.get("epochs", 5))
    eval_every = int(tcfg.get("eval_every", 2))
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
            save_dir = out_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "latest.pt")
            
        if (epoch + 1) % eval_every == 0:
            print(f"--> (epoch {epoch+1}) đấu thử với cấp 1-3 chưa được cài đặt trong train_sl.")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=4)
        
    print(f"Huấn luyện SL hoàn tất và đã lưu checkpoint tại {out_dir / 'latest.pt'}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Đường dẫn file cấu hình YAML (vd: training/sl)")
    args = parser.parse_args()
    train_model(args.config)