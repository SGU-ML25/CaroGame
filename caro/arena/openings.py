import json
import random
import argparse
from pathlib import Path

def generate_openings(num_openings=100):
    openings = []
    # Khởi tạo vùng 5x5 giữa bàn cờ 11x11 (tọa độ hàng 3-7, cột 3-7)
    center_squares = [(r, c) for r in range(3, 8) for c in range(3, 8)]

    for _ in range(num_openings):
        # Chọn ngẫu nhiên số nước đi từ 2 đến 4 nước[cite: 4]
        num_plies = random.randint(2, 4)
        
        # Lấy ngẫu nhiên các tọa độ không trùng lặp trong vùng 5x5[cite: 4]
        moves_2d = random.sample(center_squares, num_plies)
        
        # Chuyển đổi tọa độ (r, c) sang action (0..120) theo đúng chuẩn CaroGame
        actions = [r * 11 + c for r, c in moves_2d]
        
        openings.append(actions)

    return openings

if __name__ == "__main__":
    # Thiết lập chạy qua giao diện dòng lệnh[cite: 4]
    parser = argparse.ArgumentParser(description="Sinh bộ dữ liệu khai cuộc")
    parser.add_argument("config", nargs="?", default="", help="Đường dẫn config (hiện chưa dùng)")
    args = parser.parse_args()

    openings_data = generate_openings()
    
    # Tạo thư mục và ghi file runs/openings/openings_v1.json[cite: 4]
    out_path = Path("runs/openings/openings_v1.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(openings_data, f, indent=2)
        
    print(f"Đã ghi {len(openings_data)} thế khai cuộc vào {out_path}")