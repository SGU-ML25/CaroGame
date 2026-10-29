import argparse
import json
import os
import concurrent.futures
from pathlib import Path

# Các module của dự án
from caro.config import load_config
from caro.runlog import Run
from caro.agents import make_agent
from caro.arena.match import play_match

def worker_play(match_info):
    """Hàm chạy vòng đời 1 ván cờ độc lập trên một tiến trình (worker)."""
    game_id, agent_x_cfg, agent_o_cfg, opening_id, opening_moves, seed = match_info
    
    # Đo RAM đỉnh (hỗ trợ tự động trên Linux/CI, trên Windows trả về 0 để tránh lỗi)[cite: 4]
    peak_ram_mb = 0.0
    try:
        import resource
        ram_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        ram_before = 0
        
    # 1. Khởi tạo tác tử
    agent_x = make_agent(agent_x_cfg)
    agent_o = make_agent(agent_o_cfg)
    
    # 2. Gọi hàm chạy ván đấu (match.py)
    result = play_match(
        agent_x=agent_x,
        agent_o=agent_o,
        opening_moves=opening_moves,
        seed=seed,
        game_id=game_id
    )
    
    # Tính RAM tiêu thụ
    try:
        import resource
        ram_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_ram_mb = (ram_after - ram_before) / 1024.0  # Chuyển KB sang MB
    except ImportError:
        peak_ram_mb = 0.0

    # 3. Bổ sung các trường thông tin bắt buộc còn thiếu vào log
    result.update({
        "game_id": game_id,
        "agent_x": agent_x.name,
        "agent_o": agent_o.name,
        "opening_id": opening_id,
        "peak_ram_mb": round(peak_ram_mb, 2)
    })
    
    return result

def main():
    # Khởi tạo giao diện dòng lệnh CLI[cite: 4]
    parser = argparse.ArgumentParser(description="Chạy giải đấu Arena")
    parser.add_argument("config", help="Đường dẫn file config (vd: experiments/smoke)")
    parser.add_argument("overrides", nargs="*", help="Các tham số override dạng key=value")
    args = parser.parse_args()

    # Nạp cấu hình
    cfg = load_config(args.config, args.overrides)
    arena_cfg = cfg.get("arena", {})
    agents_cfg = cfg.get("agents", [])
    
    # Đọc file 100 thế khai cuộc đã sinh từ bước trước
    openings_path = Path("runs/openings/openings_v1.json")
    if not openings_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {openings_path}. Vui lòng chạy openings.py trước.")
    with open(openings_path, "r", encoding="utf-8") as f:
        openings = json.load(f)

    # Nếu chạy test (smoke), config có thể cấu hình số lượng khai cuộc ít đi để chạy cho nhanh
    num_openings = arena_cfg.get("num_openings", len(openings))
    used_openings = openings[:num_openings]

    matches = []
    game_id = 0
    seed = cfg.get("seed", 42)
    
    # Thiết lập cặp đấu: Mặc định là Round-robin có đổi màu quân[cite: 4]
    for i, a1_cfg in enumerate(agents_cfg):
        for j, a2_cfg in enumerate(agents_cfg):
            if i == j: 
                continue # Tạm thời round-robin không tự đấu với chính mình
            
            for op_id, op_moves in enumerate(used_openings):
                # Lưu ý: a1_cfg cầm X, a2_cfg cầm O. Vòng lặp i,j tự động tạo 2 lượt đổi màu.
                matches.append((game_id, a1_cfg, a2_cfg, op_id, op_moves, seed))
                game_id += 1

    # Ưu tiên lấy workers từ config, nếu không có thì lấy số nhân CPU trừ 1 để chừa tài nguyên cho máy[cite: 4]
    workers = arena_cfg.get("workers", max(1, (os.cpu_count() or 2) - 1))
    scores = {}

    with Run(cfg) as run:
        run.log.info(f"Bắt đầu giải đấu: {len(matches)} ván cờ, dùng {workers} workers song song.")
        
        # Khởi chạy đa tiến trình song song
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker_play, m): m for m in matches}
            
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                res["tournament_id"] = run.id
                
                # Khởi tạo điểm cho tác tử nếu chưa có trong từ điển
                ax, ao = res["agent_x"], res["agent_o"]
                scores.setdefault(ax, 0.0)
                scores.setdefault(ao, 0.0)
                
                # Tính điểm theo góc nhìn X (1: X thắng, -1: O thắng, 0: Hoà)
                if res["result"] == 1:
                    scores[ax] += 1.0
                elif res["result"] == -1:
                    scores[ao] += 1.0
                elif res["result"] == 0:
                    scores[ax] += 0.5
                    scores[ao] += 0.5

                # Ghi bản ghi ván đấu xuống games.jsonl[cite: 4]
                run.log_game(res)
        
        # Kết thúc giải: ghi bảng điểm vào summary.json[cite: 4]
        run.log.info(f"Bảng điểm chung cuộc: {scores}")
        run.finish(status="ok", score=scores)

if __name__ == "__main__":
    main()