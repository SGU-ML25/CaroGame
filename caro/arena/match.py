import time
from typing import Any

from caro.engine.game import CaroGame
from caro.agents.base import Agent
from caro.runlog import derive_seed

def play_match(
    agent_x: Agent,
    agent_o: Agent,
    opening_moves: list[int],
    seed: int,
    game_id: int,
    max_plies: int = 240
) -> dict[str, Any]:
    """Thực thi một ván cờ từ danh sách nước đi khai cuộc cho trước."""
    
    # 1. Khởi tạo bàn cờ và đi các nước khai cuộc
    game = CaroGame.from_moves(opening_moves)
    
    # 2. Cấp seed ngẫu nhiên cho 2 tác tử theo đúng tài liệu[cite: 4]
    seed_x = derive_seed(seed, game_id, "X")
    seed_o = derive_seed(seed, game_id, "O")
    agent_x.reset(seed_x)
    agent_o.reset(seed_o)
    
    agents = {0: agent_x, 1: agent_o}  # CaroGame.X = 0, CaroGame.O = 1
    
    moves = list(opening_moves)
    
    # Danh sách đo lường từng nước đi[cite: 4]
    time_ms_list = []
    nodes_list = []
    sims_list = []

    # 3. Vòng lặp ván đấu
    while not game.is_terminal() and len(moves) < max_plies:
        current_agent = agents[game.side]
        
        # Đồng hồ đo thời gian tự đếm ngoài agent
        start_time = time.perf_counter()
        
        # Agent chọn nước đi theo ngân sách (limits) quy định trong config
        action = current_agent.select_action(game.clone(), current_agent.limits)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        game.step(action)
        moves.append(action)
        
        # Ghi nhận các chỉ số thống kê từ agent (nodes, sims) theo last_stats()
        stats = current_agent.last_stats()
        time_ms_list.append(round(elapsed_ms, 2))
        nodes_list.append(stats.get("nodes", 0))
        sims_list.append(stats.get("sims", 0))

    # 4. Trả về kết quả từ điển khớp với GAME_FIELDS trong runlog.py
    # Ghi chú: Các trường 'game_id', 'tournament_id', 'opening_id', 'agent_x', 'agent_o' sẽ được module tournament.py điền nốt.
    termination = game.termination()
    if not game.is_terminal() and len(moves) >= max_plies:
        termination = "max_plies"
        
    return {
        "seed": seed,
        "moves": moves,
        "result": game.result(),
        "termination": termination,
        "time_ms": time_ms_list,
        "nodes": nodes_list,
        "sims": sims_list
    }