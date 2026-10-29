import os
import math
import random
import pandas as pd
import numpy as np
from collections import defaultdict

def wilson(k, n, z=1.96):
    """Khoảng tin cậy Wilson 95% cho tỷ lệ k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z / den * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return center - half, center + half

def bootstrap(match_pairs, n_boot=10000):
    """
    Bootstrap theo cặp ván cùng khai cuộc cho score (arena.stats.bootstrap).
    match_pairs: danh sách các tuple điểm (score_v1, score_v2) trong cùng một cặp ván đổi màu.
    """
    n_pairs = len(match_pairs)
    if n_pairs == 0:
        return 0.0, 0.0
    
    scores = np.zeros(n_boot)
    for i in range(n_boot):
        sample_idx = [random.randint(0, n_pairs - 1) for _ in range(n_pairs)]
        total_score = 0.0
        for idx in sample_idx:
            total_score += sum(match_pairs[idx])
        scores[i] = total_score / (n_pairs * 2)
        
    scores.sort()
    return scores[int(0.025 * n_boot)], scores[int(0.975 * n_boot)]

def elo(matches, base_rating=1000, k_factor=32):
    """Elo / Bradley-Terry từ toàn bộ ma trận (arena.stats.elo)."""
    ratings = defaultdict(lambda: base_rating)
    for agent_x, agent_o, result in matches:
        rx = ratings[agent_x]
        ro = ratings[agent_o]
        
        ex = 1 / (1 + 10 ** ((ro - rx) / 400))
        eo = 1 / (1 + 10 ** ((rx - ro) / 400))
        
        sx = 1.0 if result == 1 else (0.5 if result == 0 else 0.0)
        so = 1.0 - sx
        
        ratings[agent_x] += k_factor * (sx - ex)
        ratings[agent_o] += k_factor * (so - eo)
        
    return dict(ratings)

def detailed_record_by_turn(games_df):
    """Thắng / hoà / thua từ góc nhìn tác tử hàng, tách theo đi trước (X) / đi sau (O)."""
    records = {}
    agents = sorted(pd.unique(games_df[['agent_x', 'agent_o']].values.ravel()))
    
    for agent in agents:
        as_x = games_df[games_df['agent_x'] == agent]
        wins_x = len(as_x[as_x['result'] == 1])
        draws_x = len(as_x[as_x['result'] == 0])
        losses_x = len(as_x[as_x['result'] == -1])
        
        as_o = games_df[games_df['agent_o'] == agent]
        wins_o = len(as_o[as_o['result'] == -1])
        draws_o = len(as_o[as_o['result'] == 0])
        losses_o = len(as_o[as_o['result'] == 1])
        
        total_games = len(as_x) + len(as_o)
        total_wins = wins_x + wins_o
        total_draws = draws_x + draws_o
        total_losses = losses_x + losses_o
        
        score = (total_wins + 0.5 * total_draws) / total_games if total_games > 0 else 0.0
        
        records[agent] = {
            "first_hand_X": {"wins": wins_x, "draws": draws_x, "losses": losses_x},
            "second_hand_O": {"wins": wins_o, "draws": draws_o, "losses": losses_o},
            "total": {"wins": total_wins, "draws": total_draws, "losses": total_losses, "score": score}
        }
    return records

def calculate_performance(games_df):
    """Thời gian TB + p95 mỗi nước, số nút / mô phỏng TB."""
    stats = {}
    agents = sorted(pd.unique(games_df[['agent_x', 'agent_o']].values.ravel()))
    
    for agent in agents:
        agent_x_games = games_df[games_df['agent_x'] == agent]
        agent_o_games = games_df[games_df['agent_o'] == agent]
        
        all_times, all_nodes = [], []
        
        for _, row in agent_x_games.iterrows():
            all_times.extend(row.get('time_ms', [])[0::2])
            all_nodes.extend(row.get('nodes', [])[0::2] or row.get('sims', [])[0::2])
            
        for _, row in agent_o_games.iterrows():
            all_times.extend(row.get('time_ms', [])[1::2])
            all_nodes.extend(row.get('nodes', [])[1::2] or row.get('sims', [])[1::2])
            
        stats[agent] = {
            "time_mean_ms": np.mean(all_times) if all_times else 0.0,
            "time_p95_ms": np.percentile(all_times, 95) if all_times else 0.0,
            "nodes_mean": np.mean(all_nodes) if all_nodes else 0.0
        }
    return stats

def generate_matrix(games_df, out_dir="runs/tournament"):
    """Ma trận đối đầu (mục 10.3) xuất CSV / markdown vào thư mục runs/tournament."""
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "matrix.csv")
    out_md = os.path.join(out_dir, "matrix.md")
    
    agents = sorted(pd.unique(games_df[['agent_x', 'agent_o']].values.ravel()))
    matrix = pd.DataFrame(index=agents, columns=agents)
    
    for a1 in agents:
        for a2 in agents:
            if a1 == a2:
                matrix.loc[a1, a2] = "×"
                continue
                
            m_a1_x = games_df[(games_df['agent_x'] == a1) & (games_df['agent_o'] == a2)]
            m_a1_o = games_df[(games_df['agent_o'] == a1) & (games_df['agent_x'] == a2)]
            
            wins = len(m_a1_x[m_a1_x['result'] == 1]) + len(m_a1_o[m_a1_o['result'] == -1])
            draws = len(m_a1_x[m_a1_x['result'] == 0]) + len(m_a1_o[m_a1_o['result'] == 0])
            total = len(m_a1_x) + len(m_a1_o)
            
            if total == 0:
                matrix.loc[a1, a2] = "—"
                continue
                
            score = (wins + 0.5 * draws) / total
            low, high = wilson(wins + 0.5 * draws, total)
            
            matrix.loc[a1, a2] = f"{score:.3f} ({low:.3f}-{high:.3f})"
            
    matrix.to_csv(out_csv)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(matrix.to_markdown())
        
    print(f"Đã xuất thành công ma trận đối đầu vào thư mục: {out_dir}/")
    return matrix

if __name__ == "__main__":
    print("=== KIỂM THỬ VỚI SỐ LIỆU DỰNG TAY (ISSUE 18) ===")
    
    # 1. Test Wilson
    w_low, w_high = wilson(120, 200)
    print(f"Wilson 95% (120/200): {w_low:.3f} - {w_high:.3f} (Kỳ vọng: 0.531-0.665)")
    
    # 2. Test Bootstrap
    dummy_pairs = [(1.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.0, 0.0)]
    b_low, b_high = bootstrap(dummy_pairs, n_boot=1000)
    print(f"Bootstrap Score 95%: {b_low:.3f} - {b_high:.3f}")
    
    # 3. Test Elo
    dummy_matches = [("L1", "L2", -1), ("L2", "L1", 1), ("L3", "L2", 1)]
    ratings = elo(dummy_matches)
    print("Elo Ratings:", ratings)
    
    # 4. Test xuất Ma trận đối đầu ra thư mục runs/tournament
    mock_data = {
        'game_id': [1, 2, 3, 4],
        'agent_x': ['L1', 'L2', 'L1', 'L3'],
        'agent_o': ['L2', 'L1', 'L3', 'L1'],
        'result': [1, -1, 0, 1],
        'time_ms': [[100, 120], [90, 110], [200, 220], [80, 95]],
        'nodes': [[50, 60], [40, 45], [100, 110], [30, 35]]
    }
    mock_df = pd.DataFrame(mock_data)
    generate_matrix(mock_df)