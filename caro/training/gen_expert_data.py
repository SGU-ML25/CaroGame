import argparse
import concurrent.futures
import json
import random
from pathlib import Path

import numpy as np
import torch

from caro.agents import make_agent
from caro.agents.base import Limits
from caro.arena.openings import load_openings
from caro.config import load_config
from caro.engine.game import CaroGame, X
from caro.nn.features import board_to_tensor


def build_policy_and_sample(scored_moves, top_k, temperature, rng):
    """Tạo policy target trên 121 ô và lấy mẫu một nước từ Top-K."""
    if not scored_moves:
        return None, None

    scored_moves = scored_moves[:top_k] if top_k > 0 else scored_moves
    moves = np.asarray([int(move) for move, _ in scored_moves], dtype=np.int64)
    logits = np.asarray([float(score) for _, score in scored_moves], dtype=np.float64)

    temperature = max(float(temperature), 1e-5)
    logits = logits / temperature
    logits -= np.max(logits)

    weights = np.exp(logits)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.full(len(moves), 1.0 / len(moves), dtype=np.float64)
    else:
        probs = weights / total

    policy = np.zeros(121, dtype=np.float32)
    policy[moves] = probs.astype(np.float32)

    selected_index = int(rng.choice(len(moves), p=probs))
    return policy, int(moves[selected_index])


def generate_single_game(game_id, openings, config):
    """Sinh dữ liệu của một ván độc lập."""
    seed = int(config.get("seed", 42)) + int(game_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    datagen_cfg = config.get("datagen", config)
    expert_cfg = (datagen_cfg.get("expert") or {}).get(
        "from", "agents/l3_alphabeta"
    )
    expert_depth = int(datagen_cfg.get("expert_depth", 3))
    top_k = int(datagen_cfg.get("top_k", 3))
    temperature = float(datagen_cfg.get("temperature", 0.5))

    expert = make_agent(expert_cfg)
    expert.reset(seed)

    game = CaroGame()

    if openings:
        opening = random.choice(openings)
        if isinstance(opening, list):
            for move in opening:
                if game.is_terminal():
                    break
                game.step(int(move))

    limits = Limits(max_depth=expert_depth)
    game_history = []

    while not game.is_terminal():
        state_tensor = board_to_tensor(game)
        scored = expert.scored_moves(game, limits=limits, top_k=top_k)

        if not scored:
            legal_moves = game.legal_actions()
            if not legal_moves:
                break
            action = int(legal_moves[0])
            policy_target = np.zeros(121, dtype=np.float32)
            policy_target[action] = 1.0
        else:
            policy_target, action = build_policy_and_sample(
                scored, top_k, temperature, rng
            )

        if action is None or action < 0 or action >= 121:
            break

        game_history.append((state_tensor.numpy(), policy_target))
        game.step(action)

    winner = game.winner
    value_target = 1.0 if winner == X else -1.0 if winner is not None else 0.0

    records = []
    for state, policy in game_history:
        records.append({
            "state": state.tolist(),
            "policy": policy.tolist(),
            "value": value_target,
        })

    return int(game_id), records


def load_completed_games(progress_file):
    """Đọc các ván đã lưu; bỏ qua dòng cuối bị ghi dở nếu có."""
    completed = {}
    if not progress_file.exists():
        return completed

    with progress_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                game_id = int(item["game_id"])
                records = item["records"]
                completed[game_id] = records
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


def append_progress(progress_file, game_id, records):
    """Lưu kết quả một ván ngay khi worker hoàn thành."""
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(
            {"game_id": game_id, "records": records},
            separators=(",", ":"),
        ))
        f.write("\n")
        f.flush()


def write_final_dataset(output_file, completed):
    """Tổng hợp checkpoint thành JSON list tương thích định dạng cũ."""
    all_records = []
    for game_id in sorted(completed):
        all_records.extend(completed[game_id])

    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with temp_file.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, separators=(",", ":"))
    temp_file.replace(output_file)
    return len(all_records)


def main():
    parser = argparse.ArgumentParser(
        description="Sinh dữ liệu expert Alpha-Beta với Top-K, checkpoint và đa worker"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Tên file cấu hình, ví dụ: training/gen_expert_data",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    datagen_cfg = config.get("datagen", config)

    output_dir = Path(datagen_cfg.get("out", "runs/data/expert_v1"))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "expert_dataset.json"
    progress_file = output_dir / "expert_dataset_progress.jsonl"

    num_games = int(datagen_cfg.get("games", 50))
    num_workers = max(1, int(datagen_cfg.get("workers", 4)))

    if num_games <= 0:
        raise ValueError("datagen.games phải lớn hơn 0")

    openings = []
    try:
        openings = load_openings()
    except (FileNotFoundError, OSError) as e:
        print(
            f"Không tìm thấy file khai cuộc, tiếp tục không dùng khai cuộc: {e}",
            flush=True,
        )

    completed = load_completed_games(progress_file)
    pending_ids = [
        game_id for game_id in range(num_games)
        if game_id not in completed
    ]

    print(
        f"Bắt đầu sinh dữ liệu: tổng {num_games} ván, "
        f"đã có {len(completed)} ván, còn {len(pending_ids)} ván, "
        f"{num_workers} workers.",
        flush=True,
    )

    if pending_ids:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            future_to_id = {
                executor.submit(
                    generate_single_game, game_id, openings, config
                ): game_id
                for game_id in pending_ids
            }

            finished_this_run = 0
            for future in concurrent.futures.as_completed(future_to_id):
                game_id = future_to_id[future]
                try:
                    returned_id, records = future.result()
                    append_progress(progress_file, returned_id, records)
                    completed[returned_id] = records
                    finished_this_run += 1

                    if finished_this_run % 10 == 0 or len(completed) == num_games:
                        samples = sum(len(v) for v in completed.values())
                        print(
                            f"Tiến độ: {len(completed)}/{num_games} ván "
                            f"| {samples} mẫu | vừa hoàn thành ván {game_id}",
                            flush=True,
                        )
                except Exception as e:
                    print(
                        f"Lỗi ở ván {game_id}: {type(e).__name__}: {e}",
                        flush=True,
                    )

    sample_count = write_final_dataset(output_file, completed)

    print(
        f"Đã tổng hợp {len(completed)}/{num_games} ván, "
        f"{sample_count} mẫu vào: {output_file}",
        flush=True,
    )
    if len(completed) < num_games:
        print(
            "Một số ván chưa hoàn thành. File progress JSONL vẫn giữ các ván "
            "đã sinh; có thể chạy lại cùng lệnh để tiếp tục.",
            flush=True,
        )


if __name__ == "__main__":
    main()
