import argparse
import concurrent.futures
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from caro.agents import make_agent
from caro.agents.base import Limits
from caro.arena.openings import load_openings
from caro.config import load_config
from caro.engine.game import CaroGame, X
from caro.nn.features import board_to_tensor


def format_duration(seconds: float) -> str:
    """1g05p09s / 4p30s / 12s."""
    total = max(0, round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}g{minutes:02d}p{secs:02d}s"
    if minutes:
        return f"{minutes}p{secs:02d}s"
    return f"{secs}s"


def format_progress(
    done: int, total: int, new_done: int, samples: int, elapsed: float,
    now: float | None = None, width: int = 20,
) -> str:
    """Một dòng tiến độ: phần trăm, thanh, số ván, tốc độ, số mẫu, thời gian đã chạy và còn lại.

    done: tổng số ván đã có (kể cả các ván nạp lại từ file tiến độ); new_done: số ván hoàn thành
    trong lần chạy này (dùng để tính tốc độ và thời gian còn lại, không tính ván đã có sẵn).
    """
    fraction = done / total if total else 1.0
    filled = min(width, int(width * fraction))
    parts = [
        f"[{fraction * 100:5.1f}%] {'█' * filled}{'░' * (width - filled)} {done}/{total} ván",
    ]
    rate = new_done / elapsed if elapsed > 0 and new_done > 0 else 0.0
    if rate:
        parts.append(f"{rate:.1f} ván/s")
    parts.append(f"{samples:,} mẫu".replace(",", "."))
    parts.append(f"đã chạy {format_duration(elapsed)}")
    if rate and done < total:
        remaining = (total - done) / rate
        finish = time.localtime((time.time() if now is None else now) + remaining)
        parts.append(f"còn ~{format_duration(remaining)} (xong lúc {time.strftime('%H:%M', finish)})")
    return " | ".join(parts)


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
    # Giữ nguyên cả `params` (vd. backend: auto) thay vì chỉ lấy tên file cấu hình
    expert_cfg = datagen_cfg.get("expert") or "agents/l3_alphabeta"
    expert_depth = int(datagen_cfg.get("expert_depth", 3))
    top_k = int(datagen_cfg.get("top_k", 3))
    prune_root = bool(datagen_cfg.get("prune_root", True))   # False: search đầy đủ từng nước ở gốc
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
        scored = expert.scored_moves(game, limits=limits, top_k=top_k, prune_root=prune_root)

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


def iter_progress(progress_file):
    """Duyệt từng ván trong file tiến độ (mỗi lần một dòng, không giữ cả file trong RAM).

    Bỏ qua dòng cuối bị ghi dở nếu có.
    """
    if not progress_file.exists():
        return
    with progress_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                yield int(item["game_id"]), item["records"]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue


def load_completed_games(progress_file):
    """{game_id: số mẫu} của các ván đã lưu. Chỉ đếm, không giữ mẫu trong RAM."""
    return {game_id: len(records) for game_id, records in iter_progress(progress_file)}


def append_progress(progress_file, game_id, records):
    """Lưu kết quả một ván ngay khi worker hoàn thành."""
    with progress_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(
            {"game_id": game_id, "records": records},
            separators=(",", ":"),
        ))
        f.write("\n")
        f.flush()


def canonical_key(state):
    """Khoá chuẩn hoá D4: nhỏ nhất trong 8 phép đối xứng (state: 4x11x11 uint8)."""
    keys = []
    for k in range(4):
        rot = np.rot90(state, k, axes=(1, 2))
        keys.append(np.ascontiguousarray(rot).tobytes())
        keys.append(np.ascontiguousarray(rot[:, :, ::-1]).tobytes())
    return min(keys)


def write_final_dataset(output_file, progress_file, game_ids, dedup=True):
    """Gom file tiến độ thành .npz gọn: states uint8, policies float32, values, game_ids.

    Đọc từng ván một nên RAM chỉ cỡ kích thước mảng kết quả (~1 KB/mẫu).
    """
    states, policies, values, owners = [], [], [], []
    seen = set()
    wanted = set(game_ids)
    taken = set()
    for game_id, records in iter_progress(progress_file):
        if game_id not in wanted or game_id in taken:
            continue
        taken.add(game_id)
        for rec in records:
            state = np.asarray(rec["state"], dtype=np.uint8)
            if dedup:
                key = canonical_key(state)
                if key in seen:
                    continue
                seen.add(key)
            states.append(state)
            policies.append(np.asarray(rec["policy"], dtype=np.float32))
            values.append(rec["value"])
            owners.append(game_id)

    temp_file = output_file.with_name(output_file.stem + ".tmp.npz")
    np.savez_compressed(
        temp_file,
        states=np.stack(states) if states else np.zeros((0, 4, 11, 11), np.uint8),
        policies=np.stack(policies) if policies else np.zeros((0, 121), np.float32),
        values=np.asarray(values, dtype=np.float32),
        game_ids=np.asarray(owners, dtype=np.int32),
    )
    temp_file.replace(output_file)
    return len(values)


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

    output_file = output_dir / "expert_dataset.npz"
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

    datagen = {
        "độ sâu": datagen_cfg.get("expert_depth", 3),
        "top-k": datagen_cfg.get("top_k", 3),
        "nhiệt độ": datagen_cfg.get("temperature", 0.5),
        "cắt tỉa gốc": datagen_cfg.get("prune_root", True),
    }
    print("=" * 78, flush=True)
    print(
        f"Sinh dữ liệu chuyên gia: {num_games} ván | đã có {len(completed)} | còn {len(pending_ids)} | "
        f"{num_workers} workers",
        flush=True,
    )
    print("Cấu hình: " + ", ".join(f"{k} {v}" for k, v in datagen.items()), flush=True)
    print(f"Ghi tiến độ vào: {progress_file} (dừng bằng Ctrl+C, chạy lại cùng lệnh để tiếp tục)", flush=True)
    print("=" * 78, flush=True)

    log_every = float(datagen_cfg.get("log_every_s", 5))     # giây giữa hai dòng tiến độ
    samples = sum(completed.values())
    failed = 0
    started = time.monotonic()

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
            last_print = started
            for future in concurrent.futures.as_completed(future_to_id):
                game_id = future_to_id[future]
                try:
                    returned_id, records = future.result()
                    append_progress(progress_file, returned_id, records)
                    completed[returned_id] = len(records)
                    finished_this_run += 1
                    samples += len(records)
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    print(f"[LỖI] ván {game_id}: {type(e).__name__}: {e}", flush=True)

                now = time.monotonic()
                if now - last_print >= log_every or len(completed) == num_games:
                    last_print = now
                    print(
                        format_progress(len(completed), num_games, finished_this_run, samples, now - started),
                        flush=True,
                    )

    elapsed = time.monotonic() - started
    if failed:
        print(f"[CẢNH BÁO] {failed} ván bị lỗi, xem các dòng [LỖI] ở trên.", flush=True)

    print("Đang gom dữ liệu thành file .npz (loại trùng đối xứng D4)...", flush=True)
    sample_count = write_final_dataset(
        output_file, progress_file, completed, dedup=datagen_cfg.get("dedup", "d4_canonical") == "d4_canonical"
    )

    print("=" * 78, flush=True)
    print(
        f"Xong: {len(completed)}/{num_games} ván, " + f"{sample_count:,} mẫu".replace(",", ".")
        + f" | thời gian lần chạy này {format_duration(elapsed)}",
        flush=True,
    )
    print(f"File dữ liệu: {output_file}", flush=True)
    if len(completed) < num_games:
        print(
            "Một số ván chưa hoàn thành. File progress JSONL vẫn giữ các ván "
            "đã sinh; có thể chạy lại cùng lệnh để tiếp tục.",
            flush=True,
        )


if __name__ == "__main__":
    main()
