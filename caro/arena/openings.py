
import argparse
import json
import random
from pathlib import Path

DEFAULT_OPENINGS_FILE = Path("runs/openings/openings_v1.json")
BOARD_SIZE = 11


def generate_openings(
    num_openings=100,
    plies=(2, 4),
    region=5,
    seed=None,
):
    """Sinh danh sách khai cuộc trong một vùng vuông ở giữa bàn cờ."""
    num_openings = int(num_openings)
    region = int(region)

    if num_openings < 0:
        raise ValueError("num_openings phải lớn hơn hoặc bằng 0")

    if not 1 <= region <= BOARD_SIZE:
        raise ValueError(f"region phải nằm trong khoảng 1..{BOARD_SIZE}")

    if len(plies) != 2:
        raise ValueError("plies phải có dạng [min_plies, max_plies]")

    min_plies, max_plies = map(int, plies)

    if min_plies < 0 or max_plies < min_plies:
        raise ValueError("Khoảng plies không hợp lệ")

    if max_plies > region * region:
        raise ValueError("Số plies tối đa vượt quá số ô trong vùng khai cuộc")

    rng = random.Random(seed)

    start = (BOARD_SIZE - region) // 2
    end = start + region

    center_squares = [
        (r, c)
        for r in range(start, end)
        for c in range(start, end)
    ]

    openings = []

    for _ in range(num_openings):
        num_plies = rng.randint(min_plies, max_plies)
        moves_2d = rng.sample(center_squares, num_plies)
        actions = [r * BOARD_SIZE + c for r, c in moves_2d]
        openings.append(actions)

    return openings


def load_openings(config=None):
    """
    Tải hoặc sinh khai cuộc.

    Không truyền config:
        Giữ hành vi cũ để tương thích với pipeline tạo dữ liệu L5.

    Có config:
        Đọc count, plies, region, seed và file từ cấu hình Arena.
        Nếu file là None, chỉ sinh trong bộ nhớ.
    """
    if config is None:
        if DEFAULT_OPENINGS_FILE.exists():
            with DEFAULT_OPENINGS_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)

        return generate_openings()

    count = config.get("count", 100)
    plies = config.get("plies", [2, 4])
    region = config.get("region", 5)
    seed = config.get("seed")
    file_path = config.get("file")

    if file_path is None:
        return generate_openings(
            num_openings=count,
            plies=plies,
            region=region,
            seed=seed,
        )

    out_path = Path(file_path)

    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    openings = generate_openings(
        num_openings=count,
        plies=plies,
        region=region,
        seed=seed,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(openings, f, indent=2)

    return openings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh bộ dữ liệu khai cuộc")
    parser.add_argument("config", nargs="?", default="", help="Đường dẫn config (hiện chưa dùng)")
    args = parser.parse_args()

    openings_data = generate_openings()

    DEFAULT_OPENINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with DEFAULT_OPENINGS_FILE.open("w", encoding="utf-8") as f:
        json.dump(openings_data, f, indent=2)

    print(f"Đã ghi {len(openings_data)} thế khai cuộc vào {DEFAULT_OPENINGS_FILE}")
