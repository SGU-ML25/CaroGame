# Caro 11×11 — Game AI nhiều cấp độ

Bàn 11×11, **5 quân liên tiếp trở lên là thắng**, không có luật chặn hai đầu.
Engine dùng bitboard; tác tử từ cấp 1 (ngẫu nhiên) đến cấp 8 (AlphaZero-inspired).

Tài liệu thiết kế: [`docs/engine-caro-11x11.html`](docs/engine-caro-11x11.html) ·
lộ trình học bitboard: [`docs/lo-trinh-hoc-bitboard.html`](docs/lo-trinh-hoc-bitboard.html)

## Cài đặt

```bash
uv sync --group dev              # numpy, pyyaml, pytest, ruff
uv sync --group dev --extra nn   # thêm PyTorch cho cấp 5–8
uv run pytest
```

## Cấu trúc

```
caro/
├── config.py      # nạp config tập trung (defaults → file → override)
├── runlog.py      # thư mục run, meta, log chữ, JSONL, seed
├── engine/        # bitboard, CaroGame, Zobrist, đối xứng D4
├── agents/        # cấp 1–8, chung interface Agent
├── nn/            # đặc trưng 4×11×11, ResNet policy-value
├── training/      # sinh dữ liệu, SL, policy gradient, vòng lặp AlphaZero
├── arena/         # trận, giải round-robin, khai cuộc, thống kê
└── ui/            # FastAPI + static
configs/           # MỌI tham số thực nghiệm nằm ở đây
tests/
runs/              # kết quả chạy (không commit)
notebooks/         # sinh bảng / biểu đồ từ log thô
docs/
```

## Config tập trung

```
configs/
├── base.yaml            # seed, engine, đường dẫn run, cấu hình log, checkpoint
├── budgets/             # mode_a_time (cùng ms/nước) · mode_b_fixed (độ sâu / số mô phỏng cố định)
├── agents/              # l1_random … l8_alphazero — tham số mặc định của từng cấp
├── arena/               # round_robin (200 ván/cặp, 100 khai cuộc) · smoke · bench_positions
├── model/               # resnet_small (64 kênh × 6 khối)
├── training/            # gen_expert_data · sl · pg · alphazero
└── experiments/         # main_mode_a/b, a1…a7, e3, smoke — file được chạy trực tiếp
```

Một config được ghép theo thứ tự **`defaults` → nội dung file → override dòng lệnh**.
Dict ghép đệ quy, list bị thay thế nguyên khối.

```yaml
# configs/experiments/a3_mcts_sims.yaml
defaults: [base, budgets/mode_b_fixed, arena/round_robin]
experiment: {name: a3_mcts_sims, rq: [RQ1]}
agents:
  - {from: agents/l4_mcts, id: MCTS-100, budget: {max_sims: 100}}   # kế thừa + ghi đè
  - {from: agents/l4_mcts, id: MCTS-400, budget: {max_sims: 400}}
```

- Tham chiếu tác tử là chuỗi `agents/l2_rule` hoặc `{from: ..., id: ..., params: ..., budget: ...}`.
  Mỗi tác tử sau khi giải có thêm `limits` = budget chung ghép với budget riêng.
- Override: `khoá.con=giá_trị` (giá trị đọc bằng YAML nên có kiểu: `3e-4`, `false`, `[1,2]`).
- Đường dẫn tương đối trong config luôn tính từ gốc dự án.

Xem config đã ghép:

```bash
uv run caro-config experiments/smoke
uv run caro-config experiments/a5b_rl_init train.init=random run.tag=from_random
uv run caro-config experiments/main_mode_b --hash
```

## Thư mục run và log

```python
from caro.config import load_config
from caro.runlog import Run, derive_seed

cfg = load_config("experiments/smoke", ["seed=7"])
with Run(cfg) as run:
    run.log.info("bắt đầu giải")
    run.log_game({...})              # một ván = một dòng games.jsonl
    run.log_metric(step, loss=0.31)  # metrics.jsonl
```

```
runs/<kind>/<name>/<YYYYmmdd-HHMMSS>[_<tag>]/
├── config.yaml      # config đã ghép — chạy lại được bằng chính file này
├── meta.json        # git commit (+dirty), Python, thư viện, CPU/RAM/GPU, seed, hash config
├── run.log
├── games.jsonl
├── metrics.jsonl
├── summary.json     # ok | error | interrupted, thời lượng, số bản ghi
└── checkpoints/
runs/<kind>/<name>/latest → run gần nhất
runs/index.jsonl           # sổ đăng ký: dòng start/end cho mọi run
```

Bản ghi một ván (bắt buộc): `game_id, tournament_id, agent_x, agent_o, opening_id, seed, moves, result, termination`;
tuỳ chọn theo nước: `time_ms, nodes, sims` (tắt bằng `log.per_move_stats=false`).
`result` luôn theo góc nhìn X (+1 X thắng, −1 O thắng, 0 hoà).

Seed của từng tác tử trong từng ván: `derive_seed(cfg["seed"], game_id, color)` — ổn định giữa các lần chạy.

## Thực nghiệm

| Config | Nội dung | RQ |
|---|---|---|
| `experiments/smoke` | L1 vs L2, 10 ván — kiểm tra pipeline | |
| `experiments/main_mode_a` · `main_mode_b` | Giải chính 8 cấp, 2 chế độ ngân sách | RQ1, RQ2, RQ4 |
| `experiments/a1_ab_pruning` | Alpha-Beta bật / tắt cắt tỉa | RQ3 |
| `experiments/a2_ordering_tt` | Sắp xếp nước, bảng chuyển vị | RQ3 |
| `experiments/a3_mcts_sims` | MCTS 100 / 400 / 1600 / 6400 mô phỏng | RQ1 |
| `experiments/a4_rollout` | Rollout ngẫu nhiên / có luật | RQ1 |
| `experiments/a5_policy_vs_puct` | Policy-only vs policy + MCTS | RQ2 |
| `experiments/a5b_rl_init` | RL khởi tạo từ SL / ngẫu nhiên | RQ2 |
| `experiments/a6_augment` | Có / không augmentation ×8 | |
| `experiments/a7_eval_weights` | Heuristic mặc định / bỏ phòng thủ | |
| `experiments/e3_first_move` | Lợi thế đi trước | RQ4 |
