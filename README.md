# Caro 11×11 — Game AI nhiều cấp độ

Bàn 11×11, **5 quân liên tiếp trở lên là thắng**, không có luật chặn hai đầu.
Engine dùng bitboard; tác tử từ cấp 1 (ngẫu nhiên) đến cấp 8 (AlphaZero-inspired).


Tài liệu thiết kế: [`docs/engine-caro-11x11.html`](docs/engine-caro-11x11.html) ·
lộ trình học bitboard: [`docs/lo-trinh-hoc-bitboard.html`](docs/lo-trinh-hoc-bitboard.html)

[engine-caro-11x11](https://htmlpreview.github.io/?https://raw.githubusercontent.com/SGU-ML25/CaroGame/refs/heads/main/docs/engine-caro-11x11.html)

**Hướng dẫn sử dụng đầy đủ (cài đặt, config, log, Hugging Face, quy trình nhóm, các phần sắp làm):
[`docs/HUONG_DAN.md`](docs/HUONG_DAN.md)**

## Cài đặt

```bash
uv sync --group dev              # numpy, pyyaml, huggingface_hub, pytest, ruff
uv sync --group dev --extra nn   # thêm PyTorch cho cấp 5–8
uv run pytest
```

## Cấu trúc

```
caro/
├── config.py      # nạp config tập trung (defaults → file → override)
├── runlog.py      # thư mục run, meta, log chữ, JSONL, seed
├── hub.py         # tải / đẩy model Hugging Face theo configs/models.yaml
├── engine/        # bitboard, CaroGame, Zobrist, đối xứng D4
├── agents/        # cấp 1–8, chung interface Agent
├── nn/            # đặc trưng 4×11×11, ResNet policy-value
├── training/      # sinh dữ liệu, SL, policy gradient, vòng lặp AlphaZero
├── arena/         # trận, giải round-robin, khai cuộc, thống kê
└── ui/            # FastAPI + static
configs/           # MỌI tham số thực nghiệm nằm ở đây
tests/
runs/              # log mọi lần chạy — commit lên GitHub (checkpoint thì lên Hugging Face)
notebooks/         # sinh bảng / biểu đồ từ log thô
docs/
```

## Config tập trung

```
configs/
├── base.yaml            # seed, engine, đường dẫn run, cấu hình log, checkpoint
├── models.yaml          # con trỏ model trên Hugging Face (tên → file + revision)
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
runs/<kind>/<name>/latest → run gần nhất      (chỉ trên máy, không commit)
runs/index.jsonl           # sổ đăng ký start/end (chỉ trên máy, không commit)
```

**Log ở GitHub, model ở Hugging Face:** chạy xong thì `git add runs/` và commit. `.gitignore` tự loại
`checkpoints/`, file nhị phân (`*.pt`, `*.npz`…), `runs/data/`, `runs/scratch/` (run nháp: `run.root=runs/scratch`),
`latest` và `runs/index.jsonl`.

Bản ghi một ván (bắt buộc): `game_id, tournament_id, agent_x, agent_o, opening_id, seed, moves, result, termination`;
tuỳ chọn theo nước: `time_ms, nodes, sims` (tắt bằng `log.per_move_stats=false`).
`result` luôn theo góc nhìn X (+1 X thắng, −1 O thắng, 0 hoà).

Seed của từng tác tử trong từng ván: `derive_seed(cfg["seed"], game_id, color)` — ổn định giữa các lần chạy.

## Model trên Hugging Face

Trọng số **chỉ** nằm trên Hugging Face; GitHub giữ `configs/models.yaml` ghim đúng commit của từng model.
Config tham chiếu bằng `model:<tên>`, nên checkout commit GitHub nào là chạy đúng model của commit đó.

```yaml
# configs/models.yaml
repo: solozy/CaroGame
models:
  l5: {file: l5/best.pt, revision: 3f2a9c1e…}
```

| Trong config | Nghĩa |
|---|---|
| `checkpoint: model:l5` | Tải bản đã ghim trong `models.yaml` (mặc định cho cấp 5–8) |
| `checkpoint: hf://org/repo/l5/best.pt@rev` | Tải trực tiếp một bản bất kỳ để thử |
| `checkpoint: runs/train/sl_policy/latest/checkpoints/best.pt` | File cục bộ, vừa train xong chưa push |

**Lần đầu (mỗi thành viên)**

1. Trưởng nhóm tạo Organization trên huggingface.co, mời thành viên với vai trò *write*,
   sửa `repo:` trong `configs/models.yaml`, rồi chạy `uv run caro-hub init --private` (bỏ `--private` nếu public).
2. Mỗi người tạo token **fine-grained** của riêng mình (quyền ghi repo của organization), rồi:
   ```bash
   cp .env.example .env    # dán token vào HF_TOKEN=…  (.env không bao giờ được commit)
   ```

**Người chạy model** — không cần làm gì thêm: lần đầu nạp `model:l5` sẽ tự tải về
`~/.cache/huggingface`, các lần sau dùng cache. Tải trước cho cả bộ: `uv run caro-hub pull`.

**Người train**

```bash
uv run caro-hub push l5 runs/train/sl_policy/latest      # hoặc --pr để mở PR bên HF
git add configs/models.yaml && git commit -m "model: l5 từ run …"
```

`push` đẩy `best.pt` + `config.yaml` của run + `card.json` (git commit, hash config, seed, kết quả)
trong **một** commit HF, rồi tự ghi commit hash mới vào `models.yaml`. Commit file đó lên GitHub
là cả nhóm dùng bản mới. Xem tình trạng: `uv run caro-hub status`.

GitHub Actions: đặt `HF_TOKEN` trong Environment hoặc Repository secrets; biến môi trường có sẵn
được ưu tiên hơn `.env`.

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
