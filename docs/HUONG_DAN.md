# Hướng dẫn sử dụng dự án Caro 11×11

Tài liệu cho mọi thành viên: cài đặt, làm việc với config, chạy thực nghiệm, đọc log,
chia sẻ model qua Hugging Face — và cách các phần sẽ được thêm vào sau này.

Ký hiệu: ✅ đã có và chạy được · 🚧 dự kiến (chưa có code, mô tả là quy ước để mọi người làm giống nhau).

Tài liệu liên quan:
[`engine-caro-11x11.html`](engine-caro-11x11.html) (thiết kế đầy đủ, Phần 01–12) ·
[`lo-trinh-hoc-bitboard.html`](lo-trinh-hoc-bitboard.html) (học bitboard qua ví dụ tính bit) ·
[`TTNTNC_De_tai_cuoi_ky_Game_AI.pdf`](TTNTNC_De_tai_cuoi_ky_Game_AI.pdf) (đề bài).

---

## 0. Trạng thái hiện tại

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| Cây thư mục, `pyproject.toml`, `uv.lock` | ✅ | |
| Config tập trung `configs/` + `caro/config.py` | ✅ | 31 file YAML, lệnh `caro-config` |
| Thư mục run, meta, log JSONL `caro/runlog.py` | ✅ | |
| Model trên Hugging Face `caro/hub.py` | ✅ | lệnh `caro-hub`; cần organization + token |
| Test `tests/` | ✅ | 112 test |
| Engine bitboard `caro/engine/` | ✅ | Python thuần, stride 11; test đối chiếu với bản duyệt mảng |
| Notebook thử engine `notebooks/engine_playground.ipynb` | ✅ | Minh hoạ từng tính năng + bàn cờ bấm được |
| Interface tác tử + cấp 1 `caro/agents/` | ✅ | `Agent`, `Limits`, `Deadline`, `REGISTRY`, `make_agent`, `RandomAgent` |
| Tác tử cấp 2–8 `caro/agents/` | 🚧 | Phần 04–09 |
| Mạng nơ-ron `caro/nn/`, huấn luyện `caro/training/` | 🚧 | Phần 07–09 |
| Arena, giải đấu, thống kê `caro/arena/` | 🚧 | Phần 10 |
| UI `caro/ui/` | 🚧 | Phần 11 |
| CI GitHub Actions | 🚧 | mục 9 bên dưới |

---

## 1. Cài đặt lần đầu

### 1.1 Công cụ

- **Git**
- **uv** — quản lý Python, môi trường ảo và thư viện (thay cho pip + venv + pyenv):
  ```bash
  # Linux / macOS
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  uv tự tải đúng phiên bản Python (≥ 3.11) nếu máy chưa có.

### 1.2 Lấy code và cài môi trường

```bash
git clone <url-repo-github> CaroGame
cd CaroGame
uv sync --group dev              # tạo .venv/, cài numpy, pyyaml, huggingface_hub, pytest, ruff
uv sync --group dev --extra nn   # thêm PyTorch — chỉ khi làm cấp 5–8
uv run pytest                    # phải thấy "112 passed"
```

`uv.lock` khoá chính xác phiên bản mọi thư viện → cả nhóm có môi trường giống hệt nhau.
Không cần `activate`: mọi lệnh chạy qua `uv run …`.

Thêm thư viện mới: `uv add <tên>` (hoặc `uv add --group dev <tên>` cho công cụ phát triển),
rồi commit cả `pyproject.toml` lẫn `uv.lock`.

### 1.3 Token Hugging Face (chỉ cần khi push model hoặc repo model để private)

```bash
cp .env.example .env     # Windows: copy .env.example .env
```

Mở `.env`, dán token **của chính bạn** vào `HF_TOKEN=`. Tạo token tại
huggingface.co → Settings → Access Tokens → **Fine-grained**, chỉ cấp quyền ghi vào repo của organization nhóm.

- `.env` nằm trong `.gitignore` — không bao giờ commit, không gửi qua chat.
- Lỡ lộ token: vào Hugging Face **revoke** ngay rồi tạo token mới.

---

## 2. Bản đồ dự án

```
CaroGame/
├── caro/                  # package Python
│   ├── config.py      ✅  ghép config, override, giải tham chiếu tác tử
│   ├── runlog.py      ✅  thư mục run, meta.json, log chữ, JSONL, derive_seed
│   ├── hub.py         ✅  model trên Hugging Face
│   ├── engine/        ✅  bitboard.py, game.py, zobrist.py, symmetry.py
│   ├── agents/        ✅  base.py, random_agent.py · 🚧 rule_agent.py … puct.py
│   ├── nn/            🚧  features.py, model.py, dataset.py
│   ├── training/      🚧  gen_expert_data.py, train_sl.py, train_pg.py, alphazero_loop.py
│   ├── arena/         🚧  match.py, tournament.py, openings.py, stats.py
│   └── ui/            🚧  server.py, static/
├── configs/           ✅  MỌI tham số nằm ở đây (mục 3)
├── tests/             ✅
├── runs/                  log mọi lần chạy — commit lên GitHub (trừ checkpoint, mục 4.3)
├── notebooks/             engine_playground.ipynb ✅ (thử engine) · analysis.ipynb 🚧 (bảng/biểu đồ báo cáo)
└── docs/
```

**Nguyên tắc phụ thuộc** (giữ cho code dễ test và dễ thay thế):

- `engine` không import gì khác trong dự án.
- `agents` chỉ dùng `engine` (và `nn` cho cấp 5–8).
- `arena`, `training`, `ui` dùng tác tử qua interface chung, không biết bên trong tác tử.
- Không hard-code tham số trong code: đọc từ config.

---

## 3. Config tập trung ✅

### 3.1 Cấu trúc

```
configs/
├── base.yaml          seed, engine, đường dẫn run, log, checkpoint — gốc của mọi config
├── models.yaml        con trỏ model trên Hugging Face (mục 5)
├── budgets/           mode_a_time (cùng ms/nước) · mode_b_fixed (độ sâu / số mô phỏng cố định)
├── agents/            l1_random … l8_alphazero — tham số mặc định từng cấp
├── arena/             round_robin · smoke · bench_positions
├── model/             resnet_small
├── training/          gen_expert_data · sl · pg · alphazero
└── experiments/       file được CHẠY trực tiếp: smoke, main_mode_a/b, a1…a7, a5b, e3
```

### 3.2 Quy tắc ghép

```
defaults (trái → phải, đệ quy)  →  nội dung file  →  override dòng lệnh
```

- Dict ghép đệ quy; list và giá trị đơn bị thay nguyên khối.
- Tác tử: chuỗi `agents/l2_rule`, hoặc `{from: agents/…, id: …, params: {…}, budget: {…}}` để kế thừa và ghi đè.
  Sau khi ghép, mỗi tác tử có thêm `limits` = `budget` chung ghép với `budget` riêng.
- Đường dẫn tương đối trong config luôn tính từ gốc dự án, chạy lệnh ở thư mục nào cũng được.
- Mọi `model:<tên>` phải có trong `configs/models.yaml`, nếu không config báo lỗi ngay.

### 3.3 Xem và thử config

```bash
uv run caro-config experiments/smoke                         # in config đã ghép đầy đủ
uv run caro-config experiments/a3_mcts_sims budget.max_sims=50
uv run caro-config experiments/main_mode_b --hash            # hash dùng để so 2 lần chạy
```

Override dạng `khoá.con=giá_trị`, giá trị đọc theo YAML: `seed=7`, `data.augment_x8=false`,
`train.lr=3e-4`, `arena.openings.plies=[2,4]`, `run.tag=thu_lr`.

> ⚠️ Trong **file** YAML, số mũ phải có dấu chấm: `3.0e-4`, không phải `3e-4` (PyYAML đọc `3e-4` thành chuỗi).
> Override dòng lệnh đã tự xử lý trường hợp này.

### 3.4 Công thức thường dùng

**Đổi tham số cho mọi thực nghiệm** → sửa `base.yaml` (vd. `seed`) hoặc file thành phần (`budgets/`, `agents/`).

**Thêm một ablation mới** → tạo `configs/experiments/a8_ten_moi.yaml`:

```yaml
# A8 — <thay đổi gì>. Giữ: <gì cố định>. Đo: <gì>
defaults: [base, budgets/mode_b_fixed, arena/round_robin]
experiment: {name: a8_ten_moi, rq: [RQ1]}
agents:
  - {from: agents/l4_mcts, id: MCTS-c1,   params: {c_uct: 1.0}}
  - {from: agents/l4_mcts, id: MCTS-c2,   params: {c_uct: 2.0}}
```

Kiểm tra bằng `uv run caro-config experiments/a8_ten_moi`; `uv run pytest` tự nạp thử mọi file trong `experiments/`.

**Thêm một tác tử mới** → `configs/agents/<tên>.yaml` với `id`, `level`, `kind`, `params`
(và code tương ứng, mục 7.2).

---

## 4. Chạy thực nghiệm và log ✅ (khung) / 🚧 (lệnh chạy)

### 4.1 Mỗi lần chạy có một thư mục riêng

```
runs/<kind>/<name>/<YYYYmmdd-HHMMSS>[_<tag>]/
├── config.yaml      config đã ghép đầy đủ — chạy lại được bằng chính file này
├── meta.json        git commit (+ dirty), Python, thư viện, CPU/RAM/GPU, seed, hash config
├── run.log          log chữ
├── games.jsonl      mỗi ván một dòng
├── metrics.jsonl    loss / accuracy / score theo step
├── summary.json     ok | error | interrupted, thời lượng, số bản ghi
└── checkpoints/
runs/<kind>/<name>/latest  → run gần nhất          (chỉ trên máy bạn, không commit)
runs/index.jsonl           sổ đăng ký start/end      (chỉ trên máy bạn, không commit)
```

`kind` lấy từ `experiment.kind`: `tournament`, `bench`, `train`, `datagen`.

### 4.2 Dùng trong code

Mọi lệnh chạy (🚧 arena, training) sẽ theo cùng một khuôn:

```python
from caro.config import load_config
from caro.runlog import Run, derive_seed

cfg = load_config("experiments/smoke", overrides)      # overrides lấy từ dòng lệnh
with Run(cfg) as run:
    run.log.info("bắt đầu")
    seed_x = derive_seed(cfg["seed"], game_id, "X")    # cùng config → cùng ván
    run.log_game({...})                                 # một ván = một dòng games.jsonl
    run.log_metric(step, loss=0.31, acc=0.62)           # metrics.jsonl
    torch.save(state, run.checkpoint_path("best.pt"))
```

Có lỗi hoặc Ctrl+C: `summary.json` ghi `error` / `interrupted`, log vẫn được giữ.

**Bản ghi một ván** — bắt buộc: `game_id, tournament_id, agent_x, agent_o, opening_id, seed, moves, result, termination`;
tuỳ chọn: `time_ms, nodes, sims` (mảng theo từng nước, tắt bằng `log.per_move_stats=false`).
`result` luôn theo góc nhìn X: +1 X thắng, −1 O thắng, 0 hoà. Thiếu trường bắt buộc → báo lỗi ngay.

### 4.3 Run chính thức và run nháp

**Log ở GitHub, model ở Hugging Face.** Mọi file log trong thư mục run đều được commit:

| Trong `runs/` | Lên GitHub? | Lý do |
|---|---|---|
| `config.yaml`, `meta.json`, `run.log`, `games.jsonl`, `metrics.jsonl`, `summary.json` | ✅ | Log thô để tái tạo mọi bảng/biểu đồ |
| `runs/openings/*.json` (bộ khai cuộc) | ✅ | Cần để chạy lại đúng giải |
| `checkpoints/`, mọi `*.pt *.pth *.ckpt *.safetensors *.npz *.npy` | ❌ | Model nằm trên Hugging Face (`caro-hub push`) |
| `runs/data/` (dữ liệu chuyên gia) | ❌ | Quá lớn; sinh lại được từ config + seed |
| `runs/scratch/` | ❌ | Run nháp: thêm `run.root=runs/scratch` khi debug, thử tham số |
| `latest`, `runs/index.jsonl` | ❌ | Chỉ có nghĩa trên từng máy; `index.jsonl` mọi người cùng ghi nối nên sẽ xung đột khi merge |

Sau khi chạy xong:

```bash
git add runs/
git commit -m "run: main_mode_b 20261020-1405"
```

Mỗi run nằm trong thư mục riêng có dấu thời gian nên các thành viên commit log không bao giờ đụng nhau.
Một file log vượt 50 MB → `run.log` có cảnh báo (GitHub từ chối file > 100 MB); khi đó tắt
`log.per_move_stats` hoặc chia giải thành nhiều run.

### 4.4 Đọc log

```python
import pandas as pd
import json
from pathlib import Path

# mọi run của cả nhóm (sau git pull) — đọc từ file đã commit, không dựa vào latest / index.jsonl
runs = pd.DataFrame([{"dir": str(p.parent), **json.loads(p.read_text())}
                     for p in Path("runs").glob("*/*/*/summary.json")])

games = pd.concat(pd.read_json(p, lines=True)
                  for p in Path("runs/tournament/main_mode_b").glob("*/games.jsonl"))
games.groupby(["agent_x", "agent_o"]).result.mean()
```

Hoặc trên terminal: `tail -f runs/train/sl_policy/latest/run.log`.

---

## 5. Model trên Hugging Face ✅

Trọng số **chỉ** nằm trên Hugging Face. GitHub giữ `configs/models.yaml` ghim đúng commit của từng model,
nên checkout commit GitHub nào là chạy đúng model đã dùng ở commit đó.

```yaml
repo: solozy/CaroGame
models:
  l5: {file: l5/best.pt, revision: null}   # null = chưa push
  l6: {file: l6/best.pt, revision: null}
  l8: {file: l8/best.pt, revision: null}   # cấp 7 dùng chung l5
```

### 5.1 Thiết lập một lần (trưởng nhóm)

1. huggingface.co → avatar → **New Organization**; mời thành viên với vai trò **write**.
2. Sửa `repo:` trong `configs/models.yaml` thành `<organization>/<repo>`, commit.
3. `uv run caro-hub init --private` (bỏ `--private` nếu muốn công khai — khi đó người chỉ chạy model không cần token).
4. Nếu private: cấp quyền đọc cho giảng viên.

### 5.2 Ba cách ghi checkpoint trong config

| Ghi | Ý nghĩa | Dùng khi |
|---|---|---|
| `model:l5` | tải bản đã ghim trong `models.yaml` | mặc định cho cấp 5–8 |
| `hf://org/repo/l5/best.pt@<rev>` | tải một bản bất kỳ | so sánh nhanh với bản cũ |
| `runs/train/sl_policy/latest/checkpoints/best.pt` | file cục bộ | vừa train xong, chưa push |

Đấu thử bản cục bộ với bản đã ghim: override dòng lệnh không sửa được từng phần tử trong danh sách `agents`,
nên tạo một experiment nháp (không cần commit):

```yaml
# configs/experiments/nhap_l5_moi.yaml
defaults: [base, budgets/mode_b_fixed, arena/smoke]
agents:
  - agents/l5_policy_sl                                     # bản đã ghim
  - {from: agents/l5_policy_sl, id: L5-moi,
     params: {checkpoint: runs/train/sl_policy/latest/checkpoints/best.pt}}
```

### 5.3 Người chạy model

Không phải làm gì: lần đầu nạp `model:l5` tự tải về `~/.cache/huggingface`, các lần sau dùng cache.

```bash
uv run caro-hub status    # model nào đã push, đã có trong cache chưa
uv run caro-hub pull      # tải trước toàn bộ (vd. trước khi lên lớp không có mạng)
```

### 5.4 Người train

```bash
uv run caro-hub push l5 runs/train/sl_policy/latest           # commit thẳng
uv run caro-hub push l5 runs/train/sl_policy/latest --pr      # mở Pull Request bên HF để nhóm xem trước
git add configs/models.yaml
git commit -m "model: l5 từ run train/sl_policy/<thời gian>"
git push
```

`push` gửi trong **một** commit Hugging Face: `l5/best.pt`, `l5/config.yaml` (config của run — cần để dựng lại mạng)
và `l5/card.json` (git commit, hash config, seed, kết quả), rồi tự ghi commit hash mới vào `models.yaml`.
Cảnh báo nếu run được chạy khi code chưa commit.

Chỉ khi `models.yaml` được commit lên GitHub thì cả nhóm mới dùng bản mới — đây là bước "duyệt" model.

### 5.5 Quay về model cũ

Hugging Face giữ mọi phiên bản, nên chỉ cần đưa con trỏ về:

```bash
git log -- configs/models.yaml                       # tìm commit có revision mong muốn
git checkout <commit> -- configs/models.yaml         # hoặc git revert commit đổi model
```

---

## 6. Quy trình làm việc nhóm

1. Mỗi việc một nhánh: `git switch -c feat/l3-alphabeta`.
2. Trước khi push: `uv run ruff check caro tests` và `uv run pytest`.
3. Mở Pull Request vào `main`, ít nhất một người xem.
4. **Không commit**: `.env`, checkpoint/`.pt`, `runs/scratch/`, `runs/data/`, `.venv/` — `.gitignore` đã lo.
5. **Luôn commit**: `pyproject.toml` + `uv.lock` khi thêm thư viện; `configs/models.yaml` sau khi push model;
   `runs/` sau mỗi lần chạy (trừ run nháp).
6. Chạy thực nghiệm chính thức từ code **đã commit**, để `meta.json` ghi được commit tái lập.

---

## 7. Phát triển các phần tiếp theo 🚧

Thứ tự theo lộ trình ở Phần 12 của tài liệu thiết kế. Mỗi bước: viết code → test → chạy thử bằng config có sẵn.

### 7.1 Engine (tuần 1)

- `caro/engine/bitboard.py`: FULL, START, `has_five`, `dilate`, `winning_squares`, đếm cửa sổ (Phần 01).
  Học trước bằng [`lo-trinh-hoc-bitboard.html`](lo-trinh-hoc-bitboard.html).
- `caro/engine/game.py`: `CaroGame` — `reset, legal_actions, step, is_terminal, result, undo, clone, render`,
  `IllegalMoveError` (mục 02.3).
- `zobrist.py`, `symmetry.py` (8 phép đối xứng D4).
- Đọc `engine.layout` từ config (`stride11` / `stride12`).
- Test: so với cài đặt ngây thơ (duyệt từng ô) trên nhiều bàn ngẫu nhiên.

Thử engine bằng tay: `uv sync --group dev --extra analysis` rồi
`uv run jupyter lab notebooks/engine_playground.ipynb` (hoặc mở trong VS Code, chọn kernel `.venv`).

### 7.2 Tác tử (tuần 2–6) — interface ✅, cấp 2–8 🚧

Mọi tác tử theo interface ở mục 03.2 của tài liệu thiết kế (`caro/agents/base.py`):

```python
class Agent(Protocol):
    name: str
    limits: Limits                         # ngân sách mặc định lấy từ config
    def reset(self, seed: int) -> None: ...
    def select_action(self, game, limits: Limits) -> int: ...
    def last_stats(self) -> dict: ...      # nodes, sims, depth, time_ms, pv, score, reason, visits, policy, value
```

- `Limits(mode, time_ms, max_depth, max_sims)` dựng từ khoá `limits` của config đã ghép; sai kiểu / sai khoá → lỗi ngay.
  `None` = không giới hạn theo tiêu chí đó.
- `limits.deadline(margin_ms)` trả `Deadline` — gọi đầu `select_action`, kiểm tra `deadline.expired()` trong vòng lặp tìm kiếm
  (chế độ A); không có `time_ms` thì không bao giờ hết giờ.
- Khoá `last_stats()` chuẩn nằm ở `caro.agents.STATS_KEYS`; chỉ ghi khoá tác tử có.

Dùng tác tử:

```python
from caro.agents import make_agent

agent = make_agent("agents/l1_random")       # hoặc một mục trong cfg["agents"] sau load_config
agent.reset(derive_seed(cfg["seed"], game_id, "X"))
a = agent.select_action(game, agent.limits)
```

Thêm một tác tử:

1. Viết lớp trong `caro/agents/<tên>.py`, kế thừa `BaseAgent` (có sẵn `name`, `limits`, `self.rng` có seed,
   `self._stats`). `__init__` nhận đúng các khoá trong `params` làm tham số từ khoá, cộng `name`, `limits`
   (chuyển cho `super().__init__`).
2. Thêm một dòng `"<kind>": "caro.agents.<tên>:<Lớp>"` vào `REGISTRY` trong `caro/agents/__init__.py`.
   Lớp chỉ được import khi dùng, nên cấp 1–4 không kéo theo torch.
3. Tạo `configs/agents/<tên>.yaml` với cùng `kind`. `tests/test_agents.py` tự dựng thử mọi config có `kind` đã đăng ký;
   tham số trong `params` mà `__init__` không nhận → `ConfigError`.
4. Tác tử cấp 5–8 nạp trọng số bằng `caro.hub.resolve_checkpoint(params["checkpoint"])` — không tự xử lý đường dẫn.

### 7.3 Arena và thống kê (tuần 2)

Lệnh dự kiến (mọi lệnh nhận `<config> [override …]`):

```bash
uv run python -m caro.arena.openings  experiments/main_mode_b                 # sinh 100 khai cuộc một lần
uv run python -m caro.arena.tournament experiments/smoke                      # kiểm tra pipeline
uv run python -m caro.arena.tournament experiments/main_mode_b
uv run python -m caro.arena.bench      experiments/a1_ab_pruning              # A1, A2
```

`tournament` đọc `arena.*`, ghi `games.jsonl` qua `Run.log_game`, cuối giải ghi bảng score + khoảng Wilson
vào `summary.json`. `arena.pairs` hỗ trợ: mặc định round-robin, `challengers_vs_baselines` (A3, A4), `self_play` (E3).

### 7.4 Huấn luyện (tuần 4–8)

```bash
uv run python -m caro.training.gen_expert_data training/gen_expert_data      # dữ liệu cho cấp 5
uv run python -m caro.training.train_sl        training/sl
uv run caro-hub push l5 runs/train/sl_policy/latest

uv run python -m caro.training.train_pg        training/pg                   # khởi tạo từ model:l5
uv run caro-hub push l6 runs/train/pg_policy/latest

uv run python -m caro.training.alphazero_loop  training/alphazero
uv run caro-hub push l8 runs/train/alphazero/latest
```

Ablation huấn luyện chạy 2 lần với override:

```bash
uv run python -m caro.training.train_pg experiments/a5b_rl_init                                   # từ SL
uv run python -m caro.training.train_pg experiments/a5b_rl_init train.init=random run.tag=from_random
uv run python -m caro.training.train_sl experiments/a6_augment data.augment_x8=false run.tag=no_aug
```

Trong vòng lặp: `run.log_metric(...)` mỗi epoch/iteration; checkpoint lưu bằng `run.checkpoint_path("best.pt")`.

### 7.5 Phân tích (tuần 7–8)

`notebooks/analysis.ipynb` chỉ đọc `runs/**/games.jsonl`, `metrics.jsonl`, `summary.json` đã commit →
sinh **mọi** bảng và biểu đồ trong báo cáo. Không chép số tay vào báo cáo.
Cài thêm: `uv sync --group dev --extra analysis`.

### 7.6 UI (tuần 7)

```bash
uv sync --extra ui --extra nn
uv run uvicorn caro.ui.server:app --reload       # http://127.0.0.1:8000
```

Endpoint theo Phần 11: tạo ván người–máy / máy–máy, xem lại ván từ `games.jsonl`.

---

## 8. Checklist

**Trước một thực nghiệm chính thức**

- [ ] Code đã commit (meta không `dirty`), `uv run pytest` qua
- [ ] Đúng config trong `experiments/`, kiểm tra bằng `caro-config … --hash`
- [ ] Model cần dùng đã push và `models.yaml` đã commit (`caro-hub status`)
- [ ] Sau khi chạy: `git add runs/` và commit log
- [ ] Chọn đúng chế độ ngân sách (A: thời gian, B: cố định)

**Trước khi nộp**

- [ ] `README.md`, tài liệu này, tài liệu thiết kế cập nhật
- [ ] Bảng/biểu đồ sinh lại được từ log bằng notebook
- [ ] Tag cùng tên bên GitHub và Hugging Face (vd. `v1.0`); giảng viên có quyền đọc repo model
- [ ] Trích dẫn nguồn và công cụ AI đã dùng (mục 12 tài liệu thiết kế)

---

## 9. CI GitHub Actions 🚧

Máy GitHub không có GPU → chỉ dùng để kiểm tra code, không để huấn luyện.

Dự kiến `.github/workflows/test.yml`: mỗi push / Pull Request chạy `uv sync`, `ruff`, `pytest`, giải `smoke`.
Job nào cần token (vd. tạo tag Hugging Face khi release) khai báo `environment: huggingface`
và đọc `secrets.HF_TOKEN`; biến môi trường có sẵn được ưu tiên hơn `.env`.

---

## 10. Xử lý sự cố

| Hiện tượng | Nguyên nhân / cách xử lý |
|---|---|
| `model 'l5' chưa được push (revision: null)` | Chưa ai push model đó. Dùng đường dẫn cục bộ hoặc chờ người train push |
| `'model:l9' không có trong configs/models.yaml` | Sai tên, hoặc thêm dòng mới vào `models.yaml` |
| 401 / 403 khi push hoặc tải | Thiếu `HF_TOKEN` trong `.env`, token hết hạn, hoặc token không có quyền ghi repo của organization |
| Tham số `3e-4` trong YAML thành chuỗi | Viết `3.0e-4` |
| Không có `runs/.../latest` (Windows) | Hệ thống không cho tạo symlink → đọc `latest.txt` cùng thư mục |
| `warning: Failed to hardlink files` khi `uv sync` | Vô hại; tắt bằng `export UV_LINK_MODE=copy` |
| `cảnh báo: run được chạy khi code chưa commit` | Commit code rồi chạy lại nếu là run chính thức |
| `… nặng … MB — GitHub từ chối file > 100 MB` | Tắt `log.per_move_stats` hoặc chia nhỏ run; không commit file > 100 MB |
| Muốn bỏ một run nháp đã lỡ chạy vào `runs/` | Xoá thư mục run đó trước khi `git add runs/` |
| `defaults bị vòng` | Hai file config `defaults` lẫn nhau |
| `trùng id tác tử` | Hai biến thể trong cùng experiment phải có `id` khác nhau |
