# Đối thủ ngoài KataGomo cho Caro 11×11

Tài liệu này giải thích **đầy đủ** phần tích hợp KataGomo: vì sao cần, đã chọn engine/model nào,
luật của đồ án khớp hay không, kiến trúc tác tử, cách cài đặt, cách chạy, cách kiểm thử và những
hạn chế còn lại. Mục tiêu là đủ chi tiết để một người khác đọc xong có thể cài lại, chạy lại và
đánh giá kết quả mà không cần hỏi thêm.

KataGomo là **đối thủ ngoài để đối chiếu**, không thuộc thang L1–L8 và không được dùng trong các
bảng kết quả chính thức của đồ án.

---

## 1. Vì sao cần một đối thủ ngoài?

Các tác tử L1–L8 đều do đồ án tự cài. Vì vậy, khi L8 thắng L3 hay L7, ta chỉ biết nó thắng
**trong hệ sinh thái của chính đồ án**. Một engine ngoài trả lời câu hỏi khác:

> Nếu đặt tác tử của đồ án cạnh một chương trình Caro/Gomoku đã được huấn luyện độc lập,
> khoảng cách còn bao xa?

Đối thủ ngoài có ích ở ba điểm:

1. **Kiểm tra tính đúng của luật.** Một engine mạnh nhưng hiểu sai luật sẽ bộc lộ ngay các thế
   `OXXXXXO`, dãy 6, thắng ở biên — những chỗ dễ sai nhất khi tự cài.
2. **Kiểm tra chất lượng nước đi.** Nếu tác tử thắng các đối thủ nội bộ nhưng thua engine ngoài
   ở cùng ngân sách, vấn đề nằm ở chất lượng tìm kiếm/chính sách, không phải ở cách xếp hạng nội bộ.
3. **Tạo một mốc tham chiếu.** Không cần coi điểm với engine ngoài là chân lý, nhưng nó là một
   mốc độc lập để so các phiên bản L8 sau này.

Đây không phải một tác tử mới trong thang điểm. Nó không tham gia round-robin L1–L8, không được
dùng để chọn checkpoint, và không thay thế các thí nghiệm A1–A8.

---

## 2. Luật của đồ án và hai cách hiểu “Caro”

### 2.1 Luật đồ án

`configs/base.yaml` ghi:

```yaml
engine:
  size: 11
  win_length: 5             # ≥ 5 quân liên tiếp là thắng, không luật chặn 2 đầu
```

`caro/engine/bitboard.py` cài đúng như vậy: `has_five()` chỉ cần **5 quân liên tiếp** theo một
hướng. Hệ quả:

- `XXXXX` thắng, dù hai đầu có quân đối phương hay không.
- `OXXXXXO` **thắng** với bên X.
- `XXXXXX` (6 quân) cũng thắng, vì bên trong nó có dãy 5.
- Không có luật cấm dãy 6, không có luật cấm thắng ở biên.

Đây là luật **freestyle gomoku**, không phải luật Caro thi đấu kiểu “chặn hai đầu không tính”.

### 2.2 Vì sao không dùng thẳng nhánh `Caro2024`

Nhánh `Caro2024` của KataGomo có hàm `isFive_oneLine` trong `cpp/game/gamelogic.cpp`:

```cpp
if (myConNum == 5)
{
  return t1 != 0 || t2 != 0;  // oxxxxxo
}
```

`t1`/`t2` là loại ô ở hai đầu dãy. Với `OXXXXXO`, cả hai đầu là quân đối phương nên `t1 == 0`
và `t2 == 0`; hàm trả `false`. Nghĩa là Caro2024 gốc **không** coi `OXXXXXO` là thắng.

Điều này trái luật đồ án. Hậu quả khi đấu thử ban đầu:

- Đồ án kết thúc ván khi một bên có 5 quân, kể cả bị chặn hai đầu.
- KataGomo vẫn tưởng ván tiếp tục, tiếp tục đánh và báo giá trị thắng cho bên vừa thua.
- Có ván engine “thua” chỉ vì nó không nhận ra đối thủ đã thắng.

Đây là lỗi **luật**, không phải lỗi mạng hay lỗi tìm kiếm.

### 2.3 Vì sao không dùng thẳng nhánh `Gom2024`

`Gom2024` có `basicRule = FREESTYLE`, đúng luật 5 quân. Nhưng khi thử trên bàn 11×11, engine
không nhận dãy **5 quân dọc** là thắng. Ví dụ đặt X ở `F1, F2, F3, F4, F5` rồi cho O đi:

- `genmove` vẫn trả một nước bình thường;
- `Winrate` của bên đã thua vẫn còn hơn 40%, không phải 0%.

Cùng thế đó, nhánh `Caro2024` nhận đúng (trả `pass`/giá trị thua). Vì vậy `Gom2024` không dùng
nguyên bản được, dù luật danh nghĩa là freestyle.

### 2.4 Cách chọn: `Caro2024` + một patch nhỏ

Ta dùng `Caro2024` vì:

- nhận đúng dãy 5 ngang/dọc/chéo, kể cả ở biên;
- hỗ trợ bàn 11×11;
- model hiện có tải và chạy được.

Chỉ cần vá đúng một nhánh: khi `SIXWINRULE_ALWAYS` được bật, dãy 5 phải thắng **bất kể hai đầu**.
Patch nằm ở `configs/katagomo/freestyle-five.patch`:

```diff
   if (myConNum == 5)
   {
+    if(rules.sixWinRule == Rules::SIXWINRULE_ALWAYS)
+      return true;              // freestyle: blocked five still wins
     return t1 != 0 || t2 != 0;  // oxxxxxo
   }
```

Ý nghĩa:

| `sixWinRule` | Dãy 5 bị chặn hai đầu | Dãy 6+ |
|---|---|---|
| `SIXWINRULE_ALWAYS` + patch | **thắng** (luật đồ án) | thắng |
| `SIXWINRULE_CARO` | không thắng | thắng nếu còn đầu mở |
| `SIXWINRULE_NEVER` | không thắng | không thắng |

Config dùng `SIXWINRULE_ALWAYS` và `wallBlockRule = false`. Sau patch, engine hiểu đúng cả ba
trường hợp: 5 bị chặn, 5 thường, 6+.

> Patch chỉ sửa nhánh `SIXWINRULE_ALWAYS`. Các chế độ Caro/Never giữ nguyên hành vi cũ, nên không
> làm hỏng mục đích gốc của nhánh `Caro2024`.

---

## 3. Model: đúng loại chưa?

### 3.1 Model đang dùng

| Thuộc tính | Giá trị |
|---|---|
| Tên file | `b10_freestyle15x.bin.gz` |
| Release | `hzyhhzy/KataGomo` → `gomocup2025` |
| Kiến trúc | `b10c128nbt`, khoảng 6 MB (bản distill) |
| Tên model khi nạp | `gom_distill-b10c128nbt_fs15xdistill-…` |
| Luật huấn luyện | freestyle (5 quân, 6+ cũng thắng) |
| Kích thước huấn luyện | **15×15** |
| Chạy trên bàn 11×11 | được, nhờ cơ chế chuyển kích thước của KataGo |

Điểm quan trọng: **đây không phải model chuyên 11×11**. Nó là model freestyle 15×15, chạy trên
bàn 11×11 bằng cách thay đổi kích thước đầu vào. KataGo hỗ trợ việc này, nhưng sức chơi không
được bảo đảm bằng một model huấn luyện đúng 11×11.

### 3.2 Có model 11×11 riêng không?

Đã kiểm tra:

- release hiện tại `gomocup2025`: chỉ có `b10_caro15x`, `b10_freestyle15x`, `b10_freestyle20x`,
  `b10_renju15x`, `b10_standard15x` — không có bản 11×11;
- release cũ `Gomoku_20250206`: có model renju lớn và gói engine, không có model 11×11 riêng;
- nhánh `Gom2024`, `Caro2024`, `GomCaro`: không kèm model 11×11.

Vì vậy, nếu mục tiêu là “model đúng 11×11 từ đầu”, câu trả lời trung thực là: **chưa có trong
các gói chính thức đã kiểm tra**. Phương án hiện tại là model freestyle 15×15 chuyển kích thước.

### 3.3 Hệ quả cần ghi rõ trong báo cáo

- Kết quả với KataGomo là **đối chiếu tham khảo**, không phải “trần” của bài toán 11×11.
- Không nên so sánh tuyệt đối Elo của KataGomo với L1–L8.
- Nếu cần một model đúng 11×11, có thể thử các model AlphaZero 11×11 bên ngoài (ví dụ
  `minghangz/gomoku_train`) hoặc huấn luyện riêng; cả hai đều là việc khác, không nằm trong
  tích hợp này.

---

## 4. Kiến trúc tác tử `KataGomoAgent`

Tệp: `caro/agents/katagomo.py`. Kind đăng ký: `katagomo`. Config:
`configs/agents/l9_katagomo.yaml`.

### 4.1 Vòng đời tiến trình

Agent **không** khởi động engine trong `__init__`; nó khởi động ở nước đầu tiên. Lý do: UI tạo
tác tử cho cả hai bên ngay khi mở ván, kể cả khi người chơi không dùng tới.

```
__init__  → lưu đường dẫn, ngân sách, timeout
reset()   → xoá RNG/stats; nếu tiến trình đã chết thì đánh dấu để khởi động lại
select_action()
  ├─ _ensure()            → chưa có/chết thì _start()
  ├─ _select()
  │   ├─ clear_board
  │   ├─ play từng nước trong game.history (đổi màu theo thứ tự)
  │   ├─ đặt ngân sách (maxVisits hoặc time_settings)
  │   ├─ genmove b|w
  │   ├─ đổi toạ độ GTP → chỉ số 0..120
  │   └─ đọc CHAT để lấy visits/value/PV
  └─ lỗi GTP/timeout → close(), khởi động lại, thử lại đúng một lần
close()   → quit, terminate, kill nếu cần
```

`__del__` gọi `close()`. UI cũng đóng tác tử khi bỏ ván cũ khỏi bộ nhớ (xem
`caro/ui/server.py`, phần `MAX_GAMES`).

### 4.2 Đồng bộ toàn bộ ván mỗi lượt

Trước mỗi nước, agent gửi:

```
clear_board
play b <nước 1>
play w <nước 2>
...
genmove <b|w>
```

Ưu điểm: đúng cả khi UI **lùi nước**, khi arena chơi lại cùng một đối tượng qua nhiều ván, hoặc
khi tiến trình vừa được khởi động lại. Nhược điểm: thời gian tuyến tính theo số nước; không giữ
được cây tìm kiếm giữa các lượt. Với bàn 11×11 (tối đa 121 nước), chi phí này nhỏ so với 800 lượt
thăm mỗi nước.

### 4.3 Đổi toạ độ

| Đồ án | GTP |
|---|---|
| `i = r·11 + c`, `r = 0` là hàng **trên** | hàng `1` là hàng **dưới** |
| cột `c = 0..10` | `A B C D E F G H J K L` (bỏ `I`) |

Công thức:

```
action i  →  (r, c) = divmod(i, 11)
GTP       =  COLS[c] + str(11 - r)
```

Ví dụ: `i = 0` → `A11`; `i = 60` → `F6`; `i = 120` → `L1`.

Hai hàm `_gtp()` và `_action()` là nghịch đảo của nhau; có test vòng lặp cho cả 121 ô.

### 4.4 Ngân sách

| `limits.mode` | Lệnh gửi engine |
|---|---|
| `fixed` | `time_settings 0 3600 1` + `kata-set-param maxVisits <max_sims>` |
| `time` | `time_settings 0 <giây> 1` + `maxVisits` rất lớn |

`max_sims` lấy từ `limits`; nếu không có thì dùng `max_visits` trong config (mặc định 800).
Số luồng engine đặt bằng `-override-config numSearchThreads=<threads>` (mặc định 2).

### 4.5 Thống kê

KataGomo in thông tin tìm kiếm ra **stderr**, dạng:

```
CHAT:Visits 801 Winrate 79.90% Drawrate 32.06% PV F6 G6 G7 E5 F8
```

Một thread nền đọc stderr, giữ dòng `CHAT:` mới nhất. Agent parse thành:

| Khoá | Ý nghĩa |
|---|---|
| `visits`, `sims` | số lượt thăm |
| `value` | `Winrate / 100` (chỉ để hiển thị, không phải xác suất đã hiệu chỉnh) |
| `pv` | biến chính, đổi về chỉ số 0..120 |

`Winrate` của KataGomo không phải lúc nào cũng theo góc nhìn bên đi; không dùng nó làm xác suất
thắng đã hiệu chỉnh trong phân tích.

### 4.6 Lỗi và phục hồi

- Thiếu binary/model/config → `ConfigError` với hướng dẫn chạy `scripts/setup_katagomo.sh`.
- GTP trả `? …` → `KataGomoError`; agent đóng tiến trình, khởi động lại, thử lại một lần.
- Quá `move_timeout` (mặc định 120 giây) → `TimeoutError`, cùng cơ chế thử lại.
- `genmove` trả `pass`/`resign` hoặc nước không hợp lệ → chọn ô trống đầu tiên để ván tiếp tục.
  Trong luật đồ án không có `pass`, nên đây chỉ là phương án phòng hờ.
- Log của engine bị tắt (`logAllGTPCommunication=false`, `logSearchInfo=false`,
  `logFile=/dev/null`) và `cwd` đặt tại thư mục engine, để không tạo `kata.log` trong repo.

### 4.7 Tệp liên quan

| Tệp | Vai trò |
|---|---|
| `caro/agents/katagomo.py` | lớp `KataGomoAgent` |
| `caro/agents/__init__.py` | đăng ký `kind: katagomo` |
| `caro/ui/server.py` | thêm `katagomo` vào `BUDGET_KEYS`; đóng engine khi bỏ ván; trả 400 nếu chưa cài |
| `configs/agents/l9_katagomo.yaml` | id `KataGomo-11`, level hiển thị 8, ngân sách mặc định |
| `configs/katagomo/default_gtp.cfg` | luật + cấu hình engine |
| `configs/katagomo/freestyle-five.patch` | patch luật 5 quân bị chặn |
| `scripts/setup_katagomo.sh` | tải model, clone, vá, build, kiểm tra |
| `tests/test_katagomo.py` | test đổi toạ độ và parse CHAT (không cần engine) |

---

## 5. Cài đặt

### 5.1 Tự động

```bash
scripts/setup_katagomo.sh
```

Script làm tuần tự:

1. tải `b10_freestyle15x.bin.gz` từ release `gomocup2025`;
2. clone nhánh `Caro2024`;
3. áp `configs/katagomo/freestyle-five.patch`;
4. tải và cài header Eigen 3.4.0;
5. build backend Eigen (CPU), mặc định một nửa số nhân, tối đa 4 luồng;
6. chạy thử `boardsize 11` + `genmove b`.

Kết quả mặc định ở `~/.local/share/caro/katagomo`:

```
katago                  engine đã vá (symlink tới src/build/katago)
b10_freestyle15x.bin.gz model
default_gtp.cfg         luật freestyle
.engine                 dấu phiên bản "caro2024-freestyle5"
src/                    source + build
eigen-3.4.0/            header
```

Đổi vị trí: `KATAGOMO_HOME=/đường/dẫn scripts/setup_katagomo.sh`.
Build lại từ đầu: `scripts/setup_katagomo.sh --rebuild`.

Yêu cầu: `git`, `cmake` ≥ 3.16, trình biên dịch C++14, `curl`, `tar`, `patch`, Python 3
(không bắt buộc nếu không dùng phần kiểm tra). Không cần GPU.

### 5.2 Thủ công

```bash
git clone --depth 1 --branch Caro2024 https://github.com/hzyhhzy/KataGomo
cd KataGomo
patch -p1 < /đường/dẫn/CaroGame/configs/katagomo/freestyle-five.patch
grep -n "freestyle: blocked five still wins" cpp/game/gamelogic.cpp

curl -L https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar xz
cmake -S eigen-3.4.0 -B eigen-build -DCMAKE_INSTALL_PREFIX=/tmp/eigen -DEIGEN_BUILD_DOC=OFF
cmake --install eigen-build

cmake -S cpp -B build -DUSE_BACKEND=EIGEN \
  -DCMAKE_PREFIX_PATH=/tmp/eigen -DNO_LIBZIP=1 -DNO_GIT_REVISION=1 \
  -DUSE_AVX2=1 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

Sau đó đặt `KATAGOMO_BIN`, `KATAGOMO_MODEL`, `KATAGOMO_CONFIG` hoặc sửa đường dẫn trong
`configs/agents/l9_katagomo.yaml`.

### 5.3 Kiểm tra nhanh

```bash
printf 'boardsize 11\nclear_board\ngenmove b\nquit\n' | \
  ~/.local/share/caro/katagomo/katago gtp \
  -model ~/.local/share/caro/katagomo/b10_freestyle15x.bin.gz \
  -config ~/.local/share/caro/katagomo/default_gtp.cfg
```

Kỳ vọng: dòng `= <nước>` với toạ độ trong `A..L` và hàng `1..11`. Nếu báo thiếu model/binary,
chạy lại script hoặc kiểm tra ba biến môi trường.

### 5.4 Gỡ cài đặt

```bash
rm -rf ~/.local/share/caro/katagomo
```

Không ảnh hưởng code hay test của đồ án; agent chỉ báo `ConfigError` khi được chọn.

---

## 6. Cách chạy

### 6.1 UI

```bash
uv run uvicorn caro.ui.server:app --port 8000
```

Mở `http://127.0.0.1:8000`, chọn `KataGomo-11` cho một bên. Tác tử hiện ở nhóm level 8
(cùng nhóm hiển thị với L8) nhưng **không** thuộc thang điểm. Panel thống kê hiện `visits`,
`value`, `pv`.

Nếu chưa cài engine, `POST /games` vẫn tạo được ván nhưng nước đầu tiên trả HTTP 400 với thông
báo thiếu file — không phải lỗi 500.

### 6.2 Một ván bằng code

```bash
uv run python - <<'PY'
from caro.agents import make_agent
from caro.arena.match import play_match

kata = make_agent("agents/l9_katagomo")
l8 = make_agent("agents/l8_alphazero")
r = play_match(kata, l8, [], seed=2026, game_id=1, max_plies=121)
print(r["result"], r["termination"], len(r["moves"]), "nước")
kata.close()
PY
```

`result` theo góc nhìn X: `+1` X thắng, `-1` O thắng, `0` hoà.

### 6.3 Giải đấu

```bash
uv run python -m caro.arena.tournament experiments/main_mode_b \
  agents=[agents/l8_alphazero,agents/l9_katagomo] \
  experiment.name=l8_vs_katagomo arena.games_per_pair=10 \
  arena.swap_colors=true arena.workers=2 arena.stats.elo=false
```

Lưu ý:

- Mỗi worker tạo engine riêng; `workers=2` nghĩa là tối đa 2 tiến trình KataGomo song song.
- Không nên đặt `workers` lớn hơn số nhân vật lý; engine Eigen dùng CPU.
- Khi dùng `main_mode_a` (1 giây/nước), mỗi engine bị giới hạn bởi `time_settings`, không phải
  `maxVisits`.
- Kết quả nên lưu vào `runs/` như mọi giải khác; đây là **đối chiếu**, không phải bảng chính thức.

### 6.4 Tinh chỉnh

`configs/agents/l9_katagomo.yaml`:

| Khoá | Mặc định | Ghi chú |
|---|---|---|
| `max_visits` | 800 | tăng lên 2000–5000 nếu muốn mạnh hơn; thời gian tăng gần tuyến tính |
| `threads` | 2 | tăng 4 nếu máy rảnh; có thể gây lag |
| `move_timeout` | 120 | giây; quá hạn thì khởi động lại engine |
| `limits` | fixed/800 | arena/UI có thể truyền ngân sách khác |

Trên máy tham chiếu (CPU, Eigen, 2 luồng), 800 visits mất khoảng 2–4 giây/nước; 3000 visits
khoảng 10–15 giây/nước. Con số thay đổi theo CPU và theo số engine chạy song song.

---

## 7. Kiểm thử

### 7.1 Test tự động

```bash
uv run pytest -q tests/test_katagomo.py
uv run ruff check caro/agents/katagomo.py caro/ui/server.py
uv run pytest -q                     # toàn bộ
```

`tests/test_katagomo.py` kiểm tra:

- `_gtp()` và `_action()` là nghịch đảo cho cả 121 ô;
- các toạ độ biên `A11`, `L11`, `L1`, giữa bàn `F6`;
- `pass`, `resign`, chuỗi rỗng, toạ độ ngoài bàn → `None`;
- parser `CHAT` lấy đúng visits, winrate và PV.

Không cần engine/model cho các test này, nên CI không phụ thuộc vào file ngoài repo.

### 7.2 Kiểm tra chiến thuật thủ công

Các thế đã dùng để kiểm tra engine sau patch:

| Thế | Kỳ vọng | Kết quả |
|---|---|---|
| X có 4 quân, X đi | thắng ngay | chọn ô hoàn thành 5 |
| O có 4 quân, X đi | chặn một trong hai đầu | chặn đúng |
| O có 3 quân mở, X đi | chặn một đầu | chặn đúng sau patch |
| 5 quân ngang ở biên | thắng | nhận đúng |
| 5 quân dọc `F1..F5` | thắng | nhận đúng (Caro2024); Gom2024 không nhận |
| `OXXXXXO` | thắng theo luật đồ án | patch trả `MP_FIVE` |

Lệnh kiểm tra một thế cụ thể: dùng `play` để dựng thế rồi `genmove`; xem thêm `CHAT` để đối
chiếu giá trị. Vì GTP không tự kết thúc ván, đừng dùng việc `genmove` còn trả nước làm bằng
chứng engine “không thấy thắng”; hãy kiểm tra nhánh `MP_FIVE`/`pass` hoặc chạy qua `CaroGame`.

### 7.3 Quan sát từ giải thử 10 ván

Một giải thử `mode A` (1 giây/nước, 10 ván đổi màu, 5 khai cuộc) đã chạy trước khi vá:

| | KataGomo | L8 |
|---|---|---|
| Điểm | 7,0 | 3,0 |
| Thắng / hoà / thua | 5 / 4 / 1 | 1 / 4 / 5 |
| Ván kết thúc bằng 5 quân | 5 | 1 |
| Ván chạm 121 nước | 4 | 4 |

Kết quả này cho thấy engine **có sức chơi thực sự**, không phải một tác tử hỏng. Tuy nhiên:

- giải chạy bằng binary `Gom2024` (trước khi phát hiện lỗi không nhận 5 quân dọc), nên **không**
  dùng làm số liệu chính thức;
- 4 ván hoà chạm trần 121 nước cho thấy cả hai bên phòng thủ tốt, không có nghĩa engine yếu;
- 10 ván là quá ít để kết luận Elo.

Sau khi chuyển sang `Caro2024` + patch, cần chạy lại cùng giao thức trước khi trích số.

### 7.4 Điều chưa có

- Chưa có trận đấu thống kê đủ lớn (ví dụ 100 ván đổi màu) giữa L8 và KataGomo sau patch.
- Chưa có nhiều seed/khai cuộc cho đối thủ ngoài.
- Chưa đo Elo tin cậy; 2–10 ván chỉ là kiểm tra khói.

Nếu thêm số liệu vào báo cáo, nên ghi rõ số ván, ngân sách, màu, khai cuộc và khoảng tin cậy.

---

## 8. Hạn chế và rủi ro

1. **Model 15×15 chuyển sang 11×11.** Đây là hạn chế lớn nhất. Nước đi có thể mang “thói quen”
   bàn lớn; giá trị ở biên có thể lệch. Không nên coi KataGomo là chuẩn tuyệt đối cho 11×11.
2. **Model distill nhỏ.** `b10c128nbt` khoảng 6 MB, không phải model lớn nhất của KataGomo. Muốn
   mạnh hơn phải tìm model lớn hơn hoặc huấn luyện riêng.
3. **Đồng bộ lại từ đầu mỗi nước.** Đơn giản và đúng, nhưng chậm hơn giữ cây; với 800 visits
   thì chi phí replay không đáng kể, với 20k visits thì đáng kể.
4. **Không có pass/resign trong luật đồ án.** Nếu engine trả `pass`, agent thay bằng ô trống đầu
   tiên. Đây là fallback, không phải nước đi tốt.
5. **Một tiến trình cho mỗi tác tử.** UI đóng khi bỏ ván, nhưng nếu tự viết script thì phải gọi
   `close()`; nếu không, tiến trình engine có thể còn sống.
6. **Patch phải áp lại khi đổi upstream.** `scripts/setup_katagomo.sh` kiểm tra chuỗi
   `freestyle: blocked five still wins` và báo lỗi nếu patch không áp được. Đừng build tay mà bỏ
   qua bước này.
7. **`Winrate` không phải xác suất đã hiệu chỉnh.** Chỉ dùng để hiển thị/tham khảo.
8. **Không thuộc thang L1–L8.** Không đưa vào bảng chính thức, không dùng để chọn checkpoint.

---

## 9. Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `ConfigError: chưa cài KataGomo (thiếu …)` | chưa chạy setup hoặc sai `KATAGOMO_HOME` | chạy `scripts/setup_katagomo.sh`; kiểm tra 3 biến môi trường |
| `KataGomo đã thoát` | binary lỗi, model không khớp, hết RAM | chạy lệnh kiểm tra ở 5.3; xem stderr |
| `TimeoutError` | engine quá chậm hoặc bị treo | tăng `move_timeout`, giảm `max_visits`/`threads` |
| Nước đi “lạ” ở biên | model 15×15 chuyển kích thước | kiểm tra lại bằng thế chiến thuật; cân nhắc model khác |
| Engine không nhận 5 quân dọc | đang dùng `Gom2024` gốc | dùng `Caro2024` + patch (script đã làm) |
| `OXXXXXO` bị coi là chưa thắng | chưa áp patch | kiểm tra `grep "freestyle: blocked five still wins"` trong source |
| UI báo 400 khi đi nước đầu | thiếu engine/model | chạy setup; không phải lỗi UI |
| `Text file busy` khi copy binary | engine đang chạy | dừng tiến trình engine hoặc copy sang tên khác |

---

## 10. Checklist trước khi push GitHub

### 10.1 Tệp nên có trong commit

```
caro/agents/katagomo.py
caro/agents/__init__.py
caro/ui/server.py
configs/agents/l9_katagomo.yaml
configs/katagomo/default_gtp.cfg
configs/katagomo/freestyle-five.patch
scripts/setup_katagomo.sh
docs/KATAGOMO.md
tests/test_katagomo.py
```

Không commit:

- binary `katago` (80–90 MB);
- model `.bin.gz` (6 MB);
- `~/.local/share/caro/katagomo`;
- log `kata.log`;
- file build trong `src/build`.

### 10.2 Kiểm tra

```bash
uv run ruff check caro/agents/katagomo.py caro/ui/server.py tests/test_katagomo.py
uv run pytest -q
bash -n scripts/setup_katagomo.sh
```

### 10.3 Nội dung PR gợi ý

**Tiêu đề:** `feat(agents): thêm KataGomo làm đối thủ ngoài 11×11`

**Mô tả:**

- Thêm `KataGomoAgent` qua GTP, đổi toạ độ 2 chiều, đồng bộ ván, ngân sách fixed/time.
- Đăng ký kind `katagomo`, config `KataGomo-11`; UI đóng engine khi bỏ ván và báo 400 nếu chưa cài.
- Dùng `Caro2024` + patch `freestyle-five.patch` để luật 5 quân bị chặn hai đầu khớp đồ án.
- Không dùng `Gom2024` vì không nhận dãy 5 dọc trên 11×11; không dùng `Caro2024` gốc vì coi
  `OXXXXXO` là chưa thắng.
- Model `b10_freestyle15x` (15×15, chuyển kích thước); chưa có model 11×11 chính thức.
- Test: 321 test pass; test toạ độ/parse; kiểm tra chiến thuật thủ công.
- Không đưa vào thang L1–L8; chỉ dùng đối chiếu.

### 10.4 Lệnh git (chỉ để tham khảo, không tự chạy)

```bash
git status
git add caro/agents/katagomo.py caro/agents/__init__.py caro/ui/server.py \
  configs/agents/l9_katagomo.yaml configs/katagomo scripts/setup_katagomo.sh \
  docs/KATAGOMO.md tests/test_katagomo.py
git commit -m "feat(agents): thêm KataGomo làm đối thủ ngoài 11×11"
```

Người dùng tự commit; tài liệu này không chạy git thay.

---

## 11. Tóm tắt một đoạn

KataGomo được tích hợp như một **đối thủ ngoài**, không phải một cấp mới. Engine dùng là
`Caro2024` đã vá một dòng để luật 5 quân khớp đồ án; model là `b10_freestyle15x` (15×15, chuyển
kích thước, vì chưa có model 11×11 chính thức). Agent nói GTP, đồng bộ lại ván mỗi lượt, đổi toạ
độ cẩn thận, lấy visits/PV từ stderr và tự phục hồi khi engine chết. Cài bằng
`scripts/setup_katagomo.sh`; chạy trên UI bằng cách chọn `KataGomo-11`; test bằng
`tests/test_katagomo.py` và các thế chiến thuật. Kết quả với KataGomo chỉ nên dùng để đối chiếu,
không dùng để xếp hạng L1–L8 hay kết luận tuyệt đối về sức mạnh trên bàn 11×11.
