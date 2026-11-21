# Ghi chú: quy ước dấu của value head lệch giữa học có giám sát (cấp 5) và RL (cấp 6)

Ngày phát hiện: 2026-11-17 · Nhánh: `feat/l6-reinforce` · Trạng thái: **đã sửa** (hướng 1: `train_sl.mover_relative_values`; SL v3 → `model:l5`, RL từ v3 → `model:l6`). Kiểm lại với SL v3: MSE 0,356, tương quan +0,77 ở cả X đi lẫn O đi.

## Triệu chứng

Chạy thử `train_pg` (khởi tạo từ `runs/train/sl_policy_v2/latest.pt`), `value_loss` trong `metrics.jsonl` luôn ở mức
**1,2–1,4**. Con số này tệ hơn cả việc luôn đoán 0 (MSE ≈ 0,9), tức value head làm baseline REINFORCE
`z − v(s)` còn tệ hơn không có baseline.

## Nguyên nhân

Hai nơi dùng hai quy ước khác nhau cho nhãn kết quả ván:

| Nơi | Quy ước | Chỗ trong code |
|---|---|---|
| Sinh dữ liệu SL, nhãn `value` | **tuyệt đối**: `+1` nếu X thắng, `−1` nếu O thắng | `caro/training/gen_expert_data.py` (`value_target = 1.0 if winner == X …`) |
| Đầu vào mạng | **góc nhìn người đi**: kênh 0 = quân "me", kênh 1 = quân "opp", kênh 2 = 1 nếu X đi | `caro/nn/features.py` |
| RL, `z` | **góc nhìn người vừa đi**: `+1` nếu người đi nước đó thắng | `caro/training/train_pg.py` (docstring + dòng tính `zs`) |

Value head học từ SL dự đoán "X có thắng không". RL cần "người đang đi có thắng không". Hai thứ trùng nhau ở thế cờ
X đi và **ngược dấu** ở thế cờ O đi, nên baseline sai dấu ở khoảng một nửa số thế cờ.

## Chứng minh

Dùng checkpoint SL v2, tập test (cùng cách chia theo ván như `train_sl.py`: `default_rng(42)`, 10% ván cuối) của
`runs/data/expert_v2`: 29.161 mẫu (X đi 14.679, O đi 14.482). Đánh giá value head với hai cách gán nhãn:

| Nhóm | MSE vs nhãn tuyệt đối | MSE vs nhãn người-đi | Tương quan (tuyệt đối) | Tương quan (người-đi) |
|---|---|---|---|---|
| Tất cả | 0,418 | **1,377** | +0,73 | +0,01 |
| X đi | 0,410 | 0,410 | +0,73 | +0,73 |
| O đi | 0,426 | **2,357** | +0,73 | **−0,73** |

- Hằng số 0 cho MSE 0,906.
- Với nhãn người-đi, MSE toàn bộ là 1,377, trùng khoảng 1,2–1,4 của `value_loss` quan sát trong RL.
- Ở thế O đi, value head có tương quan **−0,73** với nhãn RL: dự đoán đúng độ lớn, sai hẳn dấu.
- Với nhãn tuyệt đối thì mạng học tốt (tương quan +0,73 ở cả hai nhóm). Lỗi nằm ở chỗ ghép quy ước, không phải ở việc huấn luyện SL.

Tái hiện (từ thư mục gốc dự án):

```python
import numpy as np, torch
from caro.nn.model import CaroResNet
dev = torch.device("cuda")
net = CaroResNet({"num_channels": 64, "num_blocks": 6}).to(dev)
net.load_state_dict(torch.load("runs/train/sl_policy_v2/latest.pt", map_location=dev)); net.eval()
d = np.load("runs/data/expert_v2/expert_dataset.npz")
s, v, g = d["states"], d["values"], d["game_ids"]
u = np.random.default_rng(42).permutation(np.unique(g))
idx = np.where(np.isin(g, u[int(len(u) * .9):]))[0]
X = torch.tensor(s[idx].astype(np.float32))
x_turn = s[idx][:, 2, 0, 0] == 1                    # kênh 2: X đi
y_abs = v[idx]
y_rel = y_abs * np.where(x_turn, 1.0, -1.0)         # nhãn theo góc nhìn người đi
with torch.no_grad():
    p = torch.cat([net(X[i:i + 2048].to(dev))[1].squeeze(-1).cpu() for i in range(0, len(X), 2048)]).numpy()
for name, m in (("tất cả", slice(None)), ("X đi", x_turn), ("O đi", ~x_turn)):
    print(name, ((p[m] - y_abs[m]) ** 2).mean(), ((p[m] - y_rel[m]) ** 2).mean())
```

## Hệ quả

- Policy của cấp 5 **không bị ảnh hưởng**: agent chỉ dùng policy khi đánh. Số liệu và ma trận đấu ở các notebook arena vẫn đúng.
- Cấp 6 (REINFORCE) bị ảnh hưởng: baseline nhiễu làm gradient policy kém hiệu quả, nhất là giai đoạn đầu khi init từ SL.
- Cấp 7/8 dùng chung checkpoint `l5` và đọc value để đánh giá thế cờ, nên cũng cần đúng quy ước.

## Hướng sửa (chưa thực hiện)

1. **Sửa tại gốc (khuyên dùng):** ở `train_sl.py`, đổi nhãn value sang góc nhìn người đi:
   `values = values * np.where(states[:, 2, 0, 0] == 1, 1.0, -1.0)`. Train lại SL (v3), rồi dùng làm init cho RL.
   Sau đó mọi nơi dùng value (RL, PUCT, AlphaZero) cùng một quy ước.
2. **Chữa nhanh ở RL:** trong `train_pg.py`, khi `init` đến từ SL, khởi tạo lại lớp value head (hoặc cho nó học trước vài vòng
   với `z` của RL) để không thừa hưởng dấu sai.

Sau khi sửa, kiểm tra lại bằng đoạn mã trên (nhãn người-đi phải cho MSE thấp, tương quan dương ở cả X đi lẫn O đi) và xem
`value_loss` của `train_pg` giảm dưới ~0,9.

## Các số liệu liên quan khi phát hiện (để so sánh sau này)

Chạy thử `train_pg` (`run.tag=smoke2`, 896 ván, GPU, 230 giây), đánh giá argmax, L3 dùng `backend: auto` độ sâu mặc định 6:

| Mốc | vs L2 | vs L3 |
|---|---|---|
| SL v2 gốc (100 ván) | 0,66 | 0,205 |
| sau 512 ván RL (50 ván) | 0,44 | 0,29 |
| sau 896 ván RL (50 ván) | 0,66 | 0,15 |

Các mốc RL chỉ 50 ván, khoảng tin cậy cỡ ±0,13, chưa đủ để kết luận RL cải thiện hay xấu đi.
