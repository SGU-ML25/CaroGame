# Minh chứng các lượt phân tích chuyên sâu

Sinh tự động bởi `report/scripts/chuyen_sau/chay.py`; không sửa tay.

Log giải đấu từng ván nằm trong `runs/eval/tournament/<tên>/<thời điểm>/games.jsonl`.

| Bước | Thời điểm | Thời lượng (s) | Mã thoát | Nội dung | Thư mục |
|---|---|---|---|---|---|
| bo-chien-thuat | 2026-11-26T22:35:48 | 50.7 | 0 | Dựng bộ 500 thế chiến thuật | [bo-chien-thuat/20261126-223548](bo-chien-thuat/20261126-223548) |
| cham-chien-thuat | 2026-11-26T22:39:55 | 3453.3 | 0 | Chấm tác tử trên bộ thế (cấu hình gốc, quét ngân sách, biến thể tham số) | [cham-chien-thuat/20261126-223955](cham-chien-thuat/20261126-223955) |
| giai-cpuct | 2026-11-26T23:42:13 | 1039.2 | 0 | L8 với c_puct 0,5/1/3/5 gặp L7, 400 ván | [giai-cpuct/20261126-234213](giai-cpuct/20261126-234213) |
| giai-sims-l8 | 2026-11-26T23:01:58 | 2412.8 | -15 | L8 ở 1/50/200/800/3200 mô phỏng gặp L7 (200), 500 ván | [giai-sims-l8/20261126-230158](giai-sims-l8/20261126-230158) |
| kiem-thu-vcf | 2026-11-26T22:35:48 | 0.3 | 0 | Test bộ giải VCF | [kiem-thu-vcf/20261126-223548](kiem-thu-vcf/20261126-223548) |
| kiem-thu-vcf | 2026-11-27T00:06:09 | 0.3 | 0 | Test bộ giải VCF | [kiem-thu-vcf/20261127-000609](kiem-thu-vcf/20261127-000609) |
| kiem-thu-vct | 2026-11-27T00:43:07 | 7.6 | 0 | Test bộ giải VCT | [kiem-thu-vct/20261127-004307](kiem-thu-vct/20261127-004307) |
| kiem-tra-goc | 2026-11-26T22:50:36 | 301.9 | 0 | L3: nước tốt nhất của độ sâu trước có được duyệt đầu ở gốc không | [kiem-tra-goc/20261126-225036](kiem-tra-goc/20261126-225036) |
| kiem-tra-rollout | 2026-11-26T23:39:42 | 49.6 | 0 | Vì sao MCTS rollout light bỏ nước chặn (100 thế C2) | [kiem-tra-rollout/20261126-233942](kiem-tra-rollout/20261126-233942) |
| kiem-tra-thay-the | 2026-11-27T01:01:45 | 180.3 | 0 | Kiểm các nước thay thế cho sai lầm: tìm sâu hơn và hỏi KataGomo | [kiem-tra-thay-the/20261127-010145](kiem-tra-thay-the/20261127-010145) |
| phan-tich-du-lieu | 2026-11-26T22:40:29 | 76.6 | 0 | Nhãn policy hoà điểm trong dữ liệu expert và ảnh hưởng lên tập test | [phan-tich-du-lieu/20261126-224029](phan-tich-du-lieu/20261126-224029) |
| phan-tich-du-lieu | 2026-11-26T22:56:22 | 101.1 | 0 | Nhãn policy hoà điểm trong dữ liệu expert và ảnh hưởng lên tập test | [phan-tich-du-lieu/20261126-225622](phan-tich-du-lieu/20261126-225622) |
| phan-tich-mang | 2026-11-26T22:37:36 | 139.0 | 0 | Đo mạng trên tập test, đối xứng D4 và bộ thế | [phan-tich-mang/20261126-223736](phan-tich-mang/20261126-223736) |
| phan-tich-thong-ke | 2026-11-26T22:36:43 | 6.1 | 0 | Thống kê bổ sung trên log RR8 giữ riêng | [phan-tich-thong-ke/20261126-223643](phan-tich-thong-ke/20261126-223643) |
| phan-tich-tim-kiem | 2026-11-26T23:59:32 | 213.3 | 0 | Đo tốc độ và hành vi tìm kiếm (chạy khi máy rảnh) | [phan-tich-tim-kiem/20261126-235932](phan-tich-tim-kiem/20261126-235932) |
| phan-tich-van | 2026-11-26T22:36:39 | 56.5 | 0 | Gắn nhãn từng nước của 2.800 ván RR8 giữ riêng | [phan-tich-van/20261126-223639](phan-tich-van/20261126-223639) |
| phan-tich-van-sau | 2026-11-27T00:51:11 | 534.4 | 0 | Phân tích sâu 4 ván: nước quyết định và sai lầm (VCT + KataGomo) | [phan-tich-van-sau/20261127-005111](phan-tich-van-sau/20261127-005111) |
| phu-luc-van | 2026-11-26T23:00:53 | 0.8 | 0 | Sinh phụ lục 28 ván đại diện | [phu-luc-van/20261126-230053](phu-luc-van/20261126-230053) |
| phu-luc-van | 2026-11-27T00:06:09 | 0.2 | 0 | Sinh phụ lục 28 ván đại diện | [phu-luc-van/20261127-000609](phu-luc-van/20261127-000609) |
| phu-luc-van | 2026-11-27T01:02:03 | 0.2 | 0 | Sinh phụ lục 28 ván đại diện | [phu-luc-van/20261127-010203](phu-luc-van/20261127-010203) |
