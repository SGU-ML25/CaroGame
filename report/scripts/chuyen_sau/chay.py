"""Chạy một bước phân tích chuyên sâu và lưu minh chứng vào runs/chuyen_sau/.

    .venv/bin/python report/scripts/chuyen_sau/chay.py --list
    .venv/bin/python report/scripts/chuyen_sau/chay.py kiem-thu-vcf bo-chien-thuat

Mỗi lượt tạo runs/chuyen_sau/<bước>/<YYYYmmdd-HHMMSS>/ gồm:
    meta.json   lệnh, thời gian, mã thoát, git, Python, thư viện, phần cứng, SHA-256 đầu vào / đầu ra
    run.log     toàn bộ đầu ra của lệnh, mỗi dòng kèm giờ
    ma-nguon/   bản sao mã đã chạy (cây làm việc có thể chưa commit)
    data/       bản sao dữ liệu do bước này ghi ra
runs/chuyen_sau/README.md liệt kê mọi lượt và được sinh lại sau mỗi bước.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import chung

from caro.runlog import collect_meta

EVIDENCE = chung.ROOT / "runs" / "chuyen_sau"
HERE = "report/scripts/chuyen_sau"
PY = [".venv/bin/python"]
PLOT = ["uv", "run", "--no-sync", "--with", "matplotlib", "python"]
RR8 = chung.RUNS["rr8"] + "/games.jsonl"
SUITE = "report/chuyen-sau/data/bo-chien-thuat.json"
DATASET = "runs/data/expert_v2/expert_dataset.npz"
NOAUG = "runs/train/a6_noaug/best.pt"


def data(*names: str) -> list[str]:
    return [f"report/chuyen-sau/data/{n}.json" for n in names]


# tên bước → mô tả, lệnh, mã nguồn cần lưu, đầu vào, đầu ra
STEPS: dict[str, dict] = {
    "kiem-thu-vcf": {
        "title": "Test bộ giải VCF",
        "cmd": [*PY, "-m", "pytest", "tests/test_vcf.py", "-v"],
        "code": [f"{HERE}/vcf.py", "tests/test_vcf.py"],
    },
    "bo-chien-thuat": {
        "title": "Dựng bộ 500 thế chiến thuật",
        "cmd": [*PY, f"{HERE}/bo_chien_thuat.py", "build"],
        "code": [f"{HERE}/bo_chien_thuat.py", f"{HERE}/vcf.py", f"{HERE}/chung.py"],
        "inputs": [RR8], "outputs": [SUITE],
    },
    "cham-chien-thuat": {
        "title": "Chấm tác tử trên bộ thế (cấu hình gốc, quét ngân sách, biến thể tham số)",
        "cmd": [*PY, f"{HERE}/bo_chien_thuat.py", "run", "base", "scale", "variants", "--workers", "12"],
        "code": [f"{HERE}/bo_chien_thuat.py", f"{HERE}/vcf.py", f"{HERE}/chung.py"],
        "inputs": [SUITE],
        "outputs": data("chien-thuat-base", "chien-thuat-scale", "chien-thuat-variants"),
    },
    "phan-tich-van": {
        "title": "Gắn nhãn từng nước của 2.800 ván RR8 giữ riêng",
        "cmd": [*PY, f"{HERE}/phan_tich_van.py", "--workers", "4"],
        "code": [f"{HERE}/phan_tich_van.py", f"{HERE}/vcf.py", f"{HERE}/chung.py"],
        "inputs": [RR8], "outputs": data("phan-tich-van"),
    },
    "phan-tich-mang": {
        "title": "Đo mạng trên tập test, đối xứng D4 và bộ thế",
        "cmd": [*PY, f"{HERE}/phan_tich_mang.py"],
        "code": [f"{HERE}/phan_tich_mang.py", f"{HERE}/vcf.py", f"{HERE}/chung.py"],
        "inputs": [DATASET, NOAUG, SUITE, RR8], "outputs": data("phan-tich-mang"),
    },
    "phan-tich-du-lieu": {
        "title": "Nhãn policy hoà điểm trong dữ liệu expert và ảnh hưởng lên tập test",
        "cmd": [*PY, f"{HERE}/phan_tich_du_lieu.py"],
        "code": [f"{HERE}/phan_tich_du_lieu.py", f"{HERE}/phan_tich_mang.py", "caro/training/gen_expert_data.py",
                 "caro/agents/alphabeta.py"],
        "inputs": [DATASET, "runs/data/expert_v1/expert_dataset.npz", NOAUG],
        "outputs": data("phan-tich-du-lieu"),
    },
    "phan-tich-thong-ke": {
        "title": "Thống kê bổ sung trên log RR8 giữ riêng",
        "cmd": [*PY, f"{HERE}/phan_tich_thong_ke.py"],
        "code": [f"{HERE}/phan_tich_thong_ke.py", f"{HERE}/chung.py", "caro/arena/stats.py"],
        "inputs": [RR8, "runs/openings/openings_heldout_v2.json"], "outputs": data("phan-tich-thong-ke"),
    },
    "phu-luc-van": {
        "title": "Sinh phụ lục 28 ván đại diện",
        "cmd": [*PY, f"{HERE}/phu_luc_van.py"],
        "code": [f"{HERE}/phu_luc_van.py", f"{HERE}/vcf.py", f"{HERE}/chung.py"],
        "inputs": [RR8, *data("phan-tich-van", "phan-tich-thong-ke")],
        "outputs": ["report/chuyen-sau/bang/pl-van.tex"],
    },
    "kiem-thu-vct": {
        "title": "Test bộ giải VCT",
        "cmd": [*PY, "-m", "pytest", "tests/test_vct.py", "-v"],
        "code": [f"{HERE}/vct.py", f"{HERE}/vcf.py", "tests/test_vct.py"],
    },
    "phan-tich-van-sau": {
        "title": "Phân tích sâu 4 ván: nước quyết định và sai lầm (VCT + KataGomo)",
        "cmd": [*PY, f"{HERE}/phan_tich_van_sau.py"],
        "code": [f"{HERE}/phan_tich_van_sau.py", f"{HERE}/vct.py", f"{HERE}/vcf.py", "caro/agents/katagomo.py"],
        "inputs": [RR8, "runs/train/alphazero/20261124-160615_gpu-native/games.jsonl",
                   "runs/eval/tournament/l8_vs_l3/20261125-134954_round37/games.jsonl"],
        "outputs": data("phan-tich-van-sau"),
    },
    "kiem-tra-thay-the": {
        "title": "Kiểm các nước thay thế cho sai lầm: tìm sâu hơn và hỏi KataGomo",
        "cmd": [*PY, f"{HERE}/kiem_tra_thay_the.py"],
        "code": [f"{HERE}/kiem_tra_thay_the.py", f"{HERE}/vct.py", f"{HERE}/vcf.py"],
        "inputs": data("phan-tich-van-sau"), "outputs": data("kiem-tra-thay-the"),
    },
    "kiem-tra-rollout": {
        "title": "Vì sao MCTS rollout light bỏ nước chặn (100 thế C2)",
        "cmd": [*PY, f"{HERE}/kiem_tra_rollout.py"],
        "code": [f"{HERE}/kiem_tra_rollout.py", "caro/agents/mcts.py"],
        "inputs": [SUITE], "outputs": data("kiem-tra-rollout"),
    },
    "kiem-tra-goc": {
        "title": "L3: nước tốt nhất của độ sâu trước có được duyệt đầu ở gốc không",
        "cmd": [*PY, f"{HERE}/kiem_tra_goc.py"],
        "code": [f"{HERE}/kiem_tra_goc.py", "caro/agents/alphabeta.py"],
        "inputs": ["runs/openings/midgame_positions_v1.json"], "outputs": data("kiem-tra-goc"),
    },
    "phan-tich-tim-kiem": {
        "title": "Đo tốc độ và hành vi tìm kiếm (chạy khi máy rảnh)",
        "cmd": [*PY, f"{HERE}/phan_tich_tim_kiem.py"],
        "code": [f"{HERE}/phan_tich_tim_kiem.py", f"{HERE}/vcf.py", f"{HERE}/chung.py"],
        "inputs": [SUITE, "runs/openings/midgame_positions_v1.json"], "outputs": data("phan-tich-tim-kiem"),
    },
    "giai-sims-l8": {
        "title": "L8 ở 1/50/200/800/3200 mô phỏng gặp L7 (200), 500 ván",
        "cmd": [*PY, "-m", "caro.arena.tournament", "experiments/l8_sims_heldout"],
        "code": ["configs/experiments/l8_sims_heldout.yaml"], "env": {"OMP_NUM_THREADS": "1"},
        "inputs": ["runs/openings/openings_heldout_v2.json"],
    },
    "giai-cpuct": {
        "title": "L8 với c_puct 0,5/1/3/5 gặp L7, 400 ván",
        "cmd": [*PY, "-m", "caro.arena.tournament", "experiments/cpuct_heldout"],
        "code": ["configs/experiments/cpuct_heldout.yaml"], "env": {"OMP_NUM_THREADS": "1"},
        "inputs": ["runs/openings/openings_heldout_v2.json"],
    },
}


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(paths: list[str]) -> list[dict]:
    out = []
    for rel in paths:
        path = chung.ROOT / rel
        out.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)} if path.is_file()
                   else {"path": rel, "missing": True})
    return out


def write_index() -> None:
    lines = ["# Minh chứng các lượt phân tích chuyên sâu", "",
             "Sinh tự động bởi `report/scripts/chuyen_sau/chay.py`; không sửa tay.", "",
             "Log giải đấu từng ván nằm trong `runs/eval/tournament/<tên>/<thời điểm>/games.jsonl`.", "",
             "| Bước | Thời điểm | Thời lượng (s) | Mã thoát | Nội dung | Thư mục |",
             "|---|---|---|---|---|---|"]
    for meta_path in sorted(EVIDENCE.glob("*/*/meta.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        folder = meta_path.parent.relative_to(EVIDENCE).as_posix()
        lines.append(f"| {m['step']} | {m['started']} | {m['duration_s']} | {m['exit_code']} | "
                     f"{m['title']} | [{folder}]({folder}) |")
    (EVIDENCE / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_step(name: str) -> int:
    step = STEPS[name]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = EVIDENCE / name / stamp
    folder.mkdir(parents=True)
    for rel in step["code"]:
        target = folder / "ma-nguon" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(chung.ROOT / rel, target)

    extra = {"CUDA_VISIBLE_DEVICES": "", **step.get("env", {})}
    env = {**os.environ, **extra, "PYTHONUNBUFFERED": "1"}
    meta = collect_meta({})
    meta.pop("seed", None)
    meta.pop("config_hash", None)
    meta |= {"step": name, "title": step["title"], "argv": step["cmd"], "env": extra,
             "inputs": describe(step.get("inputs", [])), "code": describe(step["code"])}

    start = time.perf_counter()
    with (folder / "run.log").open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(step['cmd'])}\n")
        process = subprocess.Popen(step["cmd"], cwd=chung.ROOT, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            stamped = f"{datetime.now().strftime('%H:%M:%S')} {line}"
            log.write(stamped)
            log.flush()
            print(stamped, end="", flush=True)
        code = process.wait()

    meta |= {"ended": datetime.now().isoformat(timespec="seconds"),
             "duration_s": round(time.perf_counter() - start, 1), "exit_code": code,
             "outputs": describe(step.get("outputs", []))}
    if code == 0:
        for rel in step.get("outputs", []):
            (folder / "data").mkdir(exist_ok=True)
            shutil.copy2(chung.ROOT / rel, folder / "data")
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_index()
    print(f"[{name}] mã thoát {code}, {meta['duration_s']} s → {folder.relative_to(chung.ROOT)}", flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("steps", nargs="*")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    unknown = [s for s in args.steps if s not in STEPS]
    if args.list or not args.steps or unknown:
        for name, step in STEPS.items():
            print(f"{name:<22} {step['title']}")
        return 2 if unknown else 0
    for name in args.steps:
        code = run_step(name)
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
