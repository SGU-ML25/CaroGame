# Chờ đủ ván 0..399 (mức 1/50/200/800) rồi dừng giải thử số mô phỏng của chính phiên này.
import json, os, signal, subprocess, time
RUN = "runs/eval/tournament/l8_sims_heldout/20261126-230159"
while True:
    ids = {json.loads(l)["game_id"] for l in open(f"{RUN}/games.jsonl") if l.endswith("\n")}
    if set(range(400)) <= ids:
        break
    time.sleep(10)
out = subprocess.run(["pgrep", "-f", "^.venv/bin/python -m caro.arena.tournament experiments/l8_sims_heldout"],
                     capture_output=True, text=True).stdout.split()
for pid in out:
    os.kill(int(pid), signal.SIGTERM)
open(f"{RUN}/GHI-CHU.txt", "w").write(
    "Dung chu dong sau khi xong game_id 0..399 (L8 1/50/200/800 mo phong). Muc 3200 khong hoan tat va khong dung "
    "trong bao cao vi thoi gian chay.\n")
print("da dung; so van trong log:", len(ids), "| tien trinh:", len(out))
