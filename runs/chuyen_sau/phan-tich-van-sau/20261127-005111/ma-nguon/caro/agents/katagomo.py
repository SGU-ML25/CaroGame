"""Đối thủ ngoài: KataGomo (KataGo cho Caro/Gomoku) qua giao thức GTP.

KataGomo là bản KataGo sửa cho Caro/Gomoku, có model 15×15 và hỗ trợ bàn 11×11. Agent này
không tự tính nước đi; nó giữ một tiến trình ``katago gtp`` và đồng bộ toàn bộ ván cờ trước
mỗi lượt. Dùng nhánh ``Caro2024`` đã vá (``configs/katagomo/freestyle-five.patch``) + model
``b10_freestyle15x``: luật freestyle (5 quân thắng, kể cả khi hai đầu bị chặn) khớp đồ án.
Caro2024 gốc coi ``OXXXXXO`` là chưa thắng; nhánh ``Gom2024`` có lỗi không nhận dãy 5 dọc,
nên cả hai đều không dùng nguyên bản:

    boardsize 11 → clear_board → play từng nước → genmove

Cách đồng bộ lại từ đầu mỗi lượt tuy chậm hơn giữ cây tìm kiếm, nhưng đúng cả khi UI lùi nước
hoặc arena chơi lại cùng một đối tượng qua nhiều ván. Toạ độ đổi như sau:

    ô (r, c), r = 0 là hàng trên cùng  ↔  cột ``ABCDEFGHJKLM…`` (bỏ I), hàng ``11 - r``

Cài đặt (một lần): xem ``docs/KATAGOMO.md`` hoặc chạy ``scripts/setup_katagomo.sh``. Mặc định
tìm engine trong ``~/.local/share/caro/katagomo``; có thể ghi đè bằng biến môi trường
``KATAGOMO_BIN``, ``KATAGOMO_MODEL``, ``KATAGOMO_CONFIG`` hoặc tham số trong config.
"""

from __future__ import annotations

import os
import re
import select
import subprocess
import threading
import time
from pathlib import Path

from caro.agents.base import BaseAgent, Limits
from caro.config import ConfigError
from caro.engine import CaroGame
from caro.engine.bitboard import N, rc

COLS = "ABCDEFGHJKLMNOPQRST"   # GTP bỏ chữ I
DEFAULT_DIR = Path.home() / ".local/share/caro/katagomo"
CHAT_RE = re.compile(r"Visits (\d+).*?Winrate ([\d.]+)%.*?Drawrate ([\d.]+)%.*?PV (.*)")


class KataGomoError(RuntimeError):
    pass


def _first(*values: str | None) -> Path | None:
    for value in values:
        if value:
            return Path(value).expanduser()
    return None


class KataGomoAgent(BaseAgent):
    """Tham số: đường dẫn engine/model/config, số lượt thăm mặc định, số luồng, timeout mỗi nước."""

    def __init__(self, name=None, limits=None, binary=None, model=None, config=None,
                 max_visits: int = 800, threads: int = 2, move_timeout: float = 120.0,
                 startup_timeout: float = 60.0):
        super().__init__(name, limits)
        self.binary = _first(binary, os.environ.get("KATAGOMO_BIN"), str(DEFAULT_DIR / "katago"))
        self.model = _first(model, os.environ.get("KATAGOMO_MODEL"),
                            str(DEFAULT_DIR / "b10_freestyle15x.bin.gz"))
        self.config = _first(config, os.environ.get("KATAGOMO_CONFIG"),
                             str(DEFAULT_DIR / "default_gtp.cfg"))
        self.max_visits = int(max_visits)
        self.threads = max(1, int(threads))
        self.move_timeout = float(move_timeout)
        self.startup_timeout = float(startup_timeout)
        self.proc: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._last_chat = ""

    # ---- vòng đời tiến trình -------------------------------------------
    def _start(self) -> None:
        missing = [str(p) for p in (self.binary, self.model, self.config) if p is None or not p.is_file()]
        if missing:
            raise ConfigError(
                "chưa cài KataGomo (thiếu " + ", ".join(missing) + "). "
                "Chạy scripts/setup_katagomo.sh, hoặc đặt KATAGOMO_BIN/KATAGOMO_MODEL/KATAGOMO_CONFIG."
            )
        cmd = [
            str(self.binary), "gtp",
            "-model", str(self.model),
            "-config", str(self.config),
            "-override-config", f"numSearchThreads={self.threads}",
            "-override-config", "logAllGTPCommunication=false",
            "-override-config", "logSearchInfo=false",
            "-override-config", "logFile=/dev/null",
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=str(self.binary.parent),
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._command("boardsize 11", self.startup_timeout)

    def _drain_stderr(self) -> None:
        """KataGomo in thông tin lượt thăm (CHAT:…) ra stderr; giữ dòng mới nhất để lấy visits/PV."""
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            if line.startswith("CHAT:"):
                self._last_chat = line.strip()

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self._command("quit", 2.0)
        except (KataGomoError, TimeoutError, OSError, ValueError):
            pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __del__(self) -> None:
        try:
            self.close()
        except (OSError, ValueError, KataGomoError, TimeoutError):
            pass

    def reset(self, seed: int) -> None:
        super().reset(seed)
        # Không tắt tiến trình: ván sau chỉ cần clear_board. Nếu tiến trình chết, _ensure khởi động lại.
        if self.proc is not None and self.proc.poll() is not None:
            self.proc = None

    # ---- GTP -----------------------------------------------------------
    def _ensure(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            self._start()

    def _command(self, cmd: str, timeout: float) -> str:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise KataGomoError("tiến trình KataGomo chưa chạy")
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise KataGomoError(f"không gửi được lệnh {cmd!r}: {e}") from e
        return self._response(timeout)

    def _response(self, timeout: float) -> str:
        assert self.proc is not None and self.proc.stdout is not None
        end = time.monotonic() + timeout
        while True:
            left = end - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"KataGomo không trả lời trong {timeout:.0f} giây")
            ready, _, _ = select.select([self.proc.stdout], [], [], left)
            if not ready:
                raise TimeoutError(f"KataGomo không trả lời trong {timeout:.0f} giây")
            line = self.proc.stdout.readline()
            if not line:
                code = self.proc.poll()
                raise KataGomoError(f"KataGomo đã thoát (mã {code})")
            line = line.strip()
            if line.startswith("CHAT:"):
                self._last_chat = line
                continue
            if line.startswith("="):
                return line[1:].strip()
            if line.startswith("?"):
                raise KataGomoError(line[1:].strip())

    # ---- toạ độ --------------------------------------------------------
    @staticmethod
    def _gtp(action: int) -> str:
        r, c = rc(action)
        return f"{COLS[c]}{N - r}"

    @staticmethod
    def _action(move: str) -> int | None:
        move = move.strip().upper()
        if not move or move in {"PASS", "RESIGN"}:
            return None
        c = COLS.find(move[0])
        try:
            row = int(move[1:])
        except ValueError:
            return None
        if c < 0 or not 1 <= row <= N:
            return None
        return (N - row) * N + c

    # ---- nước đi -------------------------------------------------------
    def select_action(self, game: CaroGame, limits: Limits | None = None) -> int:
        limits = limits or self.limits
        self._ensure()
        try:
            return self._select(game, limits)
        except (KataGomoError, TimeoutError, OSError):
            # Một lần thử lại với tiến trình mới; lỗi lặp lại thì để arena/UI báo.
            self.close()
            self._ensure()
            return self._select(game, limits)

    def _select(self, game: CaroGame, limits: Limits) -> int:
        self._command("clear_board", self.startup_timeout)
        for i, action in enumerate(game.history):
            color = "b" if i % 2 == 0 else "w"
            self._command(f"play {color} {self._gtp(action)}", self.move_timeout)

        if limits.mode == "time" and limits.time_ms:
            seconds = max(0.05, limits.time_ms / 1000.0)
            self._command(f"time_settings 0 {seconds:.3f} 1", self.move_timeout)
            self._command("kata-set-param maxVisits 10000000", self.move_timeout)
        else:
            visits = limits.max_sims or self.max_visits
            self._command("time_settings 0 3600 1", self.move_timeout)
            self._command(f"kata-set-param maxVisits {visits}", self.move_timeout)

        self._last_chat = ""
        color = "b" if game.side == 0 else "w"
        reply = self._command(f"genmove {color}", self.move_timeout)
        # CHAT được in ra stderr ngay trước dòng "="; chờ một nhịp ngắn để thread đọc kịp.
        wait_until = time.monotonic() + 0.2
        while not self._last_chat and time.monotonic() < wait_until:
            time.sleep(0.005)
        action = self._action(reply)
        if action is None or action not in game.legal_actions():
            # Không có luật pass trong Caro; nếu engine pass/resign, đi ô trống đầu tiên để ván tiếp tục.
            legal = game.legal_actions()
            if not legal:
                raise KataGomoError("bàn đã đầy nhưng ván chưa kết thúc")
            action = legal[0]

        stats: dict = {}
        match = CHAT_RE.search(self._last_chat)
        if match:
            visits = int(match.group(1))
            pv = [a for m in match.group(4).split() if (a := self._action(m)) is not None]
            stats = {"visits": visits, "sims": visits, "value": float(match.group(2)) / 100.0, "pv": pv}
        self._stats = stats
        return action
