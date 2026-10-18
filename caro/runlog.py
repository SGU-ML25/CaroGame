"""Thư mục run và log thực nghiệm (mục 03.3).

Mỗi lần chạy tạo một thư mục riêng:

    runs/<kind>/<name>/<YYYYmmdd-HHMMSS>[_<tag>]/
        config.yaml     config đã ghép đầy đủ (chạy lại được bằng chính file này)
        meta.json       git commit, phiên bản Python/thư viện, phần cứng, seed, hash config
        run.log         log chữ
        games.jsonl     mỗi ván một dòng
        metrics.jsonl   loss / accuracy / score… theo step
        summary.json    trạng thái kết thúc, thời lượng, số bản ghi
        checkpoints/
    runs/<kind>/<name>/latest → run gần nhất (chỉ trên máy, không commit)
    runs/index.jsonl  sổ đăng ký trên máy: một dòng "start" và một dòng "end" cho mỗi run

Mọi file log trong thư mục run được commit lên GitHub; checkpoints/ thì không (model nằm trên Hugging Face).

Dùng:
    cfg = load_config("experiments/smoke")
    with Run(cfg) as run:
        run.log.info("bắt đầu")
        run.log_game({...})
        run.log_metric(step, loss=0.1)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Self

from caro.config import PROJECT_ROOT, config_hash, dump, project_path

GAME_FIELDS = (
    "game_id", "tournament_id", "agent_x", "agent_o", "opening_id",
    "seed", "moves", "result", "termination",
)
PER_MOVE_FIELDS = ("time_ms", "nodes", "sims")
LOG_WARN_MB = 50   # log được commit lên GitHub; GitHub từ chối file > 100 MB


def derive_seed(base_seed: int, *parts: Any) -> int:
    """seed = hash(base_seed, game_id, color, …), ổn định giữa các lần chạy (không dùng hash() của Python)."""
    key = repr((base_seed, *parts)).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") & (2**63 - 1)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5, check=True,
                             cwd=PROJECT_ROOT)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _versions(*packages: str) -> dict[str, str | None]:
    out = {}
    for p in packages:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def _gpu() -> dict | None:
    if "torch" not in sys.modules:   # không tự import torch chỉ để ghi meta
        return None
    torch = sys.modules["torch"]
    if not torch.cuda.is_available():
        return {"cuda": False}
    return {"cuda": True, "name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda}


def collect_meta(cfg: dict) -> dict:
    status = _git("status", "--porcelain")
    return {
        "started": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "seed": cfg.get("seed"),
        "config_hash": config_hash(cfg),
        "git": {"commit": _git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None},
        "python": platform.python_version(),
        "packages": _versions("caro", "numpy", "pyyaml", "torch", "fastapi"),
        "hardware": {
            "host": platform.node(),
            "os": platform.platform(),
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "ram_gb": _ram_gb(),
            "gpu": _gpu(),
        },
    }


class JsonlWriter:
    def __init__(self, path: Path, flush_every: int = 1):
        self.path = path
        self.flush_every = max(1, flush_every)
        self.count = 0
        self._f = None

    def write(self, record: dict) -> None:
        if self._f is None:   # tạo file khi có bản ghi đầu tiên
            self._f = self.path.open("a", encoding="utf-8")
        self._f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.count += 1
        if self.count % self.flush_every == 0:
            self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


class Run:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        exp, run_cfg, log_cfg = cfg["experiment"], cfg["run"], cfg["log"]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        leaf = f"{stamp}_{run_cfg['tag']}" if run_cfg.get("tag") else stamp
        parent = project_path(run_cfg["root"]) / exp["kind"] / exp["name"]
        self.dir = parent / leaf
        n = 1
        while self.dir.exists():   # hai run trong cùng một giây
            self.dir = parent / f"{leaf}-{n}"
            n += 1
        self.dir.mkdir(parents=True)
        self.id = f"{exp['kind']}/{exp['name']}/{self.dir.name}"
        self.checkpoint_dir = self.dir / cfg["checkpoint"]["dir"]
        self.checkpoint_dir.mkdir()

        self.meta = collect_meta(cfg)
        (self.dir / "config.yaml").write_text(dump(cfg), encoding="utf-8")
        (self.dir / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self._point_latest(parent)

        self.log = self._make_logger(log_cfg)
        flush = log_cfg.get("flush_every", 1)
        self.games = JsonlWriter(self.dir / log_cfg["games"], flush)
        self.metrics = JsonlWriter(self.dir / log_cfg["metrics"], flush)
        self.per_move = log_cfg.get("per_move_stats", True)
        self.index = project_path(run_cfg["index"]) if run_cfg.get("index") else None
        self._t0 = time.monotonic()
        self._closed = False
        self._index_event("start", seed=cfg.get("seed"), config_hash=self.meta["config_hash"],
                          git=self.meta["git"]["commit"], tag=run_cfg.get("tag"))
        self.log.info("run %s — config %s, seed %s", self.id, exp.get("config"), cfg.get("seed"))

    def _point_latest(self, parent: Path) -> None:
        link = parent / "latest"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(self.dir.name, target_is_directory=True)
        except OSError:   # hệ thống file không hỗ trợ symlink
            (parent / "latest.txt").write_text(self.dir.name, encoding="utf-8")

    def _make_logger(self, log_cfg: dict) -> logging.Logger:
        logger = logging.getLogger(f"caro.run.{self.id}")
        logger.setLevel(log_cfg.get("level", "INFO"))
        logger.propagate = False
        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
        handlers: list[logging.Handler] = [logging.FileHandler(self.dir / log_cfg["file"], encoding="utf-8")]
        if log_cfg.get("console", True):
            handlers.append(logging.StreamHandler())
        for h in handlers:
            h.setFormatter(fmt)
            logger.addHandler(h)
        return logger

    def _index_event(self, event: str, **fields: Any) -> None:
        if self.index is None:
            return
        self.index.parent.mkdir(parents=True, exist_ok=True)
        record = {"event": event, "run_id": self.id, "dir": str(self.dir),
                  "time": datetime.now().isoformat(timespec="seconds"), **fields}
        with self.index.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def log_game(self, record: dict) -> None:
        missing = [k for k in GAME_FIELDS if k not in record]
        if missing:
            raise ValueError(f"bản ghi ván thiếu trường {missing}")
        if not self.per_move:
            record = {k: v for k, v in record.items() if k not in PER_MOVE_FIELDS}
        self.games.write({"run_id": self.id, **record})

    def log_metric(self, step: int, **values: Any) -> None:
        self.metrics.write({"step": step, "time": round(time.monotonic() - self._t0, 3), **values})

    def checkpoint_path(self, name: str) -> Path:
        return self.checkpoint_dir / name

    def finish(self, status: str = "ok", **summary: Any) -> None:
        if self._closed:
            return
        self._closed = True
        result = {
            "status": status,
            "ended": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.monotonic() - self._t0, 3),
            "games": self.games.count,
            "metrics": self.metrics.count,
            **summary,
        }
        (self.dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        for f in self.dir.iterdir():
            if f.is_file() and f.stat().st_size > LOG_WARN_MB * 2**20:
                self.log.warning("%s nặng %.0f MB — GitHub từ chối file > 100 MB, cân nhắc tắt log.per_move_stats "
                                 "hoặc chia nhỏ", f.name, f.stat().st_size / 2**20)
        self._index_event("end", status=status, duration_s=result["duration_s"], games=self.games.count)
        self.log.info("kết thúc: %s sau %.1f s", status, result["duration_s"])
        self.games.close()
        self.metrics.close()
        for h in list(self.log.handlers):
            h.close()
            self.log.removeHandler(h)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.log.exception("run lỗi")
        self.finish("ok" if exc_type is None else ("interrupted" if exc_type is KeyboardInterrupt else "error"))
