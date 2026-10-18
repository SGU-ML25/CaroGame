"""Config tập trung cho mọi thực nghiệm.

Một config được ghép theo thứ tự:
    defaults (trái → phải, đệ quy) → nội dung file → override dòng lệnh "a.b.c=value"
rồi các tham chiếu tác tử ("agents/l3_alphabeta" hoặc {from: ..., id: ..., params: ...})
được giải thành cấu hình đầy đủ, kèm `limits` = budget chung ghép với budget riêng của tác tử.

Dùng thử:
    python -m caro.config experiments/smoke
    python -m caro.config experiments/a3_mcts_sims budget.max_sims=50 --hash
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = PROJECT_ROOT / "configs"

_SCI_FLOAT = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+")

# Các vị trí trong config chứa tham chiếu tác tử (danh sách hoặc một mục).
AGENT_REF_PATHS = ("agents", "baselines", "train.eval_vs", "datagen.expert")


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, over: dict) -> dict:
    """Ghép `over` lên `base`: dict ghép đệ quy, mọi kiểu khác (kể cả list) bị thay thế."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read(name: str, root: Path) -> dict:
    path = root / (name if name.endswith(".yaml") else f"{name}.yaml")
    if not path.is_file():
        raise ConfigError(f"không tìm thấy config '{name}' ({path})")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: gốc phải là mapping")
    return data


def _compose(name: str, root: Path, stack: tuple[str, ...] = ()) -> dict:
    if name in stack:
        raise ConfigError("defaults bị vòng: " + " → ".join((*stack, name)))
    data = _read(name, root)
    defaults = data.pop("defaults", []) or []
    if isinstance(defaults, str):
        defaults = [defaults]
    merged: dict = {}
    for parent in defaults:
        merged = deep_merge(merged, _compose(parent, root, (*stack, name)))
    return deep_merge(merged, data)


def parse_override(text: str) -> tuple[list[str], Any]:
    """'train.lr=3e-4' → (['train', 'lr'], 0.0003). Giá trị đọc bằng YAML nên có kiểu."""
    key, sep, raw = text.partition("=")
    if not sep or not key.strip():
        raise ConfigError(f"override sai dạng (cần a.b=c): {text!r}")
    value = yaml.safe_load(raw) if raw.strip() else None
    if isinstance(value, str) and _SCI_FLOAT.fullmatch(value):   # YAML 1.1 đọc "3e-4" thành chuỗi
        value = float(value)
    return key.strip().split("."), value


def set_path(cfg: dict, keys: list[str], value: Any) -> None:
    node = cfg
    for k in keys[:-1]:
        nxt = node.setdefault(k, {})
        if not isinstance(nxt, dict):
            raise ConfigError(f"không thể đặt {'.'.join(keys)}: '{k}' không phải mapping")
        node = nxt
    node[keys[-1]] = value


def get_path(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _resolve_agent(entry: Any, root: Path, budget: dict) -> dict:
    if isinstance(entry, str):
        entry = {"from": entry}
    if not isinstance(entry, dict):
        raise ConfigError(f"tham chiếu tác tử không hợp lệ: {entry!r}")
    if "from" in entry:
        rest = {k: v for k, v in entry.items() if k != "from"}
        agent = deep_merge(_compose(entry["from"], root), rest)
        agent["source"] = entry["from"]
    else:
        agent = copy.deepcopy(entry)
    for field in ("id", "kind"):
        if not agent.get(field):
            raise ConfigError(f"tác tử thiếu '{field}': {entry!r}")
    agent["limits"] = deep_merge(budget, agent.pop("budget", None) or {})
    return agent


def load_config(name: str, overrides: list[str] | tuple[str, ...] = (), root: Path | str = CONFIG_ROOT) -> dict:
    root = Path(root)
    cfg = _compose(name, root)
    for text in overrides:
        set_path(cfg, *parse_override(text))

    exp = cfg.setdefault("experiment", {})
    if not exp.get("name"):
        exp["name"] = Path(name).stem
    exp["config"] = name

    budget = cfg.get("budget") or {}
    for dotted in AGENT_REF_PATHS:
        refs = get_path(cfg, dotted)
        if refs is None:
            continue
        if isinstance(refs, list):
            resolved = [_resolve_agent(r, root, budget) for r in refs]
            ids = [a["id"] for a in resolved]
            dup = sorted({i for i in ids if ids.count(i) > 1})
            if dup:
                raise ConfigError(f"{dotted}: trùng id tác tử {dup}")
        else:
            resolved = _resolve_agent(refs, root, budget)
        set_path(cfg, dotted.split("."), resolved)

    if not isinstance(cfg.get("seed"), int):
        raise ConfigError("thiếu 'seed' kiểu int")
    return cfg


def project_path(path: str | Path) -> Path:
    """Đường dẫn tương đối trong config luôn tính từ gốc dự án, không phụ thuộc thư mục đang đứng."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def dump(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="In config đã ghép đầy đủ")
    ap.add_argument("name", help="tên config tương đối so với configs/, vd. experiments/smoke")
    ap.add_argument("overrides", nargs="*", help="a.b.c=value")
    ap.add_argument("--hash", action="store_true", help="chỉ in hash của config")
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.name, args.overrides)
    except ConfigError as e:
        print(f"lỗi config: {e}", file=sys.stderr)
        return 2
    print(config_hash(cfg) if args.hash else dump(cfg), end="\n" if args.hash else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
