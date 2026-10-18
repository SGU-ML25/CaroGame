"""Model trên Hugging Face, con trỏ phiên bản trong GitHub (configs/models.yaml).

Trong config, checkpoint ghi được 3 dạng — mọi nơi nạp trọng số đều gọi resolve_checkpoint():
    model:l5                         → tra models.yaml, tải đúng revision đã ghim
    hf://org/repo/l5/best.pt@abc123  → tải trực tiếp (thử nghiệm nhanh)
    runs/train/.../best.pt           → file cục bộ (vừa train xong, chưa push)

Dòng lệnh (token lấy từ HF_TOKEN trong .env, hoặc từ `hf auth login`):
    caro-hub init [--private]                    tạo repo model của nhóm (một lần)
    caro-hub push l5 runs/train/sl_policy/latest [--pr]
    caro-hub pull [l5 l8 …]                      tải trước về cache (mặc định: tất cả)
    caro-hub status
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, try_to_load_from_cache

from caro.config import CONFIG_ROOT, PROJECT_ROOT, load_manifest, project_path

CARD_FIELDS = ("config_hash", "seed", "git", "python", "packages", "hardware")


class HubError(RuntimeError):
    pass


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Nạp .env (KEY=VALUE) vào môi trường; biến đã đặt sẵn (vd. từ GitHub Actions) được giữ nguyên."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#") and value.strip():
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _entry(name: str, root: Path) -> tuple[dict, dict]:
    manifest = load_manifest(root)
    if name not in manifest["models"]:
        raise HubError(f"model '{name}' không có trong models.yaml (có: {sorted(manifest['models'])})")
    return manifest, manifest["models"][name]


def resolve_checkpoint(ref: str, root: Path | str = CONFIG_ROOT) -> Path:
    root = Path(root)
    if ref.startswith("model:"):
        name = ref[len("model:"):]
        manifest, entry = _entry(name, root)
        if not entry.get("revision"):
            raise HubError(f"model '{name}' chưa được push (revision: null) — chạy `caro-hub push {name} <run>`")
        load_dotenv()
        return Path(hf_hub_download(manifest["repo"], entry["file"], revision=str(entry["revision"]),
                                    repo_type=manifest.get("repo_type", "model")))
    if ref.startswith("hf://"):
        body, _, revision = ref[len("hf://"):].partition("@")
        parts = body.split("/", 2)
        if len(parts) < 3:
            raise HubError(f"cần dạng hf://org/repo/đường/dẫn[@revision]: {ref!r}")
        load_dotenv()
        return Path(hf_hub_download(f"{parts[0]}/{parts[1]}", parts[2], revision=revision or None))
    path = project_path(ref)
    if not path.is_file():
        raise HubError(f"không tìm thấy checkpoint cục bộ: {path}")
    return path


def set_revision(name: str, revision: str, root: Path | str = CONFIG_ROOT) -> None:
    """Sửa đúng một dòng trong models.yaml, giữ nguyên chú thích."""
    path = Path(root) / "models.yaml"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^(\s*{re.escape(name)}:\s*\{{.*?\brevision:\s*)([^,}}\s]+)", re.MULTILINE)
    new, n = pattern.subn(lambda m: m.group(1) + revision, text, count=1)
    if n != 1:
        raise HubError(f"không tìm thấy dòng '{name}: {{…, revision: …}}' trong {path}")
    path.write_text(new, encoding="utf-8")


def _run_card(name: str, run_dir: Path, file: str) -> dict:
    def read(fname: str) -> dict:
        p = run_dir / fname
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    meta, summary = read("meta.json"), read("summary.json")
    runs_root = project_path("runs").resolve()
    run_id = str(run_dir.relative_to(runs_root)) if run_dir.is_relative_to(runs_root) else run_dir.name
    return {"model": name, "source_file": file, "run_id": run_id,
            **{k: meta.get(k) for k in CARD_FIELDS}, "summary": summary}


def push(name: str, run_dir: str | Path, file: str = "best.pt", create_pr: bool = False,
         message: str | None = None, root: Path | str = CONFIG_ROOT):
    root = Path(root)
    manifest, entry = _entry(name, root)
    run_dir = project_path(run_dir).resolve()   # đi theo symlink latest
    ckpt = run_dir / "checkpoints" / file
    if not ckpt.is_file():
        raise HubError(f"không có {ckpt}")

    card = _run_card(name, run_dir, file)
    if (card.get("git") or {}).get("dirty"):
        print("cảnh báo: run được chạy khi code chưa commit — khó tái lập", file=sys.stderr)
    folder = entry["file"].rsplit("/", 1)[0] if "/" in entry["file"] else ""
    prefix = f"{folder}/" if folder else ""
    ops = [CommitOperationAdd(entry["file"], str(ckpt)),
           CommitOperationAdd(f"{prefix}card.json", json.dumps(card, ensure_ascii=False, indent=2).encode())]
    if (run_dir / "config.yaml").is_file():
        ops.append(CommitOperationAdd(f"{prefix}config.yaml", str(run_dir / "config.yaml")))

    load_dotenv()
    git = (card.get("git") or {}).get("commit")
    info = HfApi().create_commit(
        manifest["repo"], ops, repo_type=manifest.get("repo_type", "model"), create_pr=create_pr,
        commit_message=message or f"{name}: {card['run_id']}",
        commit_description=f"git {git or '—'} · config {card.get('config_hash') or '—'} · seed {card.get('seed')}",
    )
    set_revision(name, info.oid, root)
    return info


def _cmd_status(root: Path) -> None:
    manifest = load_manifest(root)
    print(f"repo: {manifest.get('repo')}")
    for name, entry in manifest["models"].items():
        rev = entry.get("revision")
        cached = rev and isinstance(try_to_load_from_cache(
            manifest["repo"], entry["file"], revision=str(rev), repo_type=manifest.get("repo_type", "model")), str)
        state = "chưa push" if not rev else ("có trong cache" if cached else "chưa tải")
        print(f"  {name:<4} {entry['file']:<14} {str(rev or '—')[:10]:<10}  {state}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="caro-hub", description="Đồng bộ model với Hugging Face")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="tạo repo model trên HF (bỏ qua nếu đã có)")
    p.add_argument("--private", action="store_true")
    p = sub.add_parser("push", help="đẩy checkpoint của một run và ghim revision")
    p.add_argument("name")
    p.add_argument("run_dir")
    p.add_argument("--file", default="best.pt", help="tên file trong <run>/checkpoints/")
    p.add_argument("--pr", action="store_true", help="mở Pull Request bên HF thay vì commit thẳng")
    p.add_argument("-m", "--message")
    p = sub.add_parser("pull", help="tải trước model về cache")
    p.add_argument("names", nargs="*")
    sub.add_parser("status")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "init":
            load_dotenv()
            manifest = load_manifest()
            url = HfApi().create_repo(manifest["repo"], repo_type=manifest.get("repo_type", "model"),
                                      private=args.private, exist_ok=True)
            print(url)
        elif args.cmd == "push":
            info = push(args.name, args.run_dir, args.file, args.pr, args.message)
            print(f"đã push {args.name} @ {info.oid[:10]} — {info.pr_url or info.commit_url}")
            print("configs/models.yaml đã cập nhật: commit file này lên GitHub để cả nhóm dùng bản mới")
        elif args.cmd == "pull":
            names = args.names or list(load_manifest()["models"])
            for name in names:
                try:
                    print(f"{name}: {resolve_checkpoint(f'model:{name}')}")
                except HubError as e:
                    print(f"{name}: bỏ qua — {e}")
        else:
            _cmd_status(CONFIG_ROOT)
    except HubError as e:
        print(f"lỗi: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
