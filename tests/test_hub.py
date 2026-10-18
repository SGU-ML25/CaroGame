import json
import os
from types import SimpleNamespace

import pytest

from caro import hub
from caro.config import ConfigError, load_config

MANIFEST = """\
# chú thích phải được giữ
repo: team/caro-models
repo_type: model
models:
  l5: {file: l5/best.pt, revision: null}   # cấp 5
  l8: {file: l8/best.pt, revision: abc}
"""


@pytest.fixture
def root(tmp_path):
    (tmp_path / "models.yaml").write_text(MANIFEST)
    return tmp_path


@pytest.fixture
def fake_download(monkeypatch):
    calls = []

    def download(repo, filename, revision=None, repo_type=None):
        calls.append((repo, filename, revision, repo_type))
        return f"/cache/{repo}/{revision}/{filename}"

    monkeypatch.setattr(hub, "hf_hub_download", download)
    return calls


def test_resolve_model_ref_uses_pinned_revision(root, fake_download):
    path = hub.resolve_checkpoint("model:l8", root)
    assert fake_download == [("team/caro-models", "l8/best.pt", "abc", "model")]
    assert str(path) == "/cache/team/caro-models/abc/l8/best.pt"


def test_resolve_unpushed_and_unknown(root, fake_download):
    with pytest.raises(hub.HubError, match="chưa được push"):
        hub.resolve_checkpoint("model:l5", root)
    with pytest.raises(hub.HubError, match="không có trong"):
        hub.resolve_checkpoint("model:l9", root)
    assert fake_download == []


def test_resolve_hf_url_and_local(root, fake_download, tmp_path):
    hub.resolve_checkpoint("hf://team/caro-models/l5/best.pt@r1", root)
    assert fake_download[-1][:3] == ("team/caro-models", "l5/best.pt", "r1")
    local = tmp_path / "w.pt"
    local.write_bytes(b"x")
    assert hub.resolve_checkpoint(str(local), root) == local
    with pytest.raises(hub.HubError):
        hub.resolve_checkpoint(str(tmp_path / "missing.pt"), root)


def test_set_revision_keeps_comments(root):
    hub.set_revision("l5", "deadbeef", root)
    text = (root / "models.yaml").read_text()
    assert "l5: {file: l5/best.pt, revision: deadbeef}   # cấp 5" in text
    assert "# chú thích phải được giữ" in text and "revision: abc}" in text


def test_push_uploads_card_and_pins(root, tmp_path, monkeypatch):
    run = tmp_path / "run1"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best.pt").write_bytes(b"weights")
    (run / "config.yaml").write_text("seed: 1\n")
    (run / "meta.json").write_text(json.dumps({"seed": 1, "config_hash": "h", "git": {"commit": "g", "dirty": False}}))
    (run / "summary.json").write_text(json.dumps({"status": "ok"}))

    seen = {}

    class FakeApi:
        def create_commit(self, repo, ops, **kw):
            seen.update(repo=repo, paths=[op.path_in_repo for op in ops], **kw)
            return SimpleNamespace(oid="f00d", pr_url=None, commit_url="u")

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    hub.push("l5", run, root=root)
    assert seen["repo"] == "team/caro-models"
    assert seen["paths"] == ["l5/best.pt", "l5/card.json", "l5/config.yaml"]
    assert "git g" in seen["commit_description"]
    assert "revision: f00d}" in (root / "models.yaml").read_text()


def test_push_requires_checkpoint(root, tmp_path):
    with pytest.raises(hub.HubError, match="không có"):
        hub.push("l5", tmp_path, root=root)


def test_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# chú thích\nCARO_A='1'\nCARO_B=2\nCARO_EMPTY=\n")
    monkeypatch.setenv("CARO_B", "giữ")
    monkeypatch.delenv("CARO_A", raising=False)
    hub.load_dotenv(env)
    assert os.environ["CARO_A"] == "1" and os.environ["CARO_B"] == "giữ" and "CARO_EMPTY" not in os.environ


def test_config_rejects_unknown_model(tmp_path):
    (tmp_path / "models.yaml").write_text(MANIFEST)
    (tmp_path / "x.yaml").write_text("seed: 1\nfoo: {checkpoint: model:l9}\n")
    with pytest.raises(ConfigError, match="model:l9"):
        load_config("x", root=tmp_path)


def test_repo_configs_reference_known_models():
    cfg = load_config("experiments/main_mode_b")
    refs = {a["params"].get("checkpoint") for a in cfg["agents"]} - {None}
    assert refs == {"model:l5", "model:l6", "model:l8"}
