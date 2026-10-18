from pathlib import Path

import pytest

from caro.config import CONFIG_ROOT, ConfigError, deep_merge, load_config, parse_override

ALL_EXPERIMENTS = sorted(p.stem for p in (CONFIG_ROOT / "experiments").glob("*.yaml"))
ALL_TRAINING = sorted(p.stem for p in (CONFIG_ROOT / "training").glob("*.yaml"))


@pytest.mark.parametrize("name", [f"experiments/{n}" for n in ALL_EXPERIMENTS]
                         + [f"training/{n}" for n in ALL_TRAINING])
def test_every_config_loads(name):
    cfg = load_config(name)
    assert cfg["engine"] == {"size": 11, "win_length": 5, "layout": "stride11"}
    assert cfg["experiment"]["kind"] in {"tournament", "bench", "train", "datagen"}
    for agent in cfg.get("agents", []) + cfg.get("baselines", []):
        assert agent["id"] and agent["kind"] and "limits" in agent


def test_deep_merge_replaces_lists_and_keeps_base():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    out = deep_merge(base, {"a": {"y": [3]}})
    assert out == {"a": {"x": 1, "y": [3]}, "b": 1}
    assert base["a"]["y"] == [1, 2]


def test_parse_override_types():
    assert parse_override("train.lr=3e-4") == (["train", "lr"], 3e-4)
    assert parse_override("data.augment_x8=false") == (["data", "augment_x8"], False)
    assert parse_override("run.tag=") == (["run", "tag"], None)
    assert parse_override("run.tag=inf_v2") == (["run", "tag"], "inf_v2")
    with pytest.raises(ConfigError):
        parse_override("no_equals")


def test_overrides_apply_before_agent_resolution():
    cfg = load_config("experiments/smoke", ["budget.max_sims=50", "seed=1"])
    assert cfg["seed"] == 1
    assert all(a["limits"]["max_sims"] == 50 for a in cfg["agents"])


def test_per_agent_budget_and_params():
    cfg = load_config("experiments/a3_mcts_sims")
    sims = {a["id"]: a["limits"]["max_sims"] for a in cfg["agents"]}
    assert sims == {"MCTS-100": 100, "MCTS-400": 400, "MCTS-1600": 1600, "MCTS-6400": 6400}
    assert all(a["limits"]["mode"] == "fixed" for a in cfg["agents"])
    assert [b["id"] for b in cfg["baselines"]] == ["L2-Rule", "L3-AlphaBeta"]

    a1 = {a["id"]: a for a in load_config("experiments/a1_ab_pruning")["agents"]}
    assert a1["AB-full"]["params"]["pruning"] is False
    assert a1["AB-full"]["params"]["radius"] == a1["AB-prune"]["params"]["radius"] == 2
    assert a1["AB-full"]["limits"]["max_depth"] == 3


def test_training_inherits_through_experiment():
    cfg = load_config("experiments/a5b_rl_init", ["train.init=random"])
    assert cfg["train"]["init"] == "random"
    assert cfg["model"]["blocks"] == 6
    assert cfg["experiment"]["name"] == "a5b_rl_init"
    assert [a["id"] for a in cfg["train"]["eval_vs"]] == ["L2-Rule", "L3-AlphaBeta"]


def test_name_defaults_to_file_stem():
    assert load_config("experiments/main_mode_b")["experiment"]["name"] == "main_mode_b"


def _write(root: Path, name: str, text: str) -> None:
    path = root / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cycle_and_duplicate_ids(tmp_path):
    _write(tmp_path, "a", "defaults: [b]\nseed: 1\n")
    _write(tmp_path, "b", "defaults: [a]\n")
    with pytest.raises(ConfigError, match="vòng"):
        load_config("a", root=tmp_path)

    _write(tmp_path, "agents/r", "id: R\nkind: random\n")
    _write(tmp_path, "dup", "seed: 1\nagents: [agents/r, agents/r]\n")
    with pytest.raises(ConfigError, match="trùng id"):
        load_config("dup", root=tmp_path)


def test_missing_config():
    with pytest.raises(ConfigError, match="không tìm thấy"):
        load_config("experiments/does_not_exist")
