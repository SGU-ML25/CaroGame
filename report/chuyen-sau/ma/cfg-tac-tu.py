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
