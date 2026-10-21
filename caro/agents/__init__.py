"""Tác tử cấp 1–8 và bảng đăng ký kind → lớp.

Thêm một tác tử: viết lớp trong caro/agents/<tên>.py, thêm một dòng vào REGISTRY,
tạo configs/agents/<tên>.yaml với cùng `kind`. Lớp được import khi cần, nên cấp 1–4 không kéo theo torch.

    agent = make_agent("agents/l1_random")            # hoặc một mục đã giải trong cfg["agents"]
    agent.reset(seed)
    a = agent.select_action(game, agent.limits)
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from caro.agents.base import STATS_KEYS, Agent, BaseAgent, Deadline, Limits
from caro.config import ConfigError, resolve_agent

# kind (trong configs/agents/*.yaml) → "module:Lớp"
REGISTRY: dict[str, str] = {
    "random": "caro.agents.random_agent:RandomAgent",
}


def agent_class(kind: str) -> type:
    if kind not in REGISTRY:
        raise ConfigError(f"kind tác tử '{kind}' chưa được cài (có: {sorted(REGISTRY)})")
    module, _, cls = REGISTRY[kind].partition(":")
    return getattr(importlib.import_module(module), cls)


def make_agent(agent_cfg: dict[str, Any] | str) -> Agent:
    """Dựng tác tử từ config đã ghép: `params` → tham số khởi tạo, `limits` → Limits, `id` → name."""
    if not isinstance(agent_cfg, dict) or "limits" not in agent_cfg:
        agent_cfg = resolve_agent(agent_cfg)
    cls = agent_class(agent_cfg["kind"])
    params = dict(agent_cfg.get("params") or {})
    try:
        limits = Limits.from_dict(agent_cfg.get("limits"))
    except ValueError as e:
        raise ConfigError(f"tác tử '{agent_cfg['id']}': {e}") from None

    accepted = inspect.signature(cls).parameters
    if not any(p.kind is p.VAR_KEYWORD for p in accepted.values()):
        unknown = sorted(set(params) - set(accepted))
        if unknown:
            raise ConfigError(f"tác tử '{agent_cfg['id']}' ({cls.__name__}) không nhận tham số {unknown}")
    return cls(name=agent_cfg["id"], limits=limits, **params)


__all__ = [
    "REGISTRY",
    "STATS_KEYS",
    "Agent",
    "BaseAgent",
    "Deadline",
    "Limits",
    "agent_class",
    "make_agent",
]
