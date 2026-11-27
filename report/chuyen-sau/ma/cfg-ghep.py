def deep_merge(base: dict, over: dict) -> dict:
    """Ghép `over` lên `base`: dict ghép đệ quy, mọi kiểu khác (kể cả list) bị thay thế."""
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _compose(name: str, root: Path, stack: tuple[str, ...] = ()) -> dict:
    if name in stack:
        raise ConfigError("defaults bị vòng: " + " -> ".join((*stack, name)))
    data = _read(name, root)
    defaults = data.pop("defaults", []) or []
    if isinstance(defaults, str):
        defaults = [defaults]
    merged: dict = {}
    for parent in defaults:
        merged = deep_merge(merged, _compose(parent, root, (*stack, name)))
    return deep_merge(merged, data)


def parse_override(text: str) -> tuple[list[str], Any]:
    """'train.lr=3e-4' -> (['train', 'lr'], 0.0003). Giá trị đọc bằng YAML nên có kiểu."""
    key, sep, raw = text.partition("=")
    if not sep or not key.strip():
        raise ConfigError(f"override sai dạng (cần a.b=c): {text!r}")
    value = yaml.safe_load(raw) if raw.strip() else None
    if isinstance(value, str) and _SCI_FLOAT.fullmatch(value):   # YAML 1.1 đọc "3e-4" thành chuỗi
        value = float(value)
    return key.strip().split("."), value
