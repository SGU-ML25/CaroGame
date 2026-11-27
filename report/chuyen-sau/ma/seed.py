def derive_seed(base_seed: int, *parts: Any) -> int:
    """seed = hash(base_seed, game_id, color, ...), ổn định giữa các lần chạy (không dùng hash() của Python)."""
    key = repr((base_seed, *parts)).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") & (2**63 - 1)
