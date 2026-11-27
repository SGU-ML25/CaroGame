def canonical_key(state):
    """Khoá chuẩn hoá D4: nhỏ nhất trong 8 phép đối xứng (state: 4x11x11 uint8)."""
    keys = []
    for k in range(4):
        rot = np.rot90(state, k, axes=(1, 2))
        keys.append(np.ascontiguousarray(rot).tobytes())
        keys.append(np.ascontiguousarray(rot[:, :, ::-1]).tobytes())
    return min(keys)


def write_final_dataset(output_file, progress_file, game_ids, dedup=True):
    """Gom file tiến độ thành .npz gọn: states uint8, policies float32, values, game_ids.

    Đọc từng ván một nên RAM chỉ cỡ kích thước mảng kết quả (~1 KB/mẫu).
    """
    states, policies, values, owners = [], [], [], []
    seen = set()
    wanted = set(game_ids)
    taken = set()
    for game_id, records in iter_progress(progress_file):
        if game_id not in wanted or game_id in taken:
            continue
        taken.add(game_id)
        for rec in records:
            state = np.asarray(rec["state"], dtype=np.uint8)
            if dedup:
                key = canonical_key(state)
                if key in seen:
                    continue
                seen.add(key)
            states.append(state)
            policies.append(np.asarray(rec["policy"], dtype=np.float32))
            values.append(rec["value"])
            owners.append(game_id)

    temp_file = output_file.with_name(output_file.stem + ".tmp.npz")
    np.savez_compressed(
        temp_file,
        states=np.stack(states) if states else np.zeros((0, 4, 11, 11), np.uint8),
        policies=np.stack(policies) if policies else np.zeros((0, 121), np.float32),
        values=np.asarray(values, dtype=np.float32),
        game_ids=np.asarray(owners, dtype=np.int32),
    )
    temp_file.replace(output_file)
    return len(values)
