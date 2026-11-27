class ArrayDataset(Dataset):
    """Dữ liệu dạng mảng numpy (uint8/float32), ~1 KB/mẫu; augment D4 ngẫu nhiên khi đọc.

    Mỗi lần đọc một mẫu, nếu augment=True thì chọn ngẫu nhiên 1 trong 8 phép đối xứng,
    nên qua nhiều epoch vẫn thấy đủ 8 dạng mà không phải nhân 8 bộ nhớ.
    """

    def __init__(self, states, policies, values, augment=False):
        self.states, self.policies, self.values = states, policies, values
        self.augment = augment

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        state = self.states[idx].astype(np.float32)
        policy = self.policies[idx]
        if self.augment:
            k = int(np.random.randint(4))
            flip = bool(np.random.randint(2))
            grid = np.rot90(policy.reshape(11, 11), k)
            state = np.rot90(state, k, axes=(1, 2))
            if flip:
                grid, state = np.flip(grid, axis=1), np.flip(state, axis=2)
            policy = grid.reshape(-1)
        return (
            torch.from_numpy(np.ascontiguousarray(state)),
            torch.from_numpy(np.ascontiguousarray(policy, dtype=np.float32)),
            torch.tensor(float(self.values[idx]), dtype=torch.float32),
        )
