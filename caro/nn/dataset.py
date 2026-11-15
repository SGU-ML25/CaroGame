import numpy as np
import torch
from torch.utils.data import Dataset


def compute_zobrist_hash(board_np):
    """Tính khóa Zobrist đơn giản hoặc hash chuẩn hóa để lọc trùng lặp D4."""
    # Tìm min hash qua 8 phép biến đổi đối xứng nhóm D4
    hashes = []
    current = board_np.copy()
    for k in range(4):
        rot = np.rot90(current, k, axes=(1, 2))
        hashes.append(hash(rot.tobytes()))
        flip = np.flip(rot, axis=2)
        hashes.append(hash(flip.tobytes()))
    return min(hashes)

def get_d4_transformations(board_tensor, policy_vector):
    """Áp dụng 8 phép biến đổi nhóm đối xứng D4 cho tensor bàn cờ và vector policy."""
    augmented_samples = []
    board_np = np.array(board_tensor, dtype=np.float32)
    policy_np = np.array(policy_vector, dtype=np.float32)
    policy_grid = policy_np.reshape(11, 11)

    for k in range(4):
        rot_board = np.rot90(board_np, k, axes=(1, 2))
        rot_policy = np.rot90(policy_grid, k)
        augmented_samples.append((rot_board.copy(), rot_policy.flatten().copy()))
        
        flip_board = np.flip(rot_board, axis=2)
        flip_policy = np.flip(rot_policy, axis=1)
        augmented_samples.append((flip_board.copy(), flip_policy.flatten().copy()))
        
    return augmented_samples

class CaroDataset(Dataset):
    def __init__(self, data_records, augment=False):
        """
        data_records: danh sách các mẫu chứa state, policy, value
        augment: True nếu là tập train (nhân 8 qua D4), False cho val/test.
        """
        self.samples = []
        
        # Lọc trùng lặp bằng khóa chuẩn hóa Zobrist D4
        seen_hashes = set()
        unique_records = []
        for rec in data_records:
            state = np.array(rec["state"], dtype=np.float32)
            z_hash = compute_zobrist_hash(state)
            if z_hash not in seen_hashes:
                seen_hashes.add(z_hash)
                unique_records.append(rec)

        for rec in unique_records:
            state = np.array(rec["state"], dtype=np.float32)
            policy = np.array(rec["policy"], dtype=np.float32)
            value = rec["value"]
            
            if augment:
                d4_pairs = get_d4_transformations(state, policy)
                for aug_state, aug_policy in d4_pairs:
                    self.samples.append((aug_state, aug_policy, value))
            else:
                self.samples.append((state, policy, value))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, policy, value = self.samples[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(policy, dtype=torch.float32),
            torch.tensor(value, dtype=torch.float32)
        )