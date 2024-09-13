from typing import List

from torch.utils.data import Dataset as TorchDataset
from pathlib import Path
import numpy as np
import torch

class NgramDataset(TorchDataset):
    def __init__(self, datapath: Path, vocab_size: int, len_ds: int, order: int):
        self.dataset = np.memmap(datapath, mode='r', dtype=np.uint16, shape=(1024, 2049, vocab_size))
        self.dataset = self.dataset[:len_ds]
        self.order = order

    def bfloat16_to_float32(self, x: np.ndarray):
        """Convert bfloat16 values represented as uint16 to float32."""
        x = np.asarray(x)
        return np.frombuffer(
            np.left_shift(x.astype(np.uint32), 16).tobytes(), 
            dtype=np.float32
        ).reshape(x.shape)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.from_numpy(self.bfloat16_to_float32(self.dataset[idx]))

    def verify_dataset(self):
        print("Verifying dataset...")
        for i in range(1, len(self.dataset)):
            item_float32 = self.bfloat16_to_float32(self.dataset[i])
            assert np.exp(item_float32).sum() > 2040, f'{i} not written'
        print("Dataset verification complete.")

    def select(self, idx: List[int]):
        selected_data = self.dataset[idx]
        new_dataset = NgramDataset.__new__(NgramDataset)
        new_dataset.dataset = selected_data
        new_dataset.order = self.order
        new_dataset.len_ds = len(idx)
        return new_dataset

