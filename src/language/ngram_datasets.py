from pathlib import Path

import torch
from datasets import load_from_disk, Dataset
import numpy as np
from torch.utils.data import Dataset as TorchDataset


def get_ngram_datasets(vocab_size: int, ngrams: list[int], max_ds_len: int = 1024):
    source_dataset_path = 'data/val_tokenized.hf'
    val: Dataset = load_from_disk(source_dataset_path) # type: ignore
    val = val.select(range(max_ds_len))
    val.set_format("torch", columns=["input_ids"])

    ngram_data = {
        i: NgramDataset(
            Path(f'data/smoothed-{i}-gram-pile-dists-bf16-2_shards.npy'), 
            vocab_size, 
            max_ds_len
        )
        for i in ngrams
    }
    
    return {
        'val': val,
        **ngram_data
    }

class NgramDataset(TorchDataset):
    def __init__(self, datapath: Path, vocab_size: int, len_ds):
        self.dataset = np.memmap(datapath, mode='r', dtype=np.uint16, shape=(1024, 2049, vocab_size))
        self.dataset = self.dataset[:len_ds]

    def bfloat16_to_float32(self, x: np.ndarray):
        """Convert bfloat16 values represented as uint16 to float32."""
        x = np.asarray(x)
        x_float32 = np.frombuffer(
            np.left_shift(x.astype(np.uint32), 16).tobytes(), 
            dtype=np.float32
        ).reshape(x.shape)
        return x_float32
    
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


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    datasets = get_ngram_datasets(50_254, ngrams=[2, 3], max_ds_len=48)
    dataloader = DataLoader(datasets[1], batch_size=32, shuffle=False)

    for batch in dataloader:
        pass

    print("All operations completed successfully.")