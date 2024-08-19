import torch
from datasets import load_from_disk, Dataset
import numpy as np
from torch.utils.data import Dataset as TorchDataset


def get_ngram_datasets(vocab_size: int, ngrams: list[int] = [1, 2, 3], len_ds: int = 1024):
    source_dataset_path = 'data/val_tokenized.hf'
    target_dataset_paths = [
        f'test/smoothed-{i}-gram-pile-dists-bf16-2_shards.npy'
        for i in ngrams
    ]

    val: Dataset = load_from_disk(source_dataset_path) # type: ignore
    val = val.select(range(len_ds))
    val.set_format("torch", columns=["input_ids"])

    ngram_data = {
        ngram: NgramDataset(target_dataset_path, vocab_size, len_ds)
        for ngram, target_dataset_path in zip(ngrams, target_dataset_paths)
    }
    return {
        'val': val,
        **ngram_data
    }

class NgramDataset(TorchDataset):
    def __init__(self, datapath: str, vocab_size: int, len_ds):
        self.dataset = np.memmap(datapath, mode='r', dtype=np.float16, shape=(1024, 2049, vocab_size))
        self.dataset = self.dataset[:len_ds]

    def __len__(self):
        return len(self.dataset)

    def bfloat16_to_float32(self, x):
        """Convert bfloat16 values represented as uint16 to float32."""
        x = np.asarray(x)
        return np.frombuffer(
            np.left_shift(x.astype(np.uint32), 16).tobytes(), 
            dtype=np.float32
        ).reshape(x.shape)

    def __getitem__(self, idx):
        return torch.from_numpy(self.bfloat16_to_float32(self.dataset[idx]))

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    datasets = get_ngram_datasets(50_254)
    dataloader = DataLoader(datasets[1], batch_size=32, shuffle=False)

    for batch in dataloader:
        pass

    print("All operations completed successfully.")